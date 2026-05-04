# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ViT CUDA Graph Runner class."""

from __future__ import annotations

import inspect
import logging
from contextlib import nullcontext
from typing import Dict, Hashable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

VIT_MAX_TOKEN_NUM=31768

class ViTCudaGraphRunner:
    """Generic ViT CUDA Graph Runner.

    This runner captures the "blocks + merger + deepstack merger (optional)" part
    of a vision transformer into a CUDA graph and replays it for identical shapes.

    Optional for Qwen2.5 windowed attention:
      - vit.fullatt_block_indexes: Sequence[int]
      - run() provides both cu_seqlens and cu_window_seqlens

    Optional for Qwen3 deepstack:
      - vit.deepstack_vision_indexes: Sequence[int]
      - vit.deepstack_merger_list: nn.ModuleList (same length as deepstack_vision_indexes)
    """

    def __init__(
        self,
        vit: nn.Module,
    ) -> None:
        self.vit = vit

        # graph_key -> buffers / graphs
        self.block_input: Dict[Hashable, torch.Tensor] = {}
        self.block_ws: Dict[Hashable, torch.Tensor] = {}
        self.block_graphs: Dict[Hashable, torch.cuda.CUDAGraph] = {}
        self.block_output: Dict[Hashable, torch.Tensor] = {}

        # captured seqlens buffers (addresses must be stable for cuda-graph replay)
        self.cu_full_len: Dict[Hashable, torch.Tensor] = {}
        self.cu_window_len: Dict[Hashable, torch.Tensor] = {}
        self.cu_full_len_kk: Dict[Hashable, torch.Tensor] = {}
        self.cu_window_len_kk: Dict[Hashable, torch.Tensor] = {}

        # rotary position buffers shared across graphs
        self.sin_cos_ws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.max_context_len = getattr(vit, "max_context_len", None)

        # Qwen2.5-VL specific viarable.
        self._fullatt_block_indexes = set(getattr(vit, "fullatt_block_indexes", ()))

        # Qwen3-VL specific variables.
        self._deepstack_visual_indexes = list(
            getattr(vit, "deepstack_visual_indexes", []) or []
        )
        self._deepstack_merger_list = getattr(vit, "deepstack_merger_list", None)

        first_blk = vit.blocks[0]
        self._blk_accepts_output_ws = (
            "output_ws" in inspect.signature(first_blk.forward).parameters
        )

        self._attn: Optional[VisionAttention] = getattr(first_blk, "attn", None)
        self._attn_backend = getattr(self._attn, "qkv_backend", None)
        self.input_buffer = None

        # LFU / memory-aware eviction state
        self.graph_hits: Dict[Hashable, int] = {}
        # 记录每个 graph_key 的 block_input 是否为独立分配（seq_len > VIT_MAX_TOKEN_NUM）
        self.owns_block_input: Dict[Hashable, bool] = {}
        # 显存下限阈值（byte）。<=0 表示不启用 eviction
        self.min_free_bytes: int = (
            int(envs.SGLANG_VIT_CUDA_GRAPH_MIN_FREE_MB.get()) * 1024 * 1024
        )

    @property
    def device(self) -> torch.device:
        return self.vit.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vit.dtype

    def _ensure_sin_cos_ws(self, seq_len: int, head_dim: int, sin_cos_dtype):
        if self.sin_cos_ws is None:
            max_shape = self.max_context_len or seq_len
            max_shape = max(max_shape, seq_len)
            cos_ws = torch.empty(
                max_shape, head_dim, dtype=sin_cos_dtype, device=self.device
            )
            sin_ws = torch.empty(
                max_shape, head_dim, dtype=sin_cos_dtype, device=self.device
            )
            self.sin_cos_ws = (cos_ws, sin_ws)
        else:
            if self.sin_cos_ws[0].size(0) < seq_len:
                max_shape = max(self.sin_cos_ws[0].size(0) * 2, seq_len)
                cos_ws = torch.empty(
                    max_shape, head_dim, dtype=sin_cos_dtype, device=self.device
                )
                sin_ws = torch.empty(
                    max_shape, head_dim, dtype=sin_cos_dtype, device=self.device
                )
                self.sin_cos_ws = (cos_ws, sin_ws)

    def _get_graph_key(
        self,
        x_3d: torch.Tensor,
        grid_thw_key: Optional[Tuple] = None,
    ) -> Hashable:
        # x_3d: [S, B, H]。仅凭 seq_len 不足以唯一确定 cu_seqlens / cu_window_seqlens，
        # 因此将上层传入的 grid_thw 内容一并纳入 key。
        # grid_thw 完全决定了 cu_seqlens 与 cu_window_seqlens（包括 Qwen2.5-VL window 划分），
        # 所以 (seq_len, grid_thw_tuple) 足够。
        if grid_thw_key is not None:
            return (x_3d.shape[0], grid_thw_key)
        return x_3d.shape[0]

    def _cuda_free_bytes(self) -> int:
        """查询当前设备上 GPU 的空闲显存（byte）。"""
        try:
            free, _total = torch.cuda.mem_get_info(self.device)
            return int(free)
        except Exception:
            return -1

    def _evict_one(self, key: Hashable) -> None:
        """删除一个 graph 及其所有关联 buffer。"""
        # graph 本体与输出
        self.block_graphs.pop(key, None)
        self.block_output.pop(key, None)
        # cu_* 系列
        self.cu_full_len.pop(key, None)
        self.cu_full_len_kk.pop(key, None)
        self.cu_window_len.pop(key, None)
        self.cu_window_len_kk.pop(key, None)
        # block_input：仅当独立分配（seq_len > VIT_MAX_TOKEN_NUM）时才释放显存；
        # 否则只 pop key，共享的 input_buffer 不动。
        self.block_input.pop(key, None)
        self.owns_block_input.pop(key, None)
        self.graph_hits.pop(key, None)

    def _evict_until_free(self) -> None:
        """按 hit 次数最少驱逐，直到剩余显存 >= min_free_bytes 或无 graph 可驱逐。"""
        if self.min_free_bytes <= 0:
            return
        while self.block_graphs:
            free = self._cuda_free_bytes()
            if free < 0 or free >= self.min_free_bytes:
                return
            if not self.graph_hits:
                # 理论上不应该发生：graph 存在但计数缺失
                victim = next(iter(self.block_graphs))
            else:
                victim = min(self.graph_hits, key=self.graph_hits.get)
            logger.warning(
                "[ViTCudaGraphRunner] Evicting graph key=%s (hits=%d), "
                "cuda free=%.1fMB < min_free=%.1fMB",
                victim,
                self.graph_hits.get(victim, 0),
                free / 1024 / 1024,
                self.min_free_bytes / 1024 / 1024,
            )
            self._evict_one(victim)
            torch.cuda.empty_cache()
        # 走到这里说明 block_graphs 已空但显存仍不足
        free = self._cuda_free_bytes()
        if free >= 0 and free < self.min_free_bytes:
            logger.warning(
                "[ViTCudaGraphRunner] All graphs evicted but cuda free=%.1fMB "
                "still below min_free=%.1fMB",
                free / 1024 / 1024,
                self.min_free_bytes / 1024 / 1024,
            )

    def _create_graph(
        self,
        graph_key: int,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (cos, sin), [S, D]
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
    ):

        graph = torch.cuda.CUDAGraph()
        vit = self.vit

        # Qwen2.5-VL
        if self._fullatt_block_indexes:
            cu_window = self.cu_window_len[graph_key]
            cu_window_kk = self.cu_window_len_kk[graph_key]
            max_window_len = int(cu_window_kk.max().item())

        cu_full = self.cu_full_len[graph_key]
        cu_full_kk = self.cu_full_len_kk[graph_key]
        max_full_len = int(cu_full_kk.max().item())

        override_backend = get_global_server_args().mm_attention_backend

        tp_group = get_tp_group()
        ca_comm = tp_group.ca_comm
        capture_ctx = ca_comm.capture() if ca_comm is not None else nullcontext()

        with capture_ctx, torch.cuda.graph(graph):
            y = None
            deepstack_outs: List[torch.Tensor] = []
            deepstack_capture_idx = 0

            for layer_num, blk in enumerate(vit.blocks):
                if self._fullatt_block_indexes:
                    if layer_num in vit.fullatt_block_indexes:
                        cu_seqlens_now = cu_full
                        cu_seqlens_kk_now = cu_full_kk
                        max_len = max_full_len
                    else:
                        cu_seqlens_now = cu_window
                        cu_seqlens_kk_now = cu_window_kk
                        max_len = max_window_len
                else:
                    cu_seqlens_now = cu_full
                    cu_seqlens_kk_now = cu_full_kk
                    max_len = max_full_len

                if override_backend == "triton_attn":
                    cu_seq_len_ws = [cu_seqlens_now, cu_seqlens_kk_now, max_len]
                elif override_backend == "fa3":
                    cu_seq_len_ws = [cu_seqlens_now, max_len]
                else:
                    raise RuntimeError("Not supported ViT attention backend")

                if position_embeddings is not None:
                    if layer_num == 0:
                        y = blk(
                            self.block_input[graph_key],
                            cu_seqlens=cu_seq_len_ws,
                            position_embeddings=position_embeddings,
                        )
                    else:
                        y = blk(
                            y,
                            cu_seqlens=cu_seq_len_ws,
                            position_embeddings=position_embeddings,
                        )
                elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
                    if layer_num == 0:
                        y = blk(
                            self.block_input[graph_key],
                            cu_seqlens=cu_seq_len_ws,
                            rotary_pos_emb_cos=rotary_pos_emb_cos,
                            rotary_pos_emb_sin=rotary_pos_emb_sin,
                        )
                    else:
                        y = blk(
                            y,
                            cu_seqlens=cu_seq_len_ws,
                            rotary_pos_emb_cos=rotary_pos_emb_cos,
                            rotary_pos_emb_sin=rotary_pos_emb_sin,
                        )

                # Optional deepstack support (Qwen3-VL)
                if (
                    self._deepstack_visual_indexes
                    and layer_num in self._deepstack_visual_indexes
                ):
                    if self._deepstack_merger_list is None:
                        raise RuntimeError(
                            "deepstack_visual_indexes exists but deepstack_merger_list is missing."
                        )
                    deepstack_out = self._deepstack_merger_list[deepstack_capture_idx](
                        y
                    )
                    deepstack_outs.append(deepstack_out)
                    deepstack_capture_idx += 1

            main_out = vit.merger(y)

            if deepstack_outs:
                self.block_output[graph_key] = torch.cat(
                    [main_out] + deepstack_outs, dim=1
                )
            else:
                self.block_output[graph_key] = main_out

        self.block_graphs[graph_key] = graph

    def create_graph(
        self,
        x_3d: torch.Tensor,  # [S, 1, H]
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ],  # (cos, sin), [S, D]
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        grid_thw_key: Optional[Tuple] = None,
    ) -> Hashable:
        vit = self.vit
        graph_key = self._get_graph_key(x_3d, grid_thw_key)
        seq_len = x_3d.shape[0]

        if graph_key in self.block_graphs:
            return graph_key

        # 分配新 graph 前，如启用显存阈值，按 LFU 驱逐旧 graph
        self._evict_until_free()

        # pre-allocate workspace
        attn_module: VisionAttention = vit.blocks[0].attn
        num_heads = attn_module.num_attention_heads_per_partition
        attn_head_dim = attn_module.head_size

        if graph_key not in self.block_output:
            if self.input_buffer is None:
                preallocate_buffer_shape = [VIT_MAX_TOKEN_NUM, x_3d.shape[1], x_3d.shape[2]]
                self.input_buffer = torch.empty(preallocate_buffer_shape, dtype = x_3d.dtype, device = self.device).contiguous()
                
            # only when shape > max_token_num case use new buffer
            if seq_len > VIT_MAX_TOKEN_NUM:
                self.block_input[graph_key] = torch.empty(
                    seq_len,
                    num_heads,
                    attn_head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.owns_block_input[graph_key] = True
            else:
                self.block_input[graph_key] = self.input_buffer[:seq_len, :, :]
                self.owns_block_input[graph_key] = False
            
        # Qwen2.5-VL
        if self._fullatt_block_indexes:
            if graph_key not in self.cu_window_len:
                self.cu_window_len[graph_key] = cu_window_seqlens
                self.cu_full_len[graph_key] = cu_seqlens
                self.cu_window_len_kk[graph_key] = (
                    cu_window_seqlens[1:] - cu_window_seqlens[:-1]
                )
                self.cu_full_len_kk[graph_key] = cu_seqlens[1:] - cu_seqlens[:-1]
        else:
            if graph_key not in self.cu_full_len:
                self.cu_full_len[graph_key] = cu_seqlens
                self.cu_full_len_kk[graph_key] = cu_seqlens[1:] - cu_seqlens[:-1]

        if position_embeddings is not None:
            # make sure rotary workspace
            head_dim = position_embeddings[0].shape[1]
            sin_cos_dtype = position_embeddings[0].dtype
            self._ensure_sin_cos_ws(seq_len, head_dim, sin_cos_dtype)

            used_cos_ws = self.sin_cos_ws[0][:seq_len, :]
            used_sin_ws = self.sin_cos_ws[1][:seq_len, :]
            used_cos_ws.copy_(position_embeddings[0])
            used_sin_ws.copy_(position_embeddings[1])
            persist_position_embeddings = (used_cos_ws, used_sin_ws)
            self._create_graph(
                graph_key=graph_key, position_embeddings=persist_position_embeddings
            )
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            # make sure rotary workspace
            head_dim = rotary_pos_emb_cos.shape[1]
            sin_cos_dtype = rotary_pos_emb_cos.dtype
            self._ensure_sin_cos_ws(seq_len, head_dim, sin_cos_dtype)

            used_cos_ws = self.sin_cos_ws[0][:seq_len, :]
            used_sin_ws = self.sin_cos_ws[1][:seq_len, :]
            used_cos_ws.copy_(rotary_pos_emb_cos)
            used_sin_ws.copy_(rotary_pos_emb_sin)
            self._create_graph(
                graph_key=graph_key,
                position_embeddings=None,
                rotary_pos_emb_cos=used_cos_ws,
                rotary_pos_emb_sin=used_sin_ws,
            )

        # 新建 graph 初始 hit 计为 1，避免刚建好就被下一轮驱逐。
        self.graph_hits[graph_key] = 1

        return graph_key

    def replay(
        self,
        graph_key: Hashable,
        x_3d: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = x_3d.shape[0]

        if position_embeddings is not None:
            # update rotary workspace content
            head_dim = position_embeddings[0].shape[1]
            sin_cos_dtype = position_embeddings[0].dtype
            self._ensure_sin_cos_ws(seq_len, head_dim, sin_cos_dtype)
            used_cos_ws = self.sin_cos_ws[0][:seq_len, :]
            used_sin_ws = self.sin_cos_ws[1][:seq_len, :]
            used_cos_ws.copy_(position_embeddings[0])
            used_sin_ws.copy_(position_embeddings[1])
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            # update rotary workspace content
            head_dim = rotary_pos_emb_cos.shape[1]
            sin_cos_dtype = rotary_pos_emb_cos.dtype
            self._ensure_sin_cos_ws(seq_len, head_dim, sin_cos_dtype)
            used_cos_ws = self.sin_cos_ws[0][:seq_len, :]
            used_sin_ws = self.sin_cos_ws[1][:seq_len, :]
            used_cos_ws.copy_(rotary_pos_emb_cos)
            used_sin_ws.copy_(rotary_pos_emb_sin)

        # 累加 hit 次数（LFU 驱逐依据）
        self.graph_hits[graph_key] = self.graph_hits.get(graph_key, 0) + 1

        # copy input
        self.block_input[graph_key].copy_(x_3d)

        # replay
        self.block_graphs[graph_key].replay()

        out = self.block_output[graph_key]

        # Optional output reordering (Qwen2.5-VL window permutation inverse)
        if output_indices is not None:
            out = out.index_select(0, output_indices)

        return out

    def run(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
        grid_thw_key: Optional[Tuple] = None,
    ) -> torch.Tensor:
        # x: [seq_len, hidden] -> [S, B=1, H]
        x_3d = x.unsqueeze(1)
        graph_key = self._get_graph_key(x_3d, grid_thw_key)

        if graph_key not in self.block_graphs:
            self.create_graph(
                x_3d=x_3d,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                cu_window_seqlens=cu_window_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                grid_thw_key=grid_thw_key,
            )

        return self.replay(
            graph_key=graph_key,
            x_3d=x_3d,
            position_embeddings=position_embeddings,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            output_indices=output_indices,
        )
