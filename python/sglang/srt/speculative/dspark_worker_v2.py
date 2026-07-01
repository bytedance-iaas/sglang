import logging
from copy import deepcopy
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus
from sglang.srt.speculative.dspark_info import (
    DSparkDraftBlockInput,
    DSparkDraftInputV2,
    DSparkVerifyInput,
)
from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.speculative.triton_ops.dspark import (
    _compute_dspark_accept_bonus_triton_unchecked,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.common import empty_context

logger = logging.getLogger(__name__)


class DSparkWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device

        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        with (
            empty_context(),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._draft_worker = TpModelWorker(
                server_args=draft_server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                pp_rank=0,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                dp_rank=dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self._draft_worker.model_runner
        self._draft_worker.draft_runner = self.draft_model_runner
        self.draft_model = self.draft_model_runner.model
        self._draft_inner = self.draft_model.model
        self.dspark_target_layer_ids = list(self._draft_inner.target_layer_ids)
        self.expected_main_hidden_dim = int(self._draft_inner.main_proj.input_size)
        self._ensure_target_dspark_capture()

        self.draft_model.model.embed_tokens.weight = (
            self.target_worker.model_runner.model.model.embed_tokens.weight
        )
        self.draft_model.lm_head.weight = (
            self.target_worker.model_runner.model.lm_head.weight
        )

        self.block_size = int(server_args.speculative_num_draft_tokens)
        model_block_size = int(getattr(self.draft_model, "block_size", self.block_size))
        if model_block_size != self.block_size:
            logger.warning(
                "DSpark block size mismatch: using speculative_num_draft_tokens=%s "
                "but draft model block_size=%s.",
                self.block_size,
                model_block_size,
            )
        self.speculative_num_draft_tokens = int(self.block_size)

        self.noise_token_id = int(self._draft_inner.noise_token_id)
        self.markov_rank = int(self._draft_inner.markov_rank)
        self.num_dspark_layers = int(self.draft_model.num_dspark_layers)
        self.confidence_threshold = float(
            server_args.speculative_dspark_confidence_threshold
        )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self._use_triton_accept_bonus = is_cuda() or is_hip()
        self._accept_bonus_buffer_cap: int = 0
        self._accept_bonus_buffer_slot: int = 0
        self._commit_lens_bufs: List[torch.Tensor] = []
        self._bonus_id_bufs: List[torch.Tensor] = []
        self._out_tokens_bufs: List[torch.Tensor] = []
        self._new_seq_lens_bufs: List[torch.Tensor] = []
        self._markov_refine_buffer_cap: int = 0
        self._markov_candidates_buf: Optional[torch.Tensor] = None
        self._markov_embeds_buf: Optional[torch.Tensor] = None
        self._stacked_wqkv_fp8_proj = None
        self._stacked_wqkv_kv_offsets: list[tuple[int, int]] = []
        self._stacked_wqkv_out_sizes: list[int] = []
        self._init_fp8_wqkv_stack()

        if self.tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. model=%s, block_size=%s, "
                "num_dspark_layers=%s, noise_token_id=%s, markov_rank=%s, "
                "confidence_threshold=%s",
                self.draft_model.__class__.__name__,
                self.block_size,
                self.num_dspark_layers,
                self.noise_token_id,
                self.markov_rank,
                self.confidence_threshold,
            )

    def _ensure_target_dspark_capture(self) -> None:
        target_model = self.target_worker.model_runner.model
        if hasattr(target_model, "set_dspark_layers_to_capture"):
            target_model.set_dspark_layers_to_capture(self.dspark_target_layer_ids)
            target_inner_model = getattr(target_model, "model", None)
            capture_aux_hidden_states = getattr(
                target_model, "capture_aux_hidden_states", None
            )
            layers_to_capture = (
                getattr(target_inner_model, "layers_to_capture", None)
                if target_inner_model is not None
                else None
            )
            if capture_aux_hidden_states is False or list(
                layers_to_capture or []
            ) != list(self.dspark_target_layer_ids):
                raise RuntimeError(
                    "DSpark failed to enable target aux hidden capture: "
                    f"capture_aux_hidden_states={capture_aux_hidden_states}, "
                    f"layers_to_capture={layers_to_capture}, expected "
                    f"{self.dspark_target_layer_ids}."
                )
            return
        raise RuntimeError(
            f"Target model {target_model.__class__.__name__} does not support "
            "DSpark aux hidden capture."
        )

    @staticmethod
    def _pop_hidden_states(logits_output, context: str) -> Tuple[torch.Tensor, object]:
        if isinstance(logits_output, torch.Tensor):
            return logits_output, LogitsProcessorOutput(next_token_logits=None)

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                f"DSpark requires target hidden states for {context}, but got None."
            )
        logits_output.hidden_states = None
        return hidden, logits_output

    @staticmethod
    def _tensor_to_logits_output(
        tensor: torch.Tensor,
        forward_batch: ForwardBatch,
        target_model,
        context: str,
    ) -> LogitsProcessorOutput:
        vocab_size = getattr(getattr(target_model, "config", None), "vocab_size", None)
        if tensor.dim() >= 2 and vocab_size is not None and tensor.shape[-1] == vocab_size:
            return LogitsProcessorOutput(next_token_logits=tensor)

        if not hasattr(target_model, "logits_processor") or not hasattr(
            target_model, "lm_head"
        ):
            raise RuntimeError(
                f"DSpark {context} got a Tensor output, but target model does not "
                "expose logits_processor/lm_head to convert hidden states to logits."
            )

        if tensor.shape[0] == forward_batch.batch_size:
            logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
            next_token_logits = target_model.logits_processor._get_logits(
                tensor,
                target_model.lm_head,
                logits_metadata,
            )
            return LogitsProcessorOutput(next_token_logits=next_token_logits)

        if tensor.shape[0] == forward_batch.input_ids.shape[0]:
            return target_model.logits_processor(
                forward_batch.input_ids,
                tensor,
                target_model.lm_head,
                forward_batch,
            )

        raise RuntimeError(
            f"DSpark {context} got unsupported Tensor output shape "
            f"{tuple(tensor.shape)} for batch_size={forward_batch.batch_size} "
            f"and num_input_tokens={forward_batch.input_ids.shape[0]}."
        )

    def _init_fp8_wqkv_stack(self) -> None:
        if not envs.SGLANG_DSPARK_FP8_WQKV_STACK.get():
            return

        layers = getattr(self._draft_inner, "layers", None)
        if not layers:
            return

        weights = []
        scales = []
        biases = []
        kv_offsets = []
        out_sizes = []
        first_proj = None
        scale_name = None
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                self._log_fp8_wqkv_stack_disabled("missing self_attn")
                return

            if getattr(attn, "fuse_wqa_wkv", False):
                proj = getattr(attn, "wqkv_a", None)
                kv_start = int(attn.q_lora_rank)
                kv_end = kv_start + int(attn.head_dim)
            else:
                proj = getattr(attn, "wkv", None)
                kv_start = 0
                kv_end = int(attn.head_dim)

            cur_scale_name, scale = self._get_fp8_wqkv_scale(proj)
            if (
                proj is None
                or not hasattr(proj, "weight")
                or proj.weight.dtype
                not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                or scale is None
                or proj.weight.dim() != 2
                or scale.dim() != 2
                or kv_end > proj.weight.shape[0]
                or getattr(proj, "skip_bias_add", False)
            ):
                reason = (
                    "unsupported wqkv FP8 layout: "
                    f"weight_dtype={getattr(getattr(proj, 'weight', None), 'dtype', None)}, "
                    f"weight_shape={tuple(proj.weight.shape) if hasattr(proj, 'weight') else None}, "
                    f"scale_dtype={getattr(scale, 'dtype', None)}, "
                    f"scale_shape={tuple(scale.shape) if scale is not None else None}"
                )
                self._log_fp8_wqkv_stack_disabled(reason)
                return

            if first_proj is None:
                first_proj = proj
                scale_name = cur_scale_name
            elif (
                proj.weight.shape[1] != first_proj.weight.shape[1]
                or proj.weight.dtype != first_proj.weight.dtype
                or proj.quant_method.__class__ is not first_proj.quant_method.__class__
                or cur_scale_name != scale_name
                or scale.shape[1:] != scales[0].shape[1:]
            ):
                self._log_fp8_wqkv_stack_disabled("mixed wqkv FP8 layouts")
                return

            weights.append(proj.weight.detach())
            scales.append(scale.detach())
            kv_offsets.append((kv_start, kv_end))
            out_sizes.append(int(proj.weight.shape[0]))

            bias = getattr(proj, "bias", None)
            if bias is not None:
                biases.append(bias.detach())
            else:
                biases.append(None)

        if not weights:
            return

        has_bias = [bias is not None for bias in biases]
        if any(has_bias) and not all(has_bias):
            self._log_fp8_wqkv_stack_disabled("mixed wqkv bias layout")
            return

        stacked_weight = torch.cat(weights, dim=0).contiguous()
        stacked_scale = torch.cat(scales, dim=0).contiguous()
        stacked_bias = torch.cat(biases, dim=0).contiguous() if all(has_bias) else None

        stacked_proj = SimpleNamespace()
        for name in (
            "input_size",
            "input_size_per_partition",
            "params_dtype",
            "quant_config",
            "quant_method",
            "skip_bias_add",
            "weight_block_size",
        ):
            if hasattr(first_proj, name):
                setattr(stacked_proj, name, getattr(first_proj, name))
        stacked_proj.output_size = int(stacked_weight.shape[0])
        stacked_proj.output_size_per_partition = int(stacked_weight.shape[0])
        stacked_proj.weight = stacked_weight
        stacked_proj.bias = stacked_bias
        setattr(stacked_proj, scale_name, stacked_scale)

        self._stacked_wqkv_fp8_proj = stacked_proj
        self._stacked_wqkv_kv_offsets = kv_offsets
        self._stacked_wqkv_out_sizes = out_sizes

        if self.tp_rank == 0:
            logger.info(
                "Enabled DSpark FP8 wqkv stack. layers=%s, weight_shape=%s, "
                "scale_shape=%s, scale_name=%s, kv_offsets=%s, "
                "env=SGLANG_DSPARK_FP8_WQKV_STACK",
                len(weights),
                tuple(stacked_weight.shape),
                tuple(stacked_scale.shape),
                scale_name,
                self._stacked_wqkv_kv_offsets,
            )

    def _log_fp8_wqkv_stack_disabled(self, reason: str) -> None:
        if self.tp_rank == 0:
            logger.warning("DSpark FP8 wqkv stack disabled: %s", reason)

    @staticmethod
    def _get_fp8_wqkv_scale(proj) -> tuple[Optional[str], Optional[torch.Tensor]]:
        if proj is None:
            return None, None
        for name in ("weight_scale_inv", "weight_scale", "scale"):
            scale = getattr(proj, name, None)
            if scale is None:
                continue
            dtype = getattr(scale, "dtype", None)
            if dtype in (
                torch.uint8,
                torch.float32,
                torch.bfloat16,
                torch.float16,
                getattr(torch, "float8_e8m0fnu", None),
            ):
                return name, scale
        return None, None

    def _get_dp_decode_global_num_tokens(
        self, batch: ScheduleBatch
    ) -> Optional[list[int]]:
        if not self.server_args.enable_dp_attention or batch.global_num_tokens is None:
            return None

        global_num_tokens = [int(x) for x in batch.global_num_tokens]
        if any(x > 0 for x in global_num_tokens):
            return [max(1, x) for x in global_num_tokens]
        return global_num_tokens

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._draft_worker.init_cuda_graphs()

    def clear_cache_pool(self):
        pass

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _materialize_main_hidden_to_draft_kv(
        self,
        *,
        main_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if main_hidden is None:
            raise RuntimeError("DSpark missing target main_hidden context features.")
        if main_hidden.numel() == 0:
            return

        device = self.device
        if main_hidden.device != device:
            main_hidden = main_hidden.to(device, non_blocking=True)
        if cache_loc.device != device:
            cache_loc = cache_loc.to(device, non_blocking=True)
        if positions.device != device:
            positions = positions.to(device, non_blocking=True)
        if cache_loc.dtype != torch.int64:
            cache_loc = cache_loc.to(torch.int64)
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)

        if main_hidden.shape[-1] != self.expected_main_hidden_dim:
            raise RuntimeError(
                "DSpark target hidden dim mismatch: "
                f"got {main_hidden.shape[-1]}, expected {self.expected_main_hidden_dim} "
                f"from target layers {self.dspark_target_layer_ids}. This usually "
                "means target aux hidden capture returned pre-hc or final hidden "
                "states instead of DSpark target layer hidden states."
            )

        attn_backend = self.draft_model_runner.attn_backend
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            torch.inference_mode(),
        ):
            main_x = self.draft_model.project_main_hidden(main_hidden)
            if self._stacked_wqkv_fp8_proj is None:
                for layer in self._draft_inner.layers:
                    layer.self_attn.kv_from_hidden(
                        main_x, positions, cache_loc, attn_backend
                    )
            else:
                stacked_out = self._stacked_wqkv_fp8_proj.quant_method.apply(
                    self._stacked_wqkv_fp8_proj,
                    main_x,
                    self._stacked_wqkv_fp8_proj.bias,
                )
                layer_outputs = torch.split(
                    stacked_out, self._stacked_wqkv_out_sizes, dim=-1
                )
                for layer_idx, layer in enumerate(self._draft_inner.layers):
                    kv_start, kv_end = self._stacked_wqkv_kv_offsets[layer_idx]
                    self._write_draft_kv_from_projected_kv(
                        attn=layer.self_attn,
                        kv=layer_outputs[layer_idx][..., kv_start:kv_end],
                        positions=positions,
                        cache_loc=cache_loc,
                        attn_backend=attn_backend,
                    )

    def _write_draft_kv_from_projected_kv(
        self,
        *,
        attn,
        kv: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        attn_backend,
    ) -> None:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(cache_loc).to(
            torch.int32
        )
        token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=attn.layer_id,
            swa_loc=swa_loc,
            kv=kv,
            kv_weight=attn.kv_norm.weight.data,
            eps=attn.eps,
            freqs_cis=attn.freqs_cis,
            positions=positions,
        )

    def _run_draft_block(
        self,
        *,
        batch: ScheduleBatch,
        bs: int,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        verify_out_cache_loc: torch.Tensor,
        dp_decode_global_num_tokens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        draft_block_spec_info = DSparkDraftBlockInput(
            draft_token=block_ids.reshape(-1),
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        draft_forward_batch = draft_block_spec_info.prepare_for_draft_block(
            batch=batch,
            draft_model_runner=self.draft_model_runner,
            out_cache_loc=verify_out_cache_loc,
            dp_decode_global_num_tokens=dp_decode_global_num_tokens,
        )

        from sglang.srt.layers.attention import deepseek_v4_backend as _dsv4_be

        _dsv4_be._DSPARK_BLOCK_FULL_ATTN = int(self.block_size)
        try:
            with torch.inference_mode():
                draft_runner_out = self.draft_model_runner.forward(draft_forward_batch)
        finally:
            _dsv4_be._DSPARK_BLOCK_FULL_ATTN = 0

        raw = draft_runner_out.logits_output
        block_hidden = raw if isinstance(raw, torch.Tensor) else raw.hidden_states
        if block_hidden is None:
            raise RuntimeError("DSpark draft model returned no block hidden states.")
        reshape_bs = bs
        keep_tokens = bs * int(self.block_size)
        if bs == 0 and dp_decode_global_num_tokens is not None:
            reshape_bs = 1
            keep_tokens = int(self.block_size)
            if block_hidden.numel() == 0:
                block_hidden = block_hidden.new_zeros(
                    int(self.block_size), self._draft_inner.hidden_size
                )
        block_hidden = block_hidden[:keep_tokens]
        return block_hidden.reshape(
            reshape_bs, int(self.block_size), block_hidden.shape[-1]
        )

    def _ensure_markov_refine_buffers(self, bs: int, device: torch.device) -> None:
        cap = self._markov_refine_buffer_cap
        if (
            cap >= int(bs)
            and self._markov_candidates_buf is not None
            and self._markov_embeds_buf is not None
            and self._markov_candidates_buf.device == device
            and self._markov_embeds_buf.device == device
        ):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        markov_weight = getattr(self._draft_inner.markov_head.markov_w1, "weight", None)
        markov_dtype = (
            markov_weight.dtype
            if markov_weight is not None
            else self.draft_model.lm_head.weight.dtype
        )
        self._markov_candidates_buf = torch.empty(
            (new_cap, int(self.block_size)), dtype=torch.int64, device=device
        )
        self._markov_embeds_buf = torch.empty(
            (new_cap, int(self.block_size), int(self.markov_rank)),
            dtype=markov_dtype,
            device=device,
        )
        self._markov_refine_buffer_cap = new_cap

    def _refine_block_markov(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        output_bs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = int(block_hidden.shape[0])
        output_bs = bs if output_bs is None else int(output_bs)
        block_size = int(self.block_size)
        if bs == 0:
            empty_tokens = torch.empty(
                (output_bs, block_size), dtype=torch.int64, device=block_hidden.device
            )
            empty_confidence = torch.empty(
                (output_bs, block_size), dtype=torch.float32, device=block_hidden.device
            )
            return empty_tokens, empty_confidence

        self._ensure_markov_refine_buffers(bs, block_hidden.device)
        assert self._markov_candidates_buf is not None
        assert self._markov_embeds_buf is not None
        candidates = self._markov_candidates_buf[:bs]
        markov_embeds = self._markov_embeds_buf[:bs]

        markov_head = self._draft_inner.markov_head
        confidence_head = self._draft_inner.confidence_head
        lm_head = self.draft_model.lm_head

        tp_size = get_tensor_model_parallel_world_size()
        vocab_size = int(self._draft_inner.vocab_size)

        def _gather_full_vocab(logits_shard: torch.Tensor) -> torch.Tensor:
            if logits_shard.shape[-1] >= vocab_size:
                return logits_shard[..., :vocab_size]
            if tp_size == 1:
                return logits_shard
            return tensor_model_parallel_all_gather(logits_shard, dim=-1)[
                ..., :vocab_size
            ]

        if bonus_tokens.numel() == bs:
            first_tokens = bonus_tokens.view(-1).to(torch.int64)
        else:
            first_tokens = torch.full(
                (bs,), self.noise_token_id, dtype=torch.int64, device=block_hidden.device
            )
        candidates[:, 0].copy_(first_tokens)

        with torch.inference_mode():
            base_logits = _gather_full_vocab(F.linear(block_hidden, lm_head.weight))
            prev_tokens = candidates[:, 0]
            for i in range(block_size):
                prev_embed = markov_head.get_prev_embeddings(prev_tokens)
                markov_embeds[:, i].copy_(prev_embed)
                bias = _gather_full_vocab(markov_head.project_bias(prev_embed))
                bias.add_(base_logits[:, i])
                next_tokens = torch.argmax(bias, dim=-1)
                if i + 1 < block_size:
                    candidates[:, i + 1].copy_(next_tokens)
                prev_tokens = next_tokens

            confidence = confidence_head(block_hidden, markov_embeds)

        return candidates[:output_bs], confidence[:output_bs]

    def _confident_prefix(self, confidence: torch.Tensor) -> torch.Tensor:
        keep = torch.sigmoid(confidence) >= self.confidence_threshold
        return keep.to(torch.int32).cumprod(dim=1).sum(dim=1)

    def _ensure_accept_bonus_buffers(self, bs: int) -> None:
        if self._accept_bonus_buffer_cap >= int(bs):
            return

        new_cap = max(
            int(bs),
            (
                self._accept_bonus_buffer_cap * 2
                if self._accept_bonus_buffer_cap > 0
                else int(bs)
            ),
        )
        device = self.device
        block_size = int(self.block_size)
        self._commit_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._bonus_id_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._out_tokens_bufs = [
            torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
            for _ in range(2)
        ]
        self._new_seq_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._accept_bonus_buffer_cap = new_cap

    def _next_accept_bonus_buffers(self, bs: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._ensure_accept_bonus_buffers(bs)
        slot = self._accept_bonus_buffer_slot
        self._accept_bonus_buffer_slot = (slot + 1) % 2
        return (
            self._commit_lens_bufs[slot][:bs],
            self._bonus_id_bufs[slot][:bs],
            self._out_tokens_bufs[slot][:bs],
            self._new_seq_lens_bufs[slot][:bs],
        )

    def _compute_accept_bonus_eager(
        self,
        *,
        candidates: torch.Tensor,
        target_predict: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, block_size = candidates.shape
        correct_len, _ = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        confident_prefix = self._confident_prefix(confidence)
        correct_len = torch.minimum(
            correct_len.to(torch.int64), confident_prefix.to(torch.int64)
        )
        bonus_tokens = target_predict.gather(1, correct_len.unsqueeze(1)).squeeze(1)
        commit_lens = correct_len.to(torch.int32) + 1

        out_tokens = torch.empty(
            (bs, block_size), dtype=torch.int64, device=candidates.device
        )
        if block_size > 1:
            out_tokens[:, : block_size - 1].copy_(candidates[:, 1:])
        out_tokens[:, block_size - 1].fill_(0)
        out_tokens.scatter_(
            1,
            correct_len.unsqueeze(1),
            bonus_tokens.unsqueeze(1).to(torch.int64),
        )
        return commit_lens, bonus_tokens, out_tokens

    def _make_next_draft_input_prefill(
        self,
        *,
        bonus_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DSparkDraftInputV2:
        return DSparkDraftInputV2(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=seq_lens.to(dtype=torch.int64),
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def _make_next_draft_input_decode(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DSparkDraftInputV2:
        return DSparkDraftInputV2(
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(model_worker_batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        sampling_info = getattr(model_worker_batch, "sampling_info", None)
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and self.tp_rank == 0
            and not getattr(self, "_warned_sampling", False)
        ):
            self._warned_sampling = True
            logger.warning(
                "DSpark verifies greedily; temperature>0 requests are served with "
                "greedy verification. Rejection-sampling support is a follow-up."
            )

        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            return self._forward_prefill(model_worker_batch, on_publish)

        return self._forward_decode(model_worker_batch, on_publish)

    def _forward_prefill(
        self, model_worker_batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        original_capture_hidden_mode = model_worker_batch.capture_hidden_mode
        target_runner = self.target_worker.model_runner
        target_graph_runner = target_runner.graph_runner
        target_piecewise_graph_runner = target_runner.piecewise_cuda_graph_runner
        target_model = target_runner.model
        target_inner_model = getattr(target_model, "model", None)
        target_capture_aux_hidden_states = getattr(
            target_model, "capture_aux_hidden_states", None
        )
        target_layers_to_capture = (
            list(target_inner_model.layers_to_capture)
            if target_inner_model is not None
            and hasattr(target_inner_model, "layers_to_capture")
            else None
        )
        try:
            target_runner.graph_runner = None
            target_runner.piecewise_cuda_graph_runner = None

            if target_capture_aux_hidden_states is not None:
                target_model.capture_aux_hidden_states = False
            if target_layers_to_capture is not None:
                target_inner_model.layers_to_capture = []
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
            self.target_worker.set_hicache_consumer(
                model_worker_batch.hicache_consumer_index
            )
            forward_batch = ForwardBatch.init_new(model_worker_batch, target_runner)
            out = target_runner.forward(forward_batch)
            logits_output = out.logits_output
            if isinstance(logits_output, torch.Tensor):
                logits_output = self._tensor_to_logits_output(
                    logits_output,
                    forward_batch,
                    target_model,
                    "prefill logits pass",
                )

            if not forward_batch.is_prefill_only:
                next_token_ids = target_runner.sample(logits_output, forward_batch)
            else:
                next_token_ids = torch.zeros(
                    len(forward_batch.seq_lens),
                    dtype=torch.long,
                    device=forward_batch.input_ids.device,
                )
                if (
                    forward_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    target_runner.compute_logprobs_only(logits_output, forward_batch)

            batch_output = GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                can_run_cuda_graph=out.can_run_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
                routed_experts_output=out.routed_experts_output,
                indexer_topk_output=out.indexer_topk_output,
            )

            if target_capture_aux_hidden_states is not None:
                target_model.capture_aux_hidden_states = target_capture_aux_hidden_states
            if target_layers_to_capture is not None:
                target_inner_model.layers_to_capture = target_layers_to_capture
            self._ensure_target_dspark_capture()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            hidden_batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch,
                is_verify=True,
            )
        finally:
            model_worker_batch.capture_hidden_mode = original_capture_hidden_mode
            target_runner.graph_runner = target_graph_runner
            target_runner.piecewise_cuda_graph_runner = target_piecewise_graph_runner
            if target_capture_aux_hidden_states is not None:
                target_model.capture_aux_hidden_states = target_capture_aux_hidden_states
            if target_layers_to_capture is not None:
                target_inner_model.layers_to_capture = target_layers_to_capture

        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = model_worker_batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        main_hidden, _ = self._pop_hidden_states(
            hidden_batch_output.logits_output, "prefill"
        )
        if model_worker_batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        device = next_token_ids.device
        extend_lens = model_worker_batch.extend_lens
        prefix_lens = model_worker_batch.prefix_lens
        if extend_lens is None or prefix_lens is None:
            reqs = getattr(model_worker_batch, "reqs", None) or []
            if len(reqs) != len(model_worker_batch.seq_lens):
                raise RuntimeError(
                    "DSpark expected extend_lens / prefix_lens in extend mode, "
                    "and could not rebuild them from batch requests."
                )
            prefix_lens = [len(req.prefix_indices) for req in reqs]
            extend_lens = [req.extend_range.length for req in reqs]

        ctx_lens = torch.tensor(extend_lens, dtype=torch.int32, device=device)
        draft_seq_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(extend_lens)),
        )
        self._materialize_main_hidden_to_draft_kv(
            main_hidden=main_hidden,
            cache_loc=model_worker_batch.out_cache_loc,
            positions=positions,
        )

        batch_output.next_draft_input = self._make_next_draft_input_prefill(
            bonus_tokens=next_token_ids,
            seq_lens=model_worker_batch.seq_lens,
            cur_allocated_seq_lens_cpu=model_worker_batch.seq_lens_cpu,
        )
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        batch_output.next_draft_input.verify_done = verify_done
        return batch_output

    def _forward_decode(
        self, model_worker_batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = DSparkDraftInputV2.create_idle_input(
                device=self.device
            )
        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DSparkDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DSparkDraftInputV2 state on the running batch."
            )

        participates_in_dp_decode = (
            self.server_args.enable_dp_attention
            and model_worker_batch.forward_mode.is_idle()
            and model_worker_batch.global_num_tokens is not None
            and any(int(x) > 0 for x in model_worker_batch.global_num_tokens)
        )
        if model_worker_batch.forward_mode.is_idle() and not participates_in_dp_decode:
            return self._forward_idle(on_publish)
        dp_decode_global_num_tokens = self._get_dp_decode_global_num_tokens(
            model_worker_batch
        )

        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        device = self.device
        bs = len(model_worker_batch.seq_lens)
        block_size = int(self.block_size)
        prefix_lens = model_worker_batch.seq_lens
        req_pool_indices = model_worker_batch.req_pool_indices

        block_ids = torch.full(
            (bs, block_size), self.noise_token_id, dtype=torch.int64, device=device
        )
        block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))

        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        positions = positions_2d.reshape(-1).to(torch.int64)

        end_offset = prefix_lens + block_size
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=end_offset,
            batch_size=bs,
            draft_token_num=block_size,
            device=device,
        )

        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            block_hidden = self._run_draft_block(
                batch=model_worker_batch,
                bs=bs,
                block_ids=block_ids,
                positions=positions,
                verify_out_cache_loc=verify_out_cache_loc,
                dp_decode_global_num_tokens=dp_decode_global_num_tokens,
            )

            candidates, confidence = self._refine_block_markov(
                block_hidden=block_hidden,
                bonus_tokens=draft_input.bonus_tokens,
                output_bs=bs,
            )

        verify_input = DSparkVerifyInput(
            draft_token=candidates.reshape(-1),
            positions=positions,
            draft_token_num=block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        model_worker_batch.out_cache_loc = verify_out_cache_loc
        if participates_in_dp_decode:
            model_worker_batch.forward_mode = ForwardMode.DECODE
        original_global_num_tokens = model_worker_batch.global_num_tokens
        original_global_num_tokens_for_logprob = (
            model_worker_batch.global_num_tokens_for_logprob
        )
        if dp_decode_global_num_tokens is not None:
            model_worker_batch.global_num_tokens = dp_decode_global_num_tokens
            if original_global_num_tokens_for_logprob is not None:
                model_worker_batch.global_num_tokens_for_logprob = (
                    dp_decode_global_num_tokens
                )
        try:
            verify_forward_batch, _ = verify_input.prepare_for_verify(
                model_worker_batch, self.target_worker
            )
        finally:
            model_worker_batch.global_num_tokens = original_global_num_tokens
            model_worker_batch.global_num_tokens_for_logprob = (
                original_global_num_tokens_for_logprob
            )
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph
        if isinstance(logits_output, torch.Tensor):
            raise RuntimeError(
                "DSpark verify requires target logits and hidden states, but got a "
                "hidden-state tensor only."
            )

        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, block_size
        )

        new_seq_lens = None
        if bs == 0:
            bonus_tokens = torch.empty((0,), dtype=torch.int64, device=device)
            commit_lens = torch.empty((0,), dtype=torch.int32, device=device)
            out_tokens = torch.empty((0, block_size), dtype=torch.int64, device=device)
        elif self._use_triton_accept_bonus:
            try:
                (
                    commit_lens,
                    bonus_tokens,
                    out_tokens,
                    new_seq_lens,
                ) = self._next_accept_bonus_buffers(bs)
                _compute_dspark_accept_bonus_triton_unchecked(
                    candidates=candidates,
                    target_top1=target_predict,
                    confidence=confidence,
                    commit_lens_out=commit_lens,
                    bonus_ids_out=bonus_tokens,
                    out_tokens_out=out_tokens,
                    prefix_lens=prefix_lens,
                    new_seq_lens_out=new_seq_lens,
                    confidence_threshold=self.confidence_threshold,
                )
            except Exception as e:
                self._use_triton_accept_bonus = False
                logger.warning(
                    "DSPARK Triton accept/bonus failed; falling back to eager path: %s",
                    e,
                )
                commit_lens, bonus_tokens, out_tokens = self._compute_accept_bonus_eager(
                    candidates=candidates,
                    target_predict=target_predict,
                    confidence=confidence,
                )
        else:
            commit_lens, bonus_tokens, out_tokens = self._compute_accept_bonus_eager(
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )

        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        hidden, logits_output = self._pop_hidden_states(logits_output, "verify")
        if bs > 0:
            hidden = hidden.view(bs, block_size, -1)
            commit_mask = (
                self._block_pos_offsets.unsqueeze(0)
                < commit_lens.unsqueeze(1).to(torch.int64)
            ).reshape(-1)
            self._materialize_main_hidden_to_draft_kv(
                main_hidden=hidden.reshape(-1, hidden.shape[-1])[commit_mask],
                cache_loc=verify_out_cache_loc[commit_mask],
                positions=positions[commit_mask],
            )

        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
        )
        next_draft_input.carry_prepare_buffers_from(draft_input)
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=block_size,
            new_seq_lens=new_seq_lens,
        )

    def _forward_idle(self, on_publish) -> GenerationBatchResult:
        empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
        empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=empty_ids,
            accept_lens=empty_lens,
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.block_size),
            new_seq_lens=next_draft_input.new_seq_lens,
        )
