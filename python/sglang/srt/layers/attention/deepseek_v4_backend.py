from __future__ import annotations

import enum
import functools
import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn.functional as F

from sglang.jit_kernel.dsv4.online_c128_mtp import OnlineC128MTPController
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.managers.hisparse_accuracy_trace import trace_event

if envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
    # NOTE: should eventually be the only compressor backend
    from sglang.srt.layers.attention.dsv4.compressor_v2 import (
        CompressorBackendMixin,
        FusedCompressMetadata,
        create_paged_compressor_data,
    )
else:
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
        FusedCompressMetadata,
        create_paged_compressor_data,
    )

from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    PagedIndexerMetadata,
    copy_metadata,
    maybe_copy_inplace,
)
from sglang.srt.layers.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


def _get_logical_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
    if forward_batch.forward_mode.is_idle():
        return forward_batch.forward_mode
    return getattr(forward_batch, "_original_forward_mode", forward_batch.forward_mode)


T = TypeVar("T", bound=Optional[torch.Tensor])


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _reshape_flashmla_query_metadata(
    tensor: Optional[torch.Tensor],
    *,
    batch_size: int,
    query_len: int,
    name: str,
    pad_value: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    expected_rows = batch_size * query_len
    if tensor.ndim == 1:
        if tensor.numel() < expected_rows:
            tensor = torch.cat(
                [
                    tensor,
                    tensor.new_full((expected_rows - tensor.numel(),), pad_value),
                ],
                dim=0,
            )
        if tensor.numel() == expected_rows:
            return tensor.view(batch_size, query_len)
        if query_len == 1 and tensor.numel() == batch_size:
            return tensor.unsqueeze(1)
    elif tensor.ndim == 2:
        if tensor.shape[0] < expected_rows:
            tensor = torch.cat(
                [
                    tensor,
                    tensor.new_full(
                        (expected_rows - tensor.shape[0], tensor.shape[1]),
                        pad_value,
                    ),
                ],
                dim=0,
            )
        if tensor.shape[0] == expected_rows:
            return tensor.view(batch_size, query_len, tensor.shape[1])
        if tensor.shape[0] == batch_size and query_len == 1:
            return tensor.unsqueeze(1)
    elif tensor.ndim >= 3:
        if tensor.shape[0] == batch_size and tensor.shape[1] == query_len:
            return tensor

    raise RuntimeError(
        f"DeepSeekV4 FlashMLA {name} has incompatible shape: "
        f"shape={tuple(tensor.shape)}, expected_rows={expected_rows}, "
        f"batch_size={batch_size}, query_len={query_len}."
    )


def _reshape_flashmla_topk_length_metadata(
    tensor: Optional[torch.Tensor],
    *,
    batch_size: int,
    query_len: int,
    name: str,
    pad_value: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    expected_rows = batch_size * query_len

    if tensor.ndim == 2:
        if tensor.shape == (batch_size, query_len):
            return tensor.max(dim=1).values.contiguous()
        if tensor.shape[-1] == 1:
            tensor = tensor.reshape(-1)
        else:
            raise RuntimeError(
                f"DeepSeekV4 FlashMLA {name} has incompatible shape: "
                f"shape={tuple(tensor.shape)}, expected batch length shape "
                f"({batch_size},)."
            )
    elif tensor.ndim > 2:
        raise RuntimeError(
            f"DeepSeekV4 FlashMLA {name} has incompatible shape: "
            f"shape={tuple(tensor.shape)}, expected batch length shape "
            f"({batch_size},)."
        )

    if tensor.ndim != 1:
        raise RuntimeError(
            f"DeepSeekV4 FlashMLA {name} must be 1D after normalization, "
            f"got shape={tuple(tensor.shape)}."
        )

    if tensor.numel() == batch_size:
        return tensor
    if tensor.numel() < expected_rows:
        tensor = torch.cat(
            [
                tensor,
                tensor.new_full((expected_rows - tensor.numel(),), pad_value),
            ],
            dim=0,
        )
    if tensor.numel() >= expected_rows:
        return tensor[:expected_rows].view(batch_size, query_len).max(dim=1).values

    raise RuntimeError(
        f"DeepSeekV4 FlashMLA {name} has incompatible shape: "
        f"shape={tuple(tensor.shape)}, expected batch length shape "
        f"({batch_size},), expected_rows={expected_rows}."
    )


def _normalize_flashmla_sparse_metadata(
    *,
    indices: Optional[torch.Tensor],
    topk_lengths: Optional[torch.Tensor],
    batch_size: int,
    query_len: int,
    indices_name: str,
    lengths_name: str,
    indices_pad_value: int,
    lengths_pad_value: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    return (
        _reshape_flashmla_query_metadata(
            indices,
            batch_size=batch_size,
            query_len=query_len,
            name=indices_name,
            pad_value=indices_pad_value,
        ),
        _reshape_flashmla_topk_length_metadata(
            topk_lengths,
            batch_size=batch_size,
            query_len=query_len,
            name=lengths_name,
            pad_value=lengths_pad_value,
        ),
    )


def _create_flashmla_metadata():
    import flash_mla

    return flash_mla.get_mla_metadata()[0]


def _create_dummy_paged_compress_data(compress_ratio: int):
    return None


def _copy_or_replace(dst, src):
    if dst is not None and src is not None:
        dst.copy_(src)
        return dst
    return src


@dataclass(frozen=True)
class TargetVerifyLayout:
    active_bs: int
    query_len: int
    semantic_num_tokens: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class DSV4AttnMetadata:
    page_size: int
    page_table: torch.Tensor
    req_pool_indices_repeated: torch.Tensor
    raw_out_loc: torch.Tensor
    cuda_int32_kwargs: dict

    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor

    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor

    c4_sparse_topk: int
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None

    c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c128_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)

    @property
    def positions(self) -> torch.Tensor:
        return self.positions_casual

    def get_flashmla_metadata(self, compress_ratio: Literal[0, 4, 128]):
        if compress_ratio == 0:
            return self.c1_flashmla_metadata
        elif compress_ratio == 4:
            return self.c4_flashmla_metadata
        elif compress_ratio == 128:
            return self.c128_flashmla_metadata
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    def copy_(self, other: DSV4AttnMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "c4_sparse_topk",
                "page_size",
                "cuda_int32_kwargs",
            ],
            copy_fields=[
                "raw_out_loc",
                "req_pool_indices_repeated",
                "seq_lens_casual",
                "positions_casual",
                "c4_out_loc",
                "c128_out_loc",
                "page_table",
                "swa_page_indices",
                "swa_topk_lengths",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
            ],
            assign_fields=[
                "c1_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
            ],
        )

    def init_compression_metadata(self):
        assert self.page_table.dim() == 2
        assert (
            self.raw_out_loc.shape == self.seq_lens_casual.shape
        ), f"{self.raw_out_loc.shape=}, {self.seq_lens_casual.shape=}"

        (
            self.c4_out_loc,
            _,
            self.c4_topk_lengths_raw,
            self.c4_topk_lengths_clamp1,
            self.c128_out_loc,
            _,
            self.c128_topk_lengths_clamp1,
            self.c128_page_indices,
        ) = _init_compression_metadata_triton(
            self.seq_lens_casual,
            self.positions_casual,
            self.raw_out_loc,
            self.page_table,
            self.page_size,
            compute_page_indices=True,
        )

        self.c128_page_indices = _pad_last_dim(self.c128_page_indices)
        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)

    _CP_REINDEX_FIELDS = [
        "seq_lens_casual",
        "positions_casual",
        "swa_page_indices",
        "swa_topk_lengths",
        "page_table",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
    ]
    _CP_GLOBAL_FIELDS = [
        "raw_out_loc",
        "c4_out_loc",
        "c128_out_loc",
    ]

    def apply_cp_reindex(self) -> None:
        cp_rank = get_attention_cp_rank()
        cp_size = get_attention_cp_size()
        idx = slice(cp_rank, None, cp_size)
        pre_global_len = self.seq_lens_casual.shape[0]
        assert pre_global_len % cp_size == 0, (
            f"apply_cp_reindex: global token count {pre_global_len} is not divisible by cp_size={cp_size}. "
            "CP round-robin requires padding to ensure divisibility."
        )
        expected_local_len = pre_global_len // cp_size
        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name, None)
            assert isinstance(
                val, torch.Tensor
            ), f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            setattr(self, field_name, val[idx].contiguous())

        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name)
            assert val.shape[0] == expected_local_len, (
                f"apply_cp_reindex post-condition: {field_name}.shape[0]={val.shape[0]} "
                f"!= expected_local_len={expected_local_len} (cp_size={cp_size})"
            )
        for field_name in self._CP_GLOBAL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None:
                continue
            assert val.shape[0] == pre_global_len, (
                f"apply_cp_reindex post-condition: global field {field_name}.shape[0]={val.shape[0]} "
                f"!= pre_global_len={pre_global_len} (must remain global for compressor write path)"
            )

    def init_flashmla_related(self):
        if self.c4_sparse_topk not in (512, 1024):
            if (
                self.c4_sparse_topk <= 0
                or self.c4_sparse_topk > 1024
                or self.c4_sparse_topk % PAGE_INDEX_ALIGNED_SIZE != 0
            ):
                raise ValueError(
                    "DSV4 c4_sparse_topk must be a positive <=1024 multiple of "
                    f"{PAGE_INDEX_ALIGNED_SIZE}, got {self.c4_sparse_topk}."
                )
            if not envs.SGLANG_OPT_USE_TOPK_V2.get():
                raise ValueError(
                    "DSV4 c4_sparse_topk values other than 512/1024 require "
                    "SGLANG_OPT_USE_TOPK_V2=1."
                )
        assert self.c4_topk_lengths_clamp1 is not None
        self.c4_sparse_topk_lengths = torch.clamp(
            self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
        )
        self.c4_sparse_page_indices = torch.full(
            (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
            -1,
            dtype=torch.int32,
            device=self.c4_topk_lengths_clamp1.device,
        )
        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)
        self.c1_flashmla_metadata = _create_flashmla_metadata()
        self.c4_flashmla_metadata = _create_flashmla_metadata()
        self.c128_flashmla_metadata = _create_flashmla_metadata()


@dataclass
class DSV4Metadata:
    core_attn_metadata: DSV4AttnMetadata
    indexer_metadata: Optional[PagedIndexerMetadata]

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    @property
    def core_metadata(self) -> DSV4AttnMetadata:
        return self.core_attn_metadata

    def copy_(self, other: DSV4Metadata):
        self.core_attn_metadata.copy_(other.core_attn_metadata)
        maybe_copy_inplace(self.indexer_metadata, src=other.indexer_metadata)
        maybe_copy_inplace(self.c4_compress_metadata, src=other.c4_compress_metadata)
        maybe_copy_inplace(
            self.c128_compress_metadata, src=other.c128_compress_metadata
        )


@dataclass
class DSV4RawVerifyMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    extend_seq_lens: Optional[torch.Tensor] = None
    num_draft_tokens: Optional[int] = None
    seq_lens_cpu: Optional[List[int]] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    def copy_(self, other: DSV4RawVerifyMetadata):
        for field_name in ("req_pool_indices", "seq_lens", "out_cache_loc"):
            dst = getattr(self, field_name)
            src = getattr(other, field_name)
            if dst.shape != src.shape:
                raise RuntimeError(
                    "DeepSeekV4 raw target-verify metadata copy shape mismatch: "
                    f"field={field_name}, dst_shape={tuple(dst.shape)}, "
                    f"src_shape={tuple(src.shape)}."
                )
            if dst is src:
                continue
            try:
                dst.copy_(src)
            except RuntimeError as exc:
                if (
                    "some elements of the input tensor and the written-to tensor "
                    "refer to a single memory location"
                ) not in str(exc):
                    raise
                dst.copy_(src.clone())

        self.extend_seq_lens = other.extend_seq_lens
        self.num_draft_tokens = other.num_draft_tokens
        self.seq_lens_cpu = other.seq_lens_cpu
        self.c128_compress_metadata = _copy_or_replace(
            self.c128_compress_metadata, other.c128_compress_metadata
        )


@dataclass
class DSV4RawDecodeMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    def copy_(self, other: DSV4RawDecodeMetadata):
        for field_name in ("req_pool_indices", "seq_lens", "out_cache_loc"):
            dst = getattr(self, field_name)
            src = getattr(other, field_name)
            if dst.shape != src.shape:
                raise RuntimeError(
                    "DeepSeekV4 raw decode metadata copy shape mismatch: "
                    f"field={field_name}, dst_shape={tuple(dst.shape)}, "
                    f"src_shape={tuple(src.shape)}."
                )
            if dst is src:
                continue
            try:
                dst.copy_(src)
            except RuntimeError as exc:
                if (
                    "some elements of the input tensor and the written-to tensor "
                    "refer to a single memory location"
                ) not in str(exc):
                    raise
                dst.copy_(src.clone())


class _GraphBucket(enum.Enum):
    DECODE_OR_IDLE = "decode_or_idle"
    TARGET_VERIFY = "target_verify"
    DRAFT_EXTEND = "draft_extend"

    @classmethod
    def of(cls, forward_mode: ForwardMode) -> _GraphBucket:
        if forward_mode.is_decode_or_idle():
            return cls.DECODE_OR_IDLE
        if forward_mode.is_target_verify():
            return cls.TARGET_VERIFY
        if forward_mode.is_draft_extend(include_v2=True):
            return cls.DRAFT_EXTEND
        raise NotImplementedError(f"unsupported {forward_mode=}")


class DeepseekV4AttnBackend(
    AttentionBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        head_dim = model_runner.model_config.head_dim
        assert (
            head_dim == 512
        ), "DSV4 MQA head_dim = qk_nope_head_dim(448) + qk_rope_head_dim(64) = 512"
        self.softmax_scale: float = head_dim**-0.5
        self.head_dim_v: int = model_runner.model_config.v_head_dim
        self.cuda_int32_kwargs = {"device": self.device, "dtype": torch.int32}
        self.swa_page_size = 128
        assert model_runner.page_size is not None
        assert model_runner.req_to_token_pool is not None
        self.page_size = model_runner.page_size
        assert self.page_size == 256, "the system hardcodes page_size=256"

        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool: DeepSeekV4TokenToKVPool = model_runner.token_to_kv_pool
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if model_runner.enable_hisparse:
            from sglang.srt.mem_cache.sparsity import resolve_hisparse_top_k

            self.c4_topk = resolve_hisparse_top_k(
                model_runner.server_args, model_runner.model_config.hf_text_config
            )
        else:
            self.c4_topk = getattr(
                model_runner.model_config.hf_text_config, "index_topk", C4_TOPK
            )

        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"
        self.mtp_enabled = self.topk > 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens: int = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id
        self.forward_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ] = None
        self._replay_forward_batch: Optional[ForwardBatch] = None  # FIXME: out-of-band
        self._accuracy_trace_enabled = (
            envs.SGLANG_DSV4_HISPARSE_ACCURACY_TRACE.get()
        )
        self.online_c128_mtp = OnlineC128MTPController(self)
        self._log_hisparse_online_c128_mtp_state(model_runner)

    def _move_to_device(self, x: List[int]) -> torch.Tensor:
        pin_tensor = torch.tensor(x, dtype=torch.int32, pin_memory=True)
        return pin_tensor.to(self.device, non_blocking=True)

    def _log_hisparse_online_c128_mtp_state(self, model_runner: ModelRunner) -> None:
        if not model_runner.enable_hisparse or not self.online_c128_mtp.enabled():
            return

        c128_state_layers = sum(
            1
            for pool in self.token_to_kv_pool.compress_state_pools
            if pool is not None and pool.ratio == 128
        )
        c128_state_dtype = None
        for pool in self.token_to_kv_pool.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                c128_state_dtype = pool.kv_score_buffer.kv_score.dtype
                break
        state_slot_offset = self.token_to_kv_pool.get_online_c128_mtp_state_slot_offset()
        max_draft_tokens = self.token_to_kv_pool.get_online_c128_mtp_max_draft_tokens()

        if getattr(model_runner, "is_draft_worker", False):
            logger.info(
                "DSV4 HiSparse online C128 MTP draft runner initialized: "
                "c128_state_layers=%d, c128_state_dtype=%s, draft_banks=%d.",
                c128_state_layers,
                c128_state_dtype,
                max_draft_tokens,
            )
            return

        if c128_state_layers > 0 and state_slot_offset <= 0:
            raise RuntimeError(
                "DSV4 HiSparse online C128 MTP target runner has C128 state "
                "layers but no pending state-bank offset."
            )
        if max_draft_tokens <= 0:
            raise RuntimeError(
                "DSV4 HiSparse online C128 MTP target runner has no draft banks."
            )

        logger.warning(
            "DSV4 HiSparse online C128 MTP target runner initialized: "
            "c128_state_layers=%d, c128_state_dtype=%s, "
            "state_slot_offset=%d, draft_banks=%d. "
            "C4 host mirror remains handled by HiSparseCoordinator.",
            c128_state_layers,
            c128_state_dtype,
            state_slot_offset,
            max_draft_tokens,
        )

    def _target_verify_lengths_cpu(
        self, seq_lens_cpu: List[int], num_draft_tokens: int
    ) -> Tuple[List[int], List[int]]:
        return (
            [int(x) + num_draft_tokens for x in seq_lens_cpu],
            [num_draft_tokens] * len(seq_lens_cpu),
        )

    def _make_target_verify_c128_metadata(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        extend_seq_lens: torch.Tensor,
        use_prefill_cuda_graph: bool,
        online_c128_state_slot_offset: int,
        num_draft_tokens: int,
    ) -> Optional[FusedCompressMetadata]:
        if not self.online_c128_mtp.enabled():
            return None

        seq_lens_cpu, extend_lens_cpu = self._target_verify_lengths_cpu(
            seq_lens_cpu, num_draft_tokens
        )
        return create_paged_compressor_data(
            compress_ratio=128,
            is_prefill=True,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens + num_draft_tokens,
            seq_lens_cpu=seq_lens_cpu,
            extend_lens=extend_seq_lens,
            extend_lens_cpu=extend_lens_cpu,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            online_state_slot_offset=online_c128_state_slot_offset,
        )

    def init_forward_metadata_indexer(self, core_attn_metadata: DSV4AttnMetadata):
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=core_attn_metadata.page_table,
            c4_seq_lens=core_attn_metadata.c4_topk_lengths_raw,
            req_pool_indices_repeated=core_attn_metadata.req_pool_indices_repeated,
        )

    def init_forward_metadata_decode(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> Union[DSV4Metadata, DSV4RawDecodeMetadata]:
        assert (
            req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]
        ), f"{req_pool_indices.shape=} {seq_lens.shape=} {out_cache_loc.shape=}"

        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            return DSV4RawDecodeMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=True,
        )

        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_prefill(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        out_cache_loc: torch.Tensor,
        num_tokens: int,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: List[int],
        need_compress: bool = True,
        use_prefill_cuda_graph: bool = False,
        online_c128_state_slot_offset: int = 0,
    ) -> DSV4Metadata:
        seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
            num_tokens=num_tokens,
            seq_lens=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            padded_num_tokens=out_cache_loc.shape[0],
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=need_compress,
            is_prefill=True,
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if need_compress
            else None
        )
        if not need_compress:
            create = _create_dummy_paged_compress_data
        else:
            create = functools.partial(
                create_paged_compressor_data,
                is_prefill=True,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                extend_lens=extend_seq_lens,
                extend_lens_cpu=extend_seq_lens_cpu,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
                online_state_slot_offset=online_c128_state_slot_offset,
            )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_target_verify(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None,
        use_prefill_cuda_graph: bool = False,
        online_c128_state_slot_offset: int = 0,
    ) -> Union[DSV4Metadata, DSV4RawVerifyMetadata]:
        num_draft_tokens = self._get_target_verify_tokens_per_req(
            len(seq_lens), num_tokens=num_tokens, out_cache_loc=out_cache_loc
        )
        out_cache_loc = self._pad_target_verify_out_cache_loc(
            seq_lens, out_cache_loc, len(seq_lens) * num_draft_tokens
        )
        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            assert out_cache_loc is not None
            seq_lens_cpu_list = seq_lens.detach().cpu().tolist()
            if (
                not hasattr(self, "extend_seq_lens_buffer")
                or getattr(self, "extend_seq_lens_buffer_num_draft_tokens", None)
                != num_draft_tokens
            ):
                self.extend_seq_lens_buffer = torch.tensor(
                    [num_draft_tokens] * 1025, device=self.device
                )
                self.extend_seq_lens_buffer_num_draft_tokens = num_draft_tokens
            extend_seq_lens = self.extend_seq_lens_buffer[: len(seq_lens)]

            return DSV4RawVerifyMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                extend_seq_lens=extend_seq_lens,
                num_draft_tokens=num_draft_tokens,
                seq_lens_cpu=seq_lens_cpu_list,
                c128_compress_metadata=self._make_target_verify_c128_metadata(
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens_cpu_list,
                    extend_seq_lens=extend_seq_lens,
                    use_prefill_cuda_graph=use_prefill_cuda_graph,
                    online_c128_state_slot_offset=online_c128_state_slot_offset,
                    num_draft_tokens=num_draft_tokens,
                ),
            )
        else:
            seq_lens_cpu = seq_lens.tolist()
            return self.init_forward_metadata_target_verify_old(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
                num_tokens=num_tokens,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
                online_c128_state_slot_offset=online_c128_state_slot_offset,
            )

    def _get_target_verify_tokens_per_req(
        self,
        batch_size: int,
        *,
        num_tokens: Optional[int] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> int:
        if batch_size == 0:
            return self.speculative_num_draft_tokens

        if num_tokens is None:
            if out_cache_loc is None:
                return self.speculative_num_draft_tokens
            expected_tokens = batch_size * self.speculative_num_draft_tokens
            if out_cache_loc.numel() >= expected_tokens:
                return self.speculative_num_draft_tokens
            num_tokens = out_cache_loc.numel()

        if num_tokens % batch_size != 0:
            raise RuntimeError(
                "DeepSeekV4 target-verify metadata requires a uniform token count: "
                f"{num_tokens=} {batch_size=} "
                f"out_cache_loc_shape={None if out_cache_loc is None else tuple(out_cache_loc.shape)}"
            )
        return num_tokens // batch_size

    def _check_target_verify_metadata_parity(
        self,
        *,
        batch_size: int,
        num_draft_tokens: int,
        out_cache_loc: Optional[torch.Tensor],
        req_pool_indices_repeated: Optional[torch.Tensor] = None,
        online_c128_state_slot_offset: int = 0,
        context: str,
    ) -> None:
        if batch_size < 0 or num_draft_tokens <= 0:
            raise RuntimeError(
                "DeepSeekV4 target-verify metadata has invalid dimensions: "
                f"context={context}, batch_size={batch_size}, "
                f"num_draft_tokens={num_draft_tokens}."
            )
        expected_tokens = batch_size * num_draft_tokens
        if out_cache_loc is not None and out_cache_loc.numel() != expected_tokens:
            raise RuntimeError(
                "DeepSeekV4 target-verify out_cache_loc shape mismatch: "
                f"context={context}, expected_tokens={expected_tokens}, "
                f"out_cache_loc_shape={tuple(out_cache_loc.shape)}."
            )
        if (
            req_pool_indices_repeated is not None
            and req_pool_indices_repeated.numel() != expected_tokens
        ):
            raise RuntimeError(
                "DeepSeekV4 target-verify repeated req indices shape mismatch: "
                f"context={context}, expected_tokens={expected_tokens}, "
                f"req_pool_indices_repeated_shape={tuple(req_pool_indices_repeated.shape)}."
            )
        if self.online_c128_mtp.enabled() and online_c128_state_slot_offset < 0:
            raise RuntimeError(
                "DeepSeekV4 online C128 state slot offset must be non-negative: "
                f"context={context}, offset={online_c128_state_slot_offset}."
            )

    def _pad_target_verify_out_cache_loc(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor:
        if out_cache_loc is None:
            return seq_lens.new_zeros(num_tokens)
        if out_cache_loc.numel() == num_tokens:
            return out_cache_loc
        if out_cache_loc.numel() > num_tokens:
            return out_cache_loc[:num_tokens]
        raise RuntimeError(
            "DeepSeekV4 target-verify out_cache_loc is shorter than the "
            "semantic token count: "
            f"out_cache_loc_shape={tuple(out_cache_loc.shape)}, num_tokens={num_tokens}."
        )

    def init_forward_metadata_target_verify_old(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None,
        use_prefill_cuda_graph: bool = False,
        online_c128_state_slot_offset: int = 0,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        num_draft_tokens = self._get_target_verify_tokens_per_req(
            batch_size, num_tokens=num_tokens, out_cache_loc=out_cache_loc
        )
        seq_lens = seq_lens + num_draft_tokens
        seq_lens_cpu = [x + num_draft_tokens for x in seq_lens_cpu]
        extend_seq_lens_cpu = [num_draft_tokens] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = num_draft_tokens * batch_size
        out_cache_loc = self._pad_target_verify_out_cache_loc(
            seq_lens, out_cache_loc, num_tokens
        )
        self._check_target_verify_metadata_parity(
            batch_size=batch_size,
            num_draft_tokens=num_draft_tokens,
            out_cache_loc=out_cache_loc,
            online_c128_state_slot_offset=online_c128_state_slot_offset,
            context="target_verify_old",
        )
        return self.init_forward_metadata_prefill(
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=True,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            online_c128_state_slot_offset=online_c128_state_slot_offset,
        )

    def make_forward_metadata_from_raw_verify(
        self,
        raw_metadata: DSV4RawVerifyMetadata,
        online_c128_state_slot_offset: int = 0,
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        bs = len(seq_lens)
        if raw_metadata.num_draft_tokens is not None:
            num_draft_tokens = raw_metadata.num_draft_tokens
        else:
            num_draft_tokens = self._get_target_verify_tokens_per_req(
                bs, out_cache_loc=out_cache_loc
        )
        seq_lens = seq_lens + num_draft_tokens
        extend_seq_lens = raw_metadata.extend_seq_lens
        num_tokens = bs * num_draft_tokens
        out_cache_loc = self._pad_target_verify_out_cache_loc(
            seq_lens, out_cache_loc, num_tokens
        )

        seq_lens_casual, req_pool_indices_repeated = (
            self.expand_extend_with_same_length(
                bs, num_draft_tokens, seq_lens, req_pool_indices
            )
        )
        self._check_target_verify_metadata_parity(
            batch_size=bs,
            num_draft_tokens=num_draft_tokens,
            out_cache_loc=out_cache_loc,
            req_pool_indices_repeated=req_pool_indices_repeated,
            online_c128_state_slot_offset=online_c128_state_slot_offset,
            context="raw_verify",
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)
        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=True,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_seq_lens,
            seq_lens_cpu=None,
            extend_lens_cpu=None,
            use_prefill_cuda_graph=True,
            num_q_tokens=num_tokens,
            online_state_slot_offset=online_c128_state_slot_offset,
        )
        c128_compress_metadata = raw_metadata.c128_compress_metadata
        if c128_compress_metadata is None:
            c128_compress_metadata = create(compress_ratio=128)
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=c128_compress_metadata,
        )

    def make_forward_metadata_from_raw_decode(
        self, raw_metadata: DSV4RawDecodeMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_draft_extend(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        num_tokens_per_bs: int,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        extend_seq_lens_cpu = [num_tokens_per_bs] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = num_tokens_per_bs * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=False,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    @staticmethod
    def _active_batch_size(
        forward_batch: ForwardBatch,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        prefer_draft_token: bool = False,
    ) -> int:
        if prefer_draft_token:
            spec_info = getattr(forward_batch, "spec_info", None)
            draft_token_num = getattr(spec_info, "draft_token_num", None)
            draft_token = getattr(spec_info, "draft_token", None)
            if draft_token_num is not None and draft_token is not None:
                draft_token_num = int(draft_token_num)
                if draft_token_num > 0:
                    draft_numel = int(draft_token.numel())
                    if draft_numel % draft_token_num == 0:
                        return min(
                            draft_numel // draft_token_num,
                            int(req_pool_indices.shape[0]),
                            int(seq_lens.shape[0]),
                        )

        # CUDA graph replay pads req/seq tensors to the captured bucket size.
        # Some graph runners temporarily set ForwardBatch.batch_size to the
        # padded bucket size, so prefer the semantic batch size when provided.
        raw_bs = int(
            getattr(
                forward_batch,
                "_cuda_graph_raw_batch_size",
                getattr(
                    forward_batch,
                    "_original_batch_size",
                    forward_batch.batch_size,
                ),
            )
        )
        return min(raw_bs, int(req_pool_indices.shape[0]), int(seq_lens.shape[0]))

    def _target_verify_query_len(
        self,
        forward_batch: ForwardBatch,
        batch_size: int,
    ) -> int:
        if batch_size == 0:
            return self.speculative_num_draft_tokens

        spec_info = forward_batch.spec_info
        if spec_info is not None:
            draft_token_num = getattr(spec_info, "draft_token_num", None)
            if draft_token_num is not None:
                query_len = int(draft_token_num)
                if query_len <= 0:
                    raise RuntimeError(
                        "DeepSeekV4 target-verify draft_token_num must be positive: "
                        f"draft_token_num={draft_token_num}."
                    )
                return query_len
            draft_token = getattr(spec_info, "draft_token", None)
            if draft_token is not None:
                draft_numel = int(draft_token.numel())
                if batch_size > 0 and draft_numel % batch_size == 0:
                    return draft_numel // batch_size

        # Last-resort fallback for non-EAGLE callers. Target verify normally
        # must not use input_ids here because TP scatter may pad it. Only accept
        # it when it still represents a uniform per-request token width.
        if forward_batch.input_ids is not None:
            input_numel = int(forward_batch.input_ids.numel())
            if batch_size > 0 and input_numel % batch_size == 0:
                return input_numel // batch_size
        raise RuntimeError(
            "DeepSeekV4 target-verify cannot determine semantic query length: "
            f"batch_size={batch_size}, "
            f"input_ids_shape={None if forward_batch.input_ids is None else tuple(forward_batch.input_ids.shape)}."
        )

    def _make_target_verify_layout(
        self,
        forward_batch: ForwardBatch,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        active_bs: Optional[int] = None,
    ) -> TargetVerifyLayout:
        if active_bs is None:
            active_bs = self._active_batch_size(
                forward_batch,
                req_pool_indices,
                seq_lens,
                prefer_draft_token=True,
            )
        query_len = self._target_verify_query_len(forward_batch, active_bs)
        return TargetVerifyLayout(
            active_bs=active_bs,
            query_len=query_len,
            semantic_num_tokens=active_bs * query_len,
            req_pool_indices=req_pool_indices[:active_bs],
            seq_lens=seq_lens[:active_bs],
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        logical_forward_mode = _get_logical_forward_mode(forward_batch)
        if self.mtp_enabled and logical_forward_mode.is_idle():
            self.online_c128_mtp.clear()
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert forward_batch.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        assert seq_lens_cpu is not None
        max_seq_len = int(seq_lens_cpu.max().item())
        active_bs = self._active_batch_size(
            forward_batch,
            req_pool_indices,
            seq_lens,
            prefer_draft_token=logical_forward_mode.is_target_verify(),
        )
        online_c128_state_slot_offset = self.online_c128_mtp.prepare_forward(
            logical_forward_mode,
            req_pool_indices,
            seq_lens,
            verify_bs=active_bs,
        )
        if self._accuracy_trace_enabled:
            trace_event(
                logger,
                "dsv4_forward_metadata",
                mode=str(logical_forward_mode),
                active_bs=active_bs,
                padded_bs=int(req_pool_indices.shape[0]),
                max_seq_len=max_seq_len,
                online_state_slot_offset=online_c128_state_slot_offset,
                out_cache_loc=forward_batch.out_cache_loc,
            )

        if logical_forward_mode.is_decode_or_idle():
            metadata = self.init_forward_metadata_decode(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif logical_forward_mode.is_target_verify():
            layout = self._make_target_verify_layout(
                forward_batch,
                req_pool_indices,
                seq_lens,
                active_bs=active_bs,
            )
            if (
                self._accuracy_trace_enabled
                and forward_batch.input_ids is not None
                and int(forward_batch.input_ids.numel())
                != layout.semantic_num_tokens
            ):
                trace_event(
                    logger,
                    "dsv4_target_verify_tp_padding",
                    active_bs=layout.active_bs,
                    query_len=layout.query_len,
                    semantic_num_tokens=layout.semantic_num_tokens,
                    input_ids_numel=int(forward_batch.input_ids.numel()),
                    out_cache_loc=forward_batch.out_cache_loc,
                )
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=max_seq_len,
                req_pool_indices=layout.req_pool_indices,
                seq_lens=layout.seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=layout.semantic_num_tokens,
                online_c128_state_slot_offset=online_c128_state_slot_offset,
            )
        elif logical_forward_mode.is_prefill(include_draft_extend_v2=True):
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            extend_seq_lens = forward_batch.extend_seq_lens
            assert (
                seq_lens is not None
                and seq_lens_cpu is not None
                and extend_seq_lens is not None
                and extend_seq_lens_cpu is not None
            )
            is_draft = logical_forward_mode.is_draft_extend(include_v2=True)
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=sum(extend_seq_lens_cpu),
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                need_compress=not is_draft,
                online_c128_state_slot_offset=online_c128_state_slot_offset,
            )
        else:
            self.online_c128_mtp.clear()
            raise NotImplementedError(f"unsupported mode {logical_forward_mode=}")

        self.forward_metadata = metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.cuda_graph_metadata_of_bucket_and_bs: Dict[
            _GraphBucket,
            Dict[
                int,
                Union[DSV4Metadata, DSV4RawDecodeMetadata, DSV4RawVerifyMetadata],
            ],
        ] = {bucket: {} for bucket in _GraphBucket}
        self.draft_extend_num_tokens_per_bs = (
            max_num_tokens // max_bs if max_bs > 0 else 1
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        assert req_pool_indices.size(0) == bs
        assert seq_lens.size(0) == bs

        bucket = _GraphBucket.of(forward_mode)
        raw_type: Optional[type] = None
        if bucket == _GraphBucket.DECODE_OR_IDLE:
            metadata = self.init_forward_metadata_decode(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=torch.zeros_like(seq_lens),
            )
            raw_type = DSV4RawDecodeMetadata
        elif bucket == _GraphBucket.TARGET_VERIFY:
            out_cache_loc = torch.zeros(num_tokens, **self.cuda_int32_kwargs)
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=True,
            )
            raw_type = DSV4RawVerifyMetadata
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_bs = num_tokens // bs
            metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError(f"{forward_mode=} not supported yet")

        self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs] = metadata
        self.forward_metadata = metadata
        if raw_type is not None:
            self._current_capture_raw = (
                metadata if isinstance(metadata, raw_type) else None
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        bucket = _GraphBucket.of(forward_mode)

        # FIXME: see cuda_graph_runner — this attribute is set out-of-band.
        fb = self._replay_forward_batch
        out_cache_loc = fb.out_cache_loc
        actual_forward_mode = fb.forward_mode
        logical_forward_mode = _get_logical_forward_mode(fb)

        if actual_forward_mode == ForwardMode.IDLE:
            logger.debug(
                f"[IDLE replay] bs={bs}, "
                f"local_seq_lens_len={len(seq_lens)}, "
                f"has_graph={bs in self.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.DECODE_OR_IDLE]}"
            )
            device = seq_lens.device
            seq_lens = torch.ones(bs, dtype=seq_lens.dtype, device=device)
            seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
            seq_lens_sum = bs
            req_pool_indices = torch.zeros(
                bs, dtype=req_pool_indices.dtype, device=device
            )
            out_cache_loc = torch.zeros(bs, dtype=torch.int64, device=device)

        assert seq_lens_cpu is not None
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        actual_max_seq_len = seq_lens_cpu.max().item()
        chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
        assert actual_max_seq_len <= chosen_max_seq_len
        active_bs = self._active_batch_size(
            fb,
            req_pool_indices,
            seq_lens,
            prefer_draft_token=logical_forward_mode.is_target_verify(),
        )
        active_req_pool_indices = req_pool_indices[:active_bs]
        active_seq_lens = seq_lens[:active_bs]

        if bucket == _GraphBucket.DECODE_OR_IDLE:
            assert out_cache_loc is not None
            assert len(out_cache_loc.shape) == 1, f"{out_cache_loc.shape=}"
            self.online_c128_mtp.prepare_forward(
                logical_forward_mode,
                active_req_pool_indices,
                active_seq_lens,
            )
            if self._accuracy_trace_enabled:
                trace_event(
                    logger,
                    "dsv4_graph_replay_decode",
                    active_bs=active_bs,
                    graph_bs=bs,
                    mode=str(logical_forward_mode),
                )
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, bs - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_decode(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
            )
        elif bucket == _GraphBucket.TARGET_VERIFY:
            layout = self._make_target_verify_layout(
                fb,
                req_pool_indices,
                seq_lens,
                active_bs=active_bs,
            )
            verify_bs = layout.active_bs
            if verify_bs < 0 or verify_bs > bs:
                raise RuntimeError(
                    "DeepSeekV4 graph replay target-verify active batch is invalid: "
                    f"verify_bs={verify_bs}, graph_bs={bs}."
                )
            if self.online_c128_mtp.enabled() and verify_bs == 0:
                self.online_c128_mtp.clear()
                self.forward_metadata = self.cuda_graph_metadata_of_bucket_and_bs[
                    bucket
                ][bs]
                return
            assert out_cache_loc is not None
            graph_num_tokens = layout.query_len * bs
            if out_cache_loc.numel() > graph_num_tokens:
                raise RuntimeError(
                    "DeepSeekV4 graph replay target-verify out_cache_loc exceeds "
                    "the graph bucket capacity: "
                    f"out_cache_loc_shape={tuple(out_cache_loc.shape)}, "
                    f"graph_bs={bs}, query_len={layout.query_len}."
                )
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, graph_num_tokens - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            online_c128_state_slot_offset = self.online_c128_mtp.prepare_forward(
                logical_forward_mode,
                req_pool_indices,
                seq_lens,
                verify_bs=verify_bs,
            )
            if self._accuracy_trace_enabled:
                trace_event(
                    logger,
                    "dsv4_graph_replay_target_verify",
                    active_bs=verify_bs,
                    graph_bs=bs,
                    query_len=layout.query_len,
                    semantic_num_tokens=layout.semantic_num_tokens,
                    graph_num_tokens=graph_num_tokens,
                    out_cache_loc=out_cache_loc,
                    online_state_slot_offset=online_c128_state_slot_offset,
                )
            temp_metadata = self.init_forward_metadata_target_verify(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
                num_tokens=graph_num_tokens,
                use_prefill_cuda_graph=True,
                online_c128_state_slot_offset=online_c128_state_slot_offset,
            )
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            self.online_c128_mtp.prepare_forward(
                logical_forward_mode,
                active_req_pool_indices,
                active_seq_lens,
            )
            if self._accuracy_trace_enabled:
                trace_event(
                    logger,
                    "dsv4_graph_replay_draft_extend",
                    active_bs=active_bs,
                    graph_bs=bs,
                    mode=str(logical_forward_mode),
                )
            num_tokens_per_bs = self.draft_extend_num_tokens_per_bs
            temp_metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            self.online_c128_mtp.clear()
            raise NotImplementedError

        self.replay_cuda_graph_metadata_from(
            bs=bs, temp_metadata=temp_metadata, bucket=bucket
        )

    def replay_cuda_graph_metadata_from(
        self,
        bs: int,
        temp_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ],
        bucket: _GraphBucket,
    ) -> None:
        chosen_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs]
        if isinstance(chosen_metadata, DSV4RawVerifyMetadata) and isinstance(
            temp_metadata, DSV4RawVerifyMetadata
        ):
            for field_name in ("req_pool_indices", "seq_lens", "out_cache_loc"):
                dst = getattr(chosen_metadata, field_name)
                src = getattr(temp_metadata, field_name)
                if dst.shape != src.shape:
                    raise RuntimeError(
                        "DeepSeekV4 target-verify graph replay metadata shape "
                        "mismatch before copy: "
                        f"field={field_name}, graph_bs={bs}, "
                        f"dst_shape={tuple(dst.shape)}, src_shape={tuple(src.shape)}."
                    )
        elif isinstance(chosen_metadata, DSV4RawDecodeMetadata) and isinstance(
            temp_metadata, DSV4RawDecodeMetadata
        ):
            for field_name in ("req_pool_indices", "seq_lens", "out_cache_loc"):
                dst = getattr(chosen_metadata, field_name)
                src = getattr(temp_metadata, field_name)
                if dst.shape != src.shape:
                    raise RuntimeError(
                        "DeepSeekV4 decode graph replay metadata shape mismatch "
                        "before copy: "
                        f"field={field_name}, graph_bs={bs}, "
                        f"dst_shape={tuple(dst.shape)}, src_shape={tuple(src.shape)}."
                    )
        chosen_metadata.copy_(temp_metadata)
        self.forward_metadata = chosen_metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def on_after_cuda_graph_warmup(self):
        metadata = self.forward_metadata
        if isinstance(metadata, DSV4Metadata) and isinstance(
            metadata.core_attn_metadata, DSV4AttnMetadata
        ):
            core = metadata.core_attn_metadata
            core.c1_flashmla_metadata = _create_flashmla_metadata()
            core.c4_flashmla_metadata = _create_flashmla_metadata()
            core.c128_flashmla_metadata = _create_flashmla_metadata()

        # PREP_IN_CUDA_GRAPH=True: warmup upgraded raw->full on the host;
        # restore raw so capture re-runs the upgrade inside the graph.
        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        raw_loc = forward_batch.out_cache_loc
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_k=swa_k,
            )
        else:
            swa_k_pack = quant_to_nope_fp8_rope_bf16_pack_triton(swa_k)
            self.token_to_kv_pool.set_swa_key_buffer_radix(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_nope_fp8_rope_bf16_pack=swa_k_pack,
            )

    def _maybe_upgrade_forward_metadata(self) -> None:
        # With SGLANG_PREP_IN_CUDA_GRAPH=1, init_forward_metadata_*
        # returns a Raw metadata that only carries a few tensors. The
        # full DSV4Metadata (including c4/c128 compress + core_attn +
        # indexer metadata) must be materialized before any caller that
        # touches those fields. For 1.6T the first two layers have
        # compress_ratio=128, so forward_core_compressor / forward_c4_indexer
        # can fire before attn_backend.forward(), and must trigger the
        # upgrade themselves.
        if isinstance(self.forward_metadata, DSV4RawVerifyMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_verify(
                raw_metadata=self.forward_metadata,
                online_c128_state_slot_offset=self.online_c128_mtp.state_slot_offset(),
            )
        elif isinstance(self.forward_metadata, DSV4RawDecodeMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_decode(
                raw_metadata=self.forward_metadata,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        self._maybe_upgrade_forward_metadata()

        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        metadata = self.forward_metadata
        core_attn_metadata = metadata.core_attn_metadata
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        if isinstance(core_attn_metadata, DSV4AttnMetadata):
            if save_kv_cache:
                self.store_cache(layer_id, swa_k, forward_batch)
            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

            extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
            if compress_ratio == 4:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c4_sparse_page_indices
                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
            elif compress_ratio == 128:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c128_page_indices
                extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

            swa_window_size = token_to_kv_pool.swa_window_size
            assert swa_k_cache.ndim == 2
            k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim
            swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
                swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
            )

            if extra_k_cache is not None:
                page_sizes = {
                    4: token_to_kv_pool.page_size // 4,
                    128: token_to_kv_pool.page_size // 128,
                }
                extra_k_cache = extra_k_cache[
                    :, : page_sizes[compress_ratio] * k_cache_total_dim
                ].view(
                    extra_k_cache.shape[0],
                    page_sizes[compress_ratio],
                    1,
                    k_cache_total_dim,
                )
            swa_page_indices = core_attn_metadata.swa_page_indices
            swa_topk_lengths = core_attn_metadata.swa_topk_lengths

            if q.ndim == 3:
                q = q.unsqueeze(1)
            query_batch_size, query_len = int(q.shape[0]), int(q.shape[1])
            swa_page_indices, swa_topk_lengths = _normalize_flashmla_sparse_metadata(
                indices=swa_page_indices,
                topk_lengths=swa_topk_lengths,
                batch_size=query_batch_size,
                query_len=query_len,
                indices_name="swa_page_indices",
                lengths_name="swa_topk_lengths",
                indices_pad_value=0,
                lengths_pad_value=1,
            )
            extra_indices, extra_topk_lengths = _normalize_flashmla_sparse_metadata(
                indices=extra_indices,
                topk_lengths=extra_topk_lengths,
                batch_size=query_batch_size,
                query_len=query_len,
                indices_name="extra_indices",
                lengths_name="extra_topk_lengths",
                indices_pad_value=-1,
                lengths_pad_value=1,
            )

            assert attn_sink is not None

            flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)

            assert (
                swa_page_indices.shape[-1] % 64 == 0
            ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            if extra_indices is not None:
                assert (
                    extra_indices.shape[-1] % 64 == 0
                ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

            import flash_mla

            o = flash_mla.flash_mla_with_kvcache(
                q=q,
                k_cache=swa_k_cache,
                head_dim_v=self.head_dim_v,
                block_table=None,
                cache_seqlens=None,
                tile_scheduler_metadata=flashmla_metadata,
                softmax_scale=self.softmax_scale,
                is_fp8_kvcache=True,
                indices=swa_page_indices,
                topk_length=swa_topk_lengths,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices,
                extra_topk_length=extra_topk_lengths,
            )[0]

            o = o.squeeze(1)
            return o

        raise NotImplementedError("ragged attention")

    def expand_prefill_casually(
        self,
        num_tokens: int,
        seq_lens: List[int],
        extend_seq_lens: List[int],
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_lens_casual = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        idx_to_req_repeated = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        offset = 0
        for i, (kv_len, qo_len) in enumerate(zip(seq_lens, extend_seq_lens)):
            out = seq_lens_casual[offset : offset + qo_len]
            offset += qo_len
            torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
            idx_to_req_repeated[offset - qo_len : offset].fill_(i)

        assert offset == num_tokens
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual,
                (0, pad_size),
                value=1,
            )
            req_pool_indices_repeated = torch.nn.functional.pad(
                req_pool_indices_repeated,
                (0, pad_size),
                value=req_pool_indices_repeated[-1].item(),
            )

        return seq_lens_casual, req_pool_indices_repeated

    def expand_extend_with_same_length(
        self,
        bs: int,
        qo_len: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        seq_lens_casual = seq_lens[:, None] + torch.arange(
            -qo_len + 1, 1, **self.cuda_int32_kwargs
        )
        seq_lens_casual = seq_lens_casual.flatten()
        idx_to_req_repeated = torch.arange(
            bs, **self.cuda_int32_kwargs
        ).repeat_interleave(qo_len)
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]
        return seq_lens_casual, req_pool_indices_repeated

    def make_core_attn_metadata(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        need_compress: bool = True,
        is_prefill: bool = False,
    ) -> DSV4AttnMetadata:
        assert self.swa_page_size == SWA_WINDOW

        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

        swa_page_indices = _pad_last_dim(
            swa_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
        )

        raw_positions = seq_lens_casual - 1
        swa_topk_lengths = torch.clamp(seq_lens_casual, max=SWA_WINDOW)

        page_table = req_to_token[
            req_pool_indices_repeated, : max_seq_len : self.page_size
        ]
        page_table = (page_table // self.page_size).to(torch.int32)

        core_attn_metadata = DSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=raw_positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_sparse_topk=self.c4_topk,
        )

        if need_compress:
            core_attn_metadata.init_compression_metadata()
            core_attn_metadata.init_flashmla_related()
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_page_indices = None
            core_attn_metadata.c1_flashmla_metadata = _create_flashmla_metadata()
            core_attn_metadata.c4_flashmla_metadata = None
            core_attn_metadata.c128_flashmla_metadata = None
        return core_attn_metadata

    def get_swa_page_indices(
        self,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
    ) -> torch.Tensor:
        pos_causal = seq_lens_casual - 1
        num_qo_tokens = seq_lens_casual.size(0)
        offsets = pos_causal.unsqueeze(1) - torch.arange(
            SWA_WINDOW, **self.cuda_int32_kwargs
        ).unsqueeze(0)
        invalid_offset_mask = offsets < 0
        offsets.masked_fill_(invalid_offset_mask, 0)
        raw_indices = self.req_to_token[req_pool_indices_repeated[:, None], offsets]
        assert raw_indices.shape == (num_qo_tokens, SWA_WINDOW)
        raw_indices.masked_fill_(invalid_offset_mask, -1)
        swa_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(raw_indices)
        return swa_indices


class DeepseekV4MultiStepBackend(DeepseekV4AttnBackend):
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DeepseekV4AttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DeepseekV4AttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )
    def _split_out_cache_loc_by_step(
        self, out_cache_loc: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if out_cache_loc is None:
            return None

        slots_per_req = self.topk * self.speculative_num_steps
        assert out_cache_loc.numel() % slots_per_req == 0, (
            "DeepSeekV4 EAGLE draft expects out_cache_loc to be laid out as "
            f"[bs, topk, speculative_num_steps], got {out_cache_loc.shape=} "
            f"{self.topk=} {self.speculative_num_steps=}"
        )
        num_reqs = out_cache_loc.numel() // slots_per_req
        return (
            out_cache_loc.reshape(num_reqs, self.topk, self.speculative_num_steps)
            .permute(2, 0, 1)
            .reshape(self.speculative_num_steps, -1)
        )
    
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_target_verify():
            super().init_forward_metadata(forward_batch)
            return

        original_out_cache_loc = forward_batch.out_cache_loc
        draft_cache_locs = getattr(forward_batch.spec_info, "draft_cache_locs", None)
        selected_out_cache_loc = original_out_cache_loc
        if draft_cache_locs is not None:
            forward_batch.out_cache_loc = draft_cache_locs
            selected_out_cache_loc = draft_cache_locs
        step_out_cache_loc = self._split_out_cache_loc_by_step(selected_out_cache_loc)

        try:
            for i in range(self.speculative_num_steps - 1):
                if step_out_cache_loc is not None:
                    forward_batch.out_cache_loc = step_out_cache_loc[i]
                self.attn_backends[i].init_forward_metadata(forward_batch)
        finally:
            forward_batch.out_cache_loc = original_out_cache_loc

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backends:
            backend.on_after_cuda_graph_warmup()

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        if self.speculative_num_steps == 1:
            return
        draft_cache_locs = getattr(forward_batch.spec_info, "draft_cache_locs", None)
        selected_out_cache_loc = (
            draft_cache_locs
            if draft_cache_locs is not None
            else forward_batch.out_cache_loc
        )
        step_out_cache_loc = self._split_out_cache_loc_by_step(
            selected_out_cache_loc
        )
        original_out_cache_loc = forward_batch.out_cache_loc
        if step_out_cache_loc is not None:
            forward_batch.out_cache_loc = step_out_cache_loc[0]

        semantic_raw_bs = int(
            getattr(forward_batch, "_original_batch_size", forward_batch.batch_size)
        )
        self.attn_backends[0]._replay_forward_batch = forward_batch
        forward_batch._cuda_graph_raw_batch_size = semantic_raw_bs
        try:
            self.attn_backends[0].init_forward_metadata_replay_cuda_graph(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_sum=forward_batch.seq_lens_sum,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )
        finally:
            self.attn_backends[0]._replay_forward_batch = None
            if hasattr(forward_batch, "_cuda_graph_raw_batch_size"):
                delattr(forward_batch, "_cuda_graph_raw_batch_size")
            forward_batch.out_cache_loc = original_out_cache_loc
        temp_metadata = self.attn_backends[0].forward_metadata

        for i in range(1, self.speculative_num_steps - 1):
            self.attn_backends[i].replay_cuda_graph_metadata_from(
                bs=bs,
                temp_metadata=temp_metadata,
                bucket=_GraphBucket.DECODE_OR_IDLE,
            )


def _pad_tensor_to_size(tensor: torch.Tensor, size: int, *, value: int = 0):
    if value == 0:
        return torch.cat(
            [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
            dim=0,
        )
    else:
        return torch.cat(
            [
                tensor,
                tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
            ],
            dim=0,
        )
