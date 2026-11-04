"""One-launch Hopper FP8 early MoE. Bind snapshots on the cold path.

Outputs alias the shared arena and are valid until ANY context sharing it next
runs. An EP group must call every layer epoch on every rank, including empty
inputs. The caller holds active slots immutable until completion. Here the
legacy logical_* names refer to the fixed physical routing-ID domain.
"""

import ctypes as C
import math
import threading
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist

from ._native import build_library
from .config import Geometry

LAYOUT = "iaas-sm90-fp8-gran8-wsf128-a128-a2-64-v1"
STAT_NAMES = (
    "kernel_start",
    "ready_publish",
    "all_ready_observed",
    "last_commit",
    "kernel_end",
    "valid_assignments",
    "eligible_local_assignments",
    "completed_assignments",
    "completed_m64_tasks",
    "partial_tasks",
    "padded_rows",
    "stop_reason",
    "max_inflight_at_stop",
    "admitted_tasks",
    "audit_overflow",
    "prep_ns",
    "l1_ns",
    "l2_ns",
    "math_ns",
    "completed_unique_tokens",
    "status",
    "mapping_version",
    "epoch",
    "workers",
    "service_union_ns",
    "execution_groups",
    "peak_inflight_blocks",
)


# Match include/iaas_kernels/early_moe/layout/early_moe.cuh exactly.
class _Call(C.Structure):
    _fields_ = (
        [
            (n, C.c_void_p)
            for n in (
                "hidden",
                "ids",
                "routing",
                "local_map",
                "reverse_map",
                "remaining",
                "done",
                "values",
                "early_count",
                "remaining_count",
                "queue",
                "workspace",
                "stats",
                "audit",
                "peers",
                "own_flags",
                "stop_signal",
            )
        ]
        + [(n, C.c_uint64) for n in ("epoch", "mapping_version", "timeout_ns")]
        + [
            (n, C.c_int)
            for n in (
                "tokens",
                "world",
                "rank",
                "workers",
                "audit_capacity",
                "diagnostic",
                "groups",
            )
        ]
        + [("clamp", C.c_float)]
    )


@lru_cache(maxsize=None)
def _library(geometry):
    lib = C.CDLL(str(build_library(geometry)))
    lib.early_error.restype = C.c_char_p
    lib.early_workspace_size.restype = C.c_size_t
    lib.early_call_size.restype = C.c_size_t
    try:
        lib.early_geometry.argtypes = [C.POINTER(C.c_int)]
        lib.early_geometry.restype = None
    except AttributeError as exc:
        raise RuntimeError(
            "rebuild this specialization: native geometry export is missing"
        ) from exc
    lib.early_create.argtypes = (
        [C.POINTER(C.c_void_p)] + [C.c_void_p] * 8 + [C.c_int] * 5
    )
    lib.early_run.argtypes = [C.c_void_p, C.POINTER(_Call), C.c_void_p]
    lib.early_busy.argtypes = lib.early_destroy.argtypes = [C.c_void_p]
    lib.early_info.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
    lib.early_export.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_uint64)]
    lib.early_open.argtypes = [
        C.c_void_p,
        C.c_uint64,
        C.c_int,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_void_p),
    ]
    lib.early_close.argtypes = [C.c_void_p]
    lib.early_delay.argtypes = [
        C.c_void_p,
        C.c_void_p,
        C.c_int,
        C.c_int,
        C.c_uint64,
        C.c_uint64,
        C.c_void_p,
    ]
    lib.early_signal_stop.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p]
    lib.early_delay_stop.argtypes = [
        C.c_void_p,
        C.c_void_p,
        C.c_int,
        C.c_int,
        C.c_uint64,
        C.c_uint64,
        C.c_void_p,
        C.c_void_p,
        C.c_void_p,
    ]
    if lib.early_call_size() != C.sizeof(_Call):
        raise RuntimeError("Python/native Call ABI mismatch")
    native = (C.c_int * len(geometry.abi))()
    lib.early_geometry(native)
    if tuple(native) != geometry.abi:
        raise RuntimeError(
            f"Python/native geometry mismatch: {geometry.abi} != {tuple(native)}"
        )
    return lib


def _check(code, library):
    if code < 0:
        raise RuntimeError(library.early_error().decode())
    return code


def _tensor(t, shape, dtype, device):
    if (
        tuple(t.shape) != tuple(shape)
        or t.dtype != dtype
        or t.device != device
        or (not t.is_contiguous())
    ):
        raise ValueError(
            f"expected contiguous {shape} {dtype} on {device}; got {tuple(t.shape)} {t.dtype} {t.device}"
        )
    if t.data_ptr() % 16:
        raise ValueError("16-byte aligned storage required")


@dataclass(frozen=True)
class ActiveWeights:
    """Already transformed, live IAAS FP8 tensors. No implicit packing."""

    l1: torch.Tensor
    l1_scale: torch.Tensor
    l2: torch.Tensor
    l2_scale: torch.Tensor
    layout: str = LAYOUT

    def validate(self, device, geometry):
        H, I, K, E, S = geometry.dimensions
        if self.layout != LAYOUT:
            raise ValueError("unsupported transformed weight ABI")
        for t, shape, dtype in (
            (self.l1, (S, 2 * I, H), torch.float8_e4m3fn),
            (self.l1_scale, (S, 2 * I // 128, H // 128), torch.float32),
            (self.l2, (S, H, I), torch.float8_e4m3fn),
            (self.l2_scale, (S, H // 128, I // 128), torch.float32),
        ):
            _tensor(t, shape, dtype, device)


class PeerEpochs:
    """IPC ownership is explicit; close collectively after GPU completion."""

    def __init__(self, group, library, *, external=False):
        self.lib = library
        self.group = group
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        self.world = dist.get_world_size(group) if dist.is_initialized() else 1
        if not 1 <= self.world <= 32:
            raise ValueError("Early-MoE supports 1..32 EP ranks")
        self.flags = torch.zeros(16, device="cuda", dtype=torch.uint64)
        if external:
            # An external collective supplies readiness through stop_signal.
            # Native peer polling remains local; no remote pointer is fabricated.
            self.bases = []
            self.table = torch.full(
                (self.world,), self.flags.data_ptr(), device="cuda", dtype=torch.uint64
            )
            return
        torch.cuda.synchronize()
        lib = self.lib
        handle, offset = (C.create_string_buffer(64), C.c_uint64())
        _check(lib.early_export(self.flags.data_ptr(), handle, C.byref(offset)), lib)
        item = (handle.raw, offset.value, torch.cuda.current_device())
        metadata = [None] * self.world
        if self.world > 1:
            dist.all_gather_object(metadata, item, group=group)
        else:
            metadata[0] = item
        self.bases, addresses = ([], [])
        for r, (h, off, device) in enumerate(metadata):
            if r == self.rank:
                addresses.append(self.flags.data_ptr())
            else:
                base, ptr = (C.c_void_p(), C.c_void_p())
                _check(lib.early_open(h, off, device, C.byref(base), C.byref(ptr)), lib)
                self.bases.append(base)
                addresses.append(ptr.value)
        self.table = torch.tensor(addresses, device="cuda", dtype=torch.uint64)
        torch.cuda.synchronize()
        if self.world > 1:
            dist.barrier(group=group)

    def delay(self, epoch, delay_us, stream=None):
        stream = stream or torch.cuda.current_stream()
        _check(
            self.lib.early_delay(
                self.table.data_ptr(),
                self.flags.data_ptr(),
                self.world,
                self.rank,
                epoch,
                round(delay_us * 1000),
                stream.cuda_stream,
            ),
            self.lib,
        )

    def delay_stop(self, epoch, delay_us, target_ready, signal, stream):
        """Benchmark-only concurrent producer of the external stop epoch."""
        _check(
            self.lib.early_delay_stop(
                self.table.data_ptr(),
                self.flags.data_ptr(),
                self.world,
                self.rank,
                epoch,
                round(delay_us * 1000),
                target_ready.data_ptr(),
                signal.data_ptr(),
                stream.cuda_stream,
            ),
            self.lib,
        )

    def close(self):
        torch.cuda.synchronize()
        if self.world > 1:
            dist.barrier(group=self.group)
        for base in self.bases:
            _check(self.lib.early_close(base), self.lib)
        self.bases.clear()
        if self.world > 1:
            dist.barrier(group=self.group)


@dataclass
class EarlyMoEResult:
    remaining_logical_ids: torch.Tensor
    early_done: torch.Tensor
    early_values: torch.Tensor
    early_count_by_expert: torch.Tensor
    remaining_count_by_expert: torch.Tensor
    stats: torch.Tensor
    audit: torch.Tensor
    status: torch.Tensor
    tokens: int = 0

    @property
    def remaining_physical_ids(self):
        """Preferred spelling; the stored legacy field uses physical IDs too."""
        return self.remaining_logical_ids


class SharedScratch:
    """Cold allocation shared by sequential layers on exactly one CUDA stream.

    The stream is pinned by the first launch and cannot later change. Every
    consumer (native combine, count copy and status/stat snapshot) must be enqueued on
    that stream before the next early call. A consumer on another stream must
    join back to the owner stream before scratch reuse. Per-layer contexts keep
    their own weight descriptors, IPC peers, native completion event and epoch.
    """

    def __init__(
        self,
        *,
        device,
        max_tokens,
        max_inflight_blocks,
        bulk_prepare,
        stats_capacity,
        audit_capacity,
        library,
        geometry,
    ):
        H, I, K, E, S = geometry.dimensions
        self.device = device
        self.spec = (
            geometry,
            max_tokens,
            max_inflight_blocks,
            bool(bulk_prepare),
            stats_capacity,
            audit_capacity,
        )
        self.enqueue_lock = threading.Lock()
        self.stream_id = None
        kw = dict(device=device)
        # Per-expert M64 padding is storage, not useful executed token count.
        self.input_rows = (
            (max_tokens * K + S * 63 + 127) // 128 * 128 if bulk_prepare else 0
        )
        self.a1 = torch.empty(
            (self.input_rows or 64 * max_inflight_blocks, H), dtype=torch.uint8, **kw
        )
        self.sf1 = torch.empty(
            (H // 128, self.input_rows or 128 * max_inflight_blocks),
            dtype=torch.float32,
            **kw,
        )
        self.a2 = torch.empty((64 * max_inflight_blocks, I), dtype=torch.uint8, **kw)
        self.sf2 = torch.empty(
            (I // 64, 128 * max_inflight_blocks), dtype=torch.float32, **kw
        )
        self.queue = torch.empty(max_tokens * K, dtype=torch.int32, **kw)
        self.workspace = torch.empty(
            library.early_workspace_size(), dtype=torch.uint8, **kw
        )
        self.stats = torch.empty((stats_capacity, 32), dtype=torch.int64, **kw)
        self.remaining = torch.empty((max_tokens, K), dtype=torch.int32, **kw)
        self.done = torch.empty((max_tokens, K), dtype=torch.uint8, **kw)
        self.values = torch.empty((max_tokens * K, H), dtype=torch.bfloat16, **kw)
        self.early_count = torch.empty(E, dtype=torch.int32, **kw)
        self.remaining_count = torch.empty(E, dtype=torch.int32, **kw)
        self.audit = torch.empty((audit_capacity, 8), dtype=torch.int64, **kw)
        self.status = self.workspace[8:12].view(torch.int32)
        # Precreate shape views on the cold path; launch allocates no GPU tensor.
        self.remaining_views = tuple(
            (self.remaining[:n] for n in range(max_tokens + 1))
        )
        self.done_views = tuple((self.done[:n] for n in range(max_tokens + 1)))

    def validate(self, device, spec):
        if self.device != device or self.spec != spec:
            raise ValueError(
                f"shared scratch configuration mismatch: {(self.device, self.spec)} != {(device, spec)}"
            )

    def claim_stream(self, stream_id):
        """Called with enqueue_lock held; no CUDA synchronization or allocation."""
        if self.stream_id is None:
            self.stream_id = stream_id
        elif self.stream_id != stream_id:
            raise RuntimeError(
                "shared early-MoE scratch is permanently owned by another CUDA stream"
            )


class EarlyMoEContext:
    def __init__(
        self,
        ep_group=None,
        *,
        max_tokens,
        hidden,
        intermediate,
        topk,
        logical_experts,
        local_slots,
        block_m=64,
        max_inflight_blocks=2,
        max_resident_ctas=None,
        stages=4,
        tile_n=256,
        stats_capacity=1,
        audit_capacity=1024,
        diagnostic=False,
        timeout_s=30,
        stop_signal=None,
        external_ready=False,
        bulk_prepare=False,
        shared_scratch=None,
    ):
        self.geometry = Geometry(
            hidden, intermediate, topk, logical_experts, local_slots
        )
        self.geometry.validate()
        H, I, K, E, S = self.geometry.dimensions
        if block_m != 64:
            raise ValueError("early computation uses M64 blocks")
        if (
            not isinstance(max_inflight_blocks, int)
            or not 1 <= max_inflight_blocks <= 16
        ):
            raise ValueError("max_inflight_blocks must be 1..16")
        if max_resident_ctas is None:
            sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            max_resident_ctas = min(65, max(2, sms - 2))
        if max_tokens < 0 or not 2 <= max_resident_ctas <= 65:
            raise ValueError("invalid token or cooperative grid capacity")
        if stages != 4 or tile_n != 256:
            raise ValueError("the supported fused path uses stages=4 and tile_n=256")
        workers = min(max_resident_ctas - 1, 64)
        if stats_capacity < 1 or audit_capacity < 0 or timeout_s <= 0:
            raise ValueError("invalid stats/audit capacity or fault timeout")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.lib = _library(self.geometry)
        self.max_tokens, self.workers, self.stages = (max_tokens, workers, stages)
        self.tile_n, self.max_inflight_blocks = (tile_n, max_inflight_blocks)
        self.stop_signal = stop_signal
        if stop_signal is not None:
            _tensor(stop_signal, (1,), torch.uint64, self.device)
        self.stats_capacity, self.audit_capacity = (stats_capacity, audit_capacity)
        self.diagnostic, self.timeout_ns = (diagnostic, int(timeout_s * 1000000000.0))
        self.bulk_prepare = bool(bulk_prepare)
        spec = (
            self.geometry,
            max_tokens,
            max_inflight_blocks,
            self.bulk_prepare,
            stats_capacity,
            audit_capacity,
        )
        if shared_scratch is not None and (
            not isinstance(shared_scratch, SharedScratch)
        ):
            raise TypeError("shared_scratch must be another context's .scratch")
        if shared_scratch is None:
            shared_scratch = SharedScratch(
                device=self.device,
                max_tokens=max_tokens,
                max_inflight_blocks=max_inflight_blocks,
                bulk_prepare=self.bulk_prepare,
                stats_capacity=stats_capacity,
                audit_capacity=audit_capacity,
                library=self.lib,
                geometry=self.geometry,
            )
        shared_scratch.validate(self.device, spec)
        self.scratch = scratch = shared_scratch
        for name in (
            "input_rows",
            "a1",
            "sf1",
            "a2",
            "sf2",
            "queue",
            "workspace",
            "stats",
        ):
            setattr(self, name, getattr(scratch, name))
        # Epoch flags are per layer even when temporary storage is shared.
        if external_ready and stop_signal is None:
            raise ValueError("External readiness requires a stop_signal tensor")
        self.peer = PeerEpochs(ep_group, self.lib, external=external_ready)
        if self.peer.world != self.geometry.world and self.peer.world != 1:
            raise ValueError("EP process group and expert shard geometry disagree")
        self.result = EarlyMoEResult(
            scratch.remaining,
            scratch.done,
            scratch.values,
            scratch.early_count,
            scratch.remaining_count,
            scratch.stats,
            scratch.audit,
            scratch.status,
        )
        self.remaining_buffer = self.result.remaining_logical_ids
        self.done_buffer = self.result.early_done
        self.remaining_views = scratch.remaining_views
        self.done_views = scratch.done_views
        self.native = C.c_void_p()
        self.epoch = 0
        self.lock = threading.Lock()
        self.weights = self.local_map = self.reverse_map = None
        self.version = None
        self.closed = False

    def bind(self, weights, local_expert_map, slot_to_logical, mapping_version):
        """Cold snapshot change: no old invocation may still be using its slots."""
        H, I, K, E, S = self.geometry.dimensions
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("activate the context's CUDA device before binding")
        with self.lock:
            if self.closed:
                raise RuntimeError("context is closed")
            if self.native.value and _check(self.lib.early_busy(self.native), self.lib):
                raise RuntimeError(
                    "cannot replace an in-flight mapping/weight snapshot"
                )
            weights.validate(self.device, self.geometry)
            _tensor(local_expert_map, (E,), torch.int32, self.device)
            _tensor(slot_to_logical, (S,), torch.int32, self.device)
            if not isinstance(mapping_version, int) or not 0 <= mapping_version < 2**63:
                raise ValueError(
                    "mapping_version must be a nonnegative signed-64 compatible integer"
                )
            lm, rm = (local_expert_map.cpu().tolist(), slot_to_logical.cpu().tolist())
            if any(
                (
                    s < -1 or s >= S or (s >= 0 and rm[s] != e)
                    for (e, s) in enumerate(lm)
                )
            ):
                raise ValueError("local and reverse mappings disagree")
            first = self.peer.rank * S
            expected_map = [
                p - first if first <= p < first + S else -1 for p in range(E)
            ]
            if lm != expected_map or rm != list(range(first, first + S)):
                raise ValueError(
                    "Physical-ID mode requires the rank's fixed physical identity maps"
                )
            replacement = C.c_void_p()
            _check(
                self.lib.early_create(
                    C.byref(replacement),
                    self.a1.data_ptr(),
                    self.sf1.data_ptr(),
                    self.a2.data_ptr(),
                    self.sf2.data_ptr(),
                    weights.l1.data_ptr(),
                    weights.l1_scale.data_ptr(),
                    weights.l2.data_ptr(),
                    weights.l2_scale.data_ptr(),
                    self.workers,
                    self.stages,
                    self.tile_n,
                    self.max_inflight_blocks,
                    self.input_rows,
                ),
                self.lib,
            )
            if self.native.value:
                _check(self.lib.early_destroy(self.native), self.lib)
            self.native = replacement
            self.weights, self.local_map, self.reverse_map = (
                weights,
                local_expert_map,
                slot_to_logical,
            )
            self.version = mapping_version
            values = (C.c_int * 9)()
            _check(self.lib.early_info(self.native, values), self.lib)
            self.launch_info = dict(
                zip(
                    (
                        "dynamic_shared_bytes",
                        "cooperative_capacity",
                        "registers",
                        "local_bytes",
                        "threads",
                        "ctas",
                        "tile_n",
                        "execution_groups",
                        "workers_per_group",
                    ),
                    values,
                )
            )

    def signal_stop(self, epoch, stream=None):
        """Enqueue a release epoch store AFTER the real external barrier.

        Use the independent communication stream; enqueueing behind the waiting
        compute kernel on the same stream cannot signal that running kernel.
        """
        if self.closed or self.stop_signal is None:
            raise RuntimeError("an open context with external stop_signal is required")
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("activate the context's CUDA device before signaling")
        if not isinstance(epoch, int) or not 0 < epoch < 2**63:
            raise ValueError(
                "stop epoch must be a positive signed-64 compatible integer"
            )
        stream = stream or torch.cuda.current_stream()
        _check(
            self.lib.early_signal_stop(
                self.stop_signal.data_ptr(), epoch, stream.cuda_stream
            ),
            self.lib,
        )

    def close(self):
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("activate the context's CUDA device before closing")
        with self.lock:
            if self.closed:
                return
            torch.cuda.synchronize()
            if self.native.value:
                _check(self.lib.early_destroy(self.native), self.lib)
                self.native = C.c_void_p()
            self.peer.close()
            self.closed = True


def create_early_moe_context(ep_group=None, **kwargs):
    return EarlyMoEContext(ep_group, **kwargs)


def early_moe(
    hidden_states,
    logical_topk_ids,
    effective_topk_weights,
    active_weights,
    local_expert_map,
    mapping_version,
    invocation_epoch,
    *,
    context,
    stats_index=0,
    activation_clamp=10.0,
):
    """Execute physical routing IDs despite the legacy logical_* argument names.

    Enqueue all consumers of the result before reusing context.scratch. The
    adapter supplies effective gate weights, including any routed scale once.
    """
    c = context
    H, I, K, E, S = c.geometry.dimensions
    if torch.cuda.current_device() != c.device.index:
        raise ValueError("activate the context's CUDA device before launching")
    activation_clamp = math.inf if activation_clamp is None else float(activation_clamp)
    if math.isnan(activation_clamp) or activation_clamp < 0:
        raise ValueError("activation_clamp must be nonnegative or None")
    if not c.lock.acquire(blocking=False):
        raise RuntimeError("concurrent calls on one context are forbidden")
    scratch_locked = False
    try:
        if not c.scratch.enqueue_lock.acquire(blocking=False):
            raise RuntimeError(
                "concurrent host submissions sharing early-MoE scratch are forbidden"
            )
        scratch_locked = True
        if c.closed or not c.native.value:
            raise RuntimeError("context must be open and bound before launch")
        if (
            c.weights is not active_weights
            or c.local_map is not local_expert_map
            or c.version != mapping_version
        ):
            raise ValueError(
                "bind this immutable active snapshot on the cold path first"
            )
        n = hidden_states.shape[0]
        if n > c.max_tokens or not 0 <= stats_index < c.stats_capacity:
            raise ValueError("token/stats capacity exceeded")
        if (
            not isinstance(invocation_epoch, int)
            or not c.epoch < invocation_epoch < 2**63
        ):
            raise ValueError(
                "invocation_epoch must strictly increase and be shared by the EP group"
            )
        _tensor(hidden_states, (n, H), torch.bfloat16, c.device)
        _tensor(logical_topk_ids, (n, K), torch.int32, c.device)
        _tensor(effective_topk_weights, (n, K), torch.float32, c.device)
        r = c.result
        call = _Call(
            hidden_states.data_ptr(),
            logical_topk_ids.data_ptr(),
            effective_topk_weights.data_ptr(),
            c.local_map.data_ptr(),
            c.reverse_map.data_ptr(),
            c.remaining_buffer.data_ptr(),
            c.done_buffer.data_ptr(),
            r.early_values.data_ptr(),
            r.early_count_by_expert.data_ptr(),
            r.remaining_count_by_expert.data_ptr(),
            c.queue.data_ptr(),
            c.workspace.data_ptr(),
            c.stats.data_ptr() + stats_index * 32 * 8,
            r.audit.data_ptr(),
            c.peer.table.data_ptr(),
            c.peer.flags.data_ptr(),
            c.stop_signal.data_ptr() if c.stop_signal is not None else 0,
            invocation_epoch,
            mapping_version,
            c.timeout_ns,
            n,
            c.peer.world,
            c.peer.rank,
            c.workers,
            c.audit_capacity,
            int(c.diagnostic),
            c.max_inflight_blocks,
            float(activation_clamp),
        )
        stream_id = torch.cuda.current_stream().cuda_stream
        c.scratch.claim_stream(stream_id)
        _check(c.lib.early_run(c.native, C.byref(call), stream_id), c.lib)
        c.epoch = invocation_epoch
        r.tokens = n
        # These are aliases, valid only until the next call sharing the arena.
        r.remaining_logical_ids = c.remaining_views[n]
        r.early_done = c.done_views[n]
        return r
    finally:
        if scratch_locked:
            c.scratch.enqueue_lock.release()
        c.lock.release()
