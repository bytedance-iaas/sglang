"""SiDP manager for IPC weight sharing and bounded cycle prefetch.

The cycle pipeline is shared by eager execution and CUDA Graph capture. Each
forward starts with cycle 0 resident, overlaps cycle c with prefetch(c + 1),
and leaves the next forward's cycle 0 resident at graph/forward completion.
"""

import logging
import pickle
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed

from sglang.srt.layers.sidp.config import (
    SidpConfig,
    SidpCopyBackend,
    SidpDynamicClaimOrder,
    SidpPrefetchPolicy,
    SidpSlotSync,
)
from sglang.srt.layers.sidp.cuda_memcpy import SidpCudaMemcpy
from sglang.srt.layers.sidp.fixed_dma_event_backend import SidpFixedDmaEventBackend
from sglang.srt.layers.sidp.fixed_dma_flag_backend import SidpFixedDmaFlagBackend
from sglang.srt.layers.sidp.graph_profiler import SidpGraphProfiler
from sglang.srt.layers.sidp.scheduler import (
    is_local_layer,
    next_forward_cycle_zero_generations,
    owner_of,
    prefetch_order,
    remote_positions,
)
from sglang.srt.layers.sidp.cycle_backend import SidpCycleBackend
from sglang.srt.layers.sidp.sm_backend import SidpSmBackend
from sglang.srt.layers.sidp.sync_strategy import (
    NoSyncStrategy,
    build_peak_sync_strategy,
)
from sglang.srt.layers.sidp.weight_codec import (
    EncodedWeight,
    WeightComputeMode,
    build_weight_codec,
)

logger = logging.getLogger(__name__)


def _reduce_tensor(t: torch.Tensor):
    """Serialize a tensor into an IPC-safe (fn, args) tuple via torch's reduce_tensor."""
    from torch.multiprocessing.reductions import reduce_tensor

    return reduce_tensor(t)


def _rebuild_tensor(reduced_tensor, src_device: int) -> torch.Tensor:
    """Rebuild a tensor from a reducer ``(fn, args)`` on its source device."""
    fn, args = reduced_tensor
    with torch.cuda.device(src_device):
        return fn(*args)


class SidpManager:
    """Central manager for SiDP weight sharing across DP ranks."""

    def __init__(self, config: SidpConfig):
        self.config = config
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        self.external_mode = config.external_mode
        self.coord_mode = config.coord_mode
        self.barrier_interval_cycles = config.barrier_interval_cycles
        # 0 means "all members"; otherwise a fixed subset of M members runs the
        # barrier (the rest must stay idle). Resolved to dp_size at setup.
        self.barrier_nptr = config.barrier_nptr
        # Dynamic nptr: derive the live participant count from a per-forward
        # host rendezvous instead of the fixed barrier_nptr.
        self.coord_dynamic_nptr = config.coord_dynamic_nptr
        # Unified scheduling: all members agree on one forward mode per round.
        self.coord_unified_schedule = getattr(
            config, "coord_unified_schedule", False
        )
        self.coord_prefill_low_watermark = getattr(
            config, "coord_prefill_low_watermark", 0.15
        )
        # Ablation: keep host rendezvous but skip the device-barrier launch.
        self.coord_disable_device_barrier = getattr(
            config, "coord_disable_device_barrier", False
        )
        # Last unified mode decided this iteration ("d"/"p"/"i"), read by the
        # scheduler to gate the outer decode branch and by the observer.
        self._last_unified_mode = None
        # Schedule-consistency observability (read-only diagnostics).
        self.coord_observe = getattr(config, "coord_observe", False)
        self.coord_observe_dir = getattr(
            config, "coord_observe_dir", "check_logs/sidp_sched_trace"
        )
        self._observer = None
        self.k = config.k
        self.cache_cycles = config.cache_cycles
        self.num_layers = config.num_layers
        self.enable_cycle_overlap = config.enable_cycle_overlap
        self.prefetch_policy = config.prefetch_policy
        if (
            config.enable_peak_shifting
            and self.prefetch_policy == SidpPrefetchPolicy.COMPUTE.value
        ):
            self.prefetch_policy = SidpPrefetchPolicy.STATIC_PEAK.value
        self.copy_backend = config.copy_backend
        self.slot_sync = config.slot_sync
        if self.slot_sync not in {mode.value for mode in SidpSlotSync}:
            raise ValueError(f"invalid SiDP slot_sync: {self.slot_sync}")
        self.dynamic_claim_order = config.dynamic_claim_order
        if self.dynamic_claim_order not in {
            order.value for order in SidpDynamicClaimOrder
        }:
            raise ValueError(
                f"invalid SiDP dynamic_claim_order: {self.dynamic_claim_order}"
            )
        if (
            self.dynamic_claim_order != SidpDynamicClaimOrder.ROTATING.value
            and self.prefetch_policy != SidpPrefetchPolicy.DYNAMIC_OWNER.value
        ):
            raise ValueError("SiDP compute_priority claim order requires dynamic_owner")
        if config.sm_use_event_sync and self.slot_sync != SidpSlotSync.EVENT.value:
            raise ValueError(
                "SiDP sm_use_event_sync compatibility input requires slot_sync=event"
            )
        self.sm_use_event_sync = (
            self.copy_backend == SidpCopyBackend.SM.value
            and self.slot_sync == SidpSlotSync.EVENT.value
        )
        self.dma_slices = config.dma_slices
        self.dma_slice_groups = config.dma_slice_groups
        if self.dma_slices < 1:
            raise ValueError("SiDP dma_slices must be positive")
        if self.dma_slice_groups < 1:
            raise ValueError("SiDP dma_slice_groups must be positive")
        if self.dma_slices == 1 and self.dma_slice_groups != 1:
            raise ValueError("SiDP dma_slice_groups only applies when dma_slices > 1")
        if self.dma_slices > 1 and not (
            self.prefetch_policy == SidpPrefetchPolicy.COMPUTE.value
            and self.copy_backend == SidpCopyBackend.DMA.value
            and self.enable_cycle_overlap
            and self.k < self.dp_size
        ):
            raise ValueError(
                "SiDP DMA slicing requires fixed compute-order DMA, cycle "
                "overlap, and remote weights"
            )
        self.sm_copy_ctas = config.sm_copy_ctas
        if self.sm_copy_ctas < 0:
            raise ValueError("SiDP sm_copy_ctas must be non-negative")
        if self.sm_copy_ctas and self.copy_backend != SidpCopyBackend.SM.value:
            raise ValueError("SiDP sm_copy_ctas is only valid with the SM backend")
        if self.sm_use_event_sync and not (
            self.copy_backend == SidpCopyBackend.SM.value
            and self.prefetch_policy == SidpPrefetchPolicy.COMPUTE.value
        ):
            raise ValueError(
                "SiDP sm_use_event_sync is a validation-only mode and "
                "requires compute + sm"
            )
        self._uses_conditional_dma = (
            self.prefetch_policy == SidpPrefetchPolicy.DYNAMIC_OWNER.value
            and self.copy_backend == SidpCopyBackend.DMA.value
        )
        if (
            self.prefetch_policy == SidpPrefetchPolicy.DYNAMIC_OWNER.value
            and self.slot_sync != SidpSlotSync.FLAG.value
        ):
            raise ValueError("SiDP dynamic_owner requires flag slot synchronization")
        if (
            self.prefetch_policy == SidpPrefetchPolicy.DYNAMIC_OWNER.value
            and config.peak_sync_strategy != "none"
        ):
            raise ValueError("SiDP dynamic_owner does not allow DP rank synchronization")
        self._needs_raw_peer_ipc = (
            self.copy_backend == SidpCopyBackend.SM.value or self._uses_conditional_dma
        )
        # K==D has no remote slots and therefore needs neither Event nor flag
        # state even if the resolved configuration says flag.
        self._uses_flag_sync = (
            self.slot_sync == SidpSlotSync.FLAG.value and self.k < self.dp_size
        )
        self.enable_peak_shifting = (
            self.prefetch_policy == SidpPrefetchPolicy.STATIC_PEAK.value
        )
        if self._needs_raw_peer_ipc and not self.enable_cycle_overlap:
            raise ValueError("SiDP SM/conditional DMA requires the cycle-overlap pipeline")
        self.enable_debug_logging = config.enable_debug_logging
        self.enable_graph_profiling = config.enable_graph_profiling
        self.profile_dummy_compute = config.profile_dummy_compute
        self.transfer_dtype = config.transfer_dtype
        self.weight_codec = build_weight_codec(self.transfer_dtype)
        self.peak_sync_strategy = config.peak_sync_strategy
        self.peak_sync_min_raw_bs = config.peak_sync_min_raw_bs
        self.peak_sync_max_replays = config.peak_sync_max_replays
        self.peak_sync_timeout_s = config.peak_sync_timeout_s

        # D2: TCPStore is created lazily in setup() so that all ranks
        # have finished load_model() before any rank tries to connect.
        self.store = None
        self._rdzv_host = config.rdzv_host
        self._rdzv_port = config.rdzv_port

        # D7: DMA engine wrapper
        self.memcpy = SidpCudaMemcpy()

        # D7/D8: asynchronous cycle-fill stream, captured alongside the model
        # stream when CUDA Graph is enabled.
        self.comm_stream = torch.cuda.Stream()

        # Per-slot RAW/WAR events for the bounded cycle ring.
        self._prefetch_events: List[torch.cuda.Event] = []
        self._consume_events: List[torch.cuda.Event] = []

        # Populated by setup()
        self.peer_views: Dict[int, Dict[str, EncodedWeight]] = {}
        self._peer_sm_ipc: Dict[int, Dict[str, Dict[str, dict]]] = {}
        self.buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        self._transfer_buffers: Dict[int, Dict[str, EncodedWeight]] = {}
        self._materialization_buffers: Dict[str, torch.Tensor] = {}
        self._layer_to_slot: Dict[int, int] = {}
        self._non_local_layers: List[int] = []
        self._fetch_schedule: List[int] = []
        self._remote_positions: List[int] = []
        self._remote_position_to_index: Dict[int, int] = {}
        self._cycle_layers: Dict[int, List[int]] = {}
        self._last_non_local_in_cycle: Dict[int, int] = {}
        self._num_cycles = 0
        self._cycle_cache_depth = 0
        self._queued_cycles: set[int] = set()
        self._next_forward_cycle_zero_queued = False
        self._layers_ref: Dict[int, Any] = {}
        self._ipc_refs: List[torch.Tensor] = []
        self._local_encoded_weights: Dict[int, Dict[str, EncodedWeight]] = {}
        self._local_encoded_refs: List[torch.Tensor] = []
        self._graph_profiler: SidpGraphProfiler | None = None
        self._cycle_backend: (
            SidpCycleBackend
            | SidpFixedDmaFlagBackend
            | SidpFixedDmaEventBackend
            | None
        ) = None
        self._launch_sync_strategy = NoSyncStrategy()

        # Direction A coordinated_static device barrier (Phase 2). Populated by
        # _setup_device_barrier() only when coord_mode is enabled.
        self._device_barrier = None
        self._bar_state: torch.Tensor | None = None  # int[2] arrive/sense
        self._nptr: torch.Tensor | None = None  # int[NPTR_RING_DEPTH] participant ring
        self._bar_ptr: int = 0
        self._nptr_ptr: int = 0
        self._barrier_refs: List[torch.Tensor] = []
        self._barrier_active_this_forward: bool = False
        # Generational nptr ring (Direction A dynamic nptr). Depth must exceed the
        # max in-flight forwards (overlap schedule looks ~1 ahead); 4 is safe and
        # trivially cheap. bpf = barriers launched per forward (set at setup); the
        # kernel reads slot (bar[1] / bpf) % depth so a later forward's nptr write
        # lands in a different slot and cannot clobber an in-flight barrier.
        self._nptr_ring_depth: int = 4
        self._barriers_per_forward: int = 0
        self._barrier_round: int = 0  # host-side count of decode forwards launched
        self._rendezvous = None  # CoordRendezvous when coord_dynamic_nptr

    def setup(self, model, model_runner=None):
        """Call after model weights are loaded. Exchanges IPC handles, releases
        non-local weights, allocates rolling buffers, primes P2P routes, and
        rebinds weight.data to prefetch buffers.

        The released storage is intentionally reflected in
        ``available_gpu_memory`` while ``pre_model_load_memory`` remains
        unchanged. This preserves the configured activation slack and lets the
        KV-cache profiler assign the newly available HBM to the KV pool.

        ``model_runner`` is accepted for call-site compatibility but is
        intentionally not mutated.
        """

        # External-worker mode (Direction A) forms the SiDP group from N
        # independent services. Cross-process CUDA IPC rebuilds a peer tensor on
        # ``src_device = owner_of(lid) = owner member rank`` and enables peer
        # access for ``dev in range(dp_size)``. Both use the member rank directly
        # as a physical CUDA ordinal, so this only works when member rank equals
        # the process's visible CUDA ordinal. server_args enforces
        # base_gpu_id == member_rank; assert the resulting invariant here so a
        # misconfigured launch fails loudly instead of doing silent cross-device
        # IPC. (Native mode already satisfies rank == ordinal by construction.)
        if self.external_mode:
            current_ordinal = torch.cuda.current_device()
            if current_ordinal != self.dp_rank:
                raise RuntimeError(
                    "SiDP external-worker mode requires this service's visible "
                    f"CUDA ordinal ({current_ordinal}) to equal its SiDP member "
                    f"rank ({self.dp_rank}). Launch each member with "
                    "base_gpu_id == member_rank and without per-process CUDA "
                    "device reindexing."
                )

        # D2: Create TCPStore now (all ranks have finished load_model at this point).
        # Rank 0 is master. Non-master ranks retry connection for up to 300s.
        if self.enable_debug_logging:
            logger.info(
                f"[SiDP rank{self.dp_rank}] creating TCPStore "
                f"(host={self._rdzv_host}, port={self._rdzv_port})..."
            )
        self.store = torch.distributed.TCPStore(
            host_name=self._rdzv_host,
            port=self._rdzv_port,
            world_size=self.dp_size,
            is_master=(self.dp_rank == 0),
            wait_for_workers=False,
        )
        if self.enable_debug_logging:
            logger.info(f"[SiDP rank{self.dp_rank}] TCPStore connected")
        self._launch_sync_strategy = build_peak_sync_strategy(
            self.peak_sync_strategy,
            enabled=self.enable_peak_shifting,
            store=self.store,
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
            min_raw_bs=self.peak_sync_min_raw_bs,
            max_replays=self.peak_sync_max_replays,
            timeout_s=self.peak_sync_timeout_s,
        )
        if self.enable_debug_logging:
            logger.info(
                f"[SiDP rank{self.dp_rank}] peak sync strategy: "
                f"{self._launch_sync_strategy.name}"
            )

        layers = self._collect_decoder_layers(model)
        self._layers_ref = layers

        if not layers:
            if self.enable_debug_logging:
                logger.info(
                    f"[SiDP rank{self.dp_rank}] no decoder layers found, "
                    "skipping setup"
                )
            return

        self.num_layers = max(layers.keys()) + 1
        if self.enable_debug_logging:
            logger.info(
                f"[SiDP rank{self.dp_rank}] setup: {len(layers)} layers, "
                f"num_layers={self.num_layers}, dp_size={self.dp_size}, k={self.k}"
            )

        # Identify local vs non-local layers
        local_layers = []
        non_local_layers = []
        for lid in sorted(layers.keys()):
            if is_local_layer(lid, self.dp_rank, self.dp_size, self.k):
                local_layers.append(lid)
            else:
                non_local_layers.append(lid)
        self._non_local_layers = non_local_layers
        self._build_cycle_schedule()
        if self.enable_graph_profiling:
            self._graph_profiler = SidpGraphProfiler(
                dp_rank=self.dp_rank,
                dp_size=self.dp_size,
                num_cycles=self._num_cycles,
                cycle_layers=self._cycle_layers,
                sample_interval=self.config.profile_sample_interval,
                warmup_replays=self.config.profile_warmup_replays,
                output_dir=self.config.profile_output_dir,
                peak_shifting=self.enable_peak_shifting,
                prefetch_policy=self.prefetch_policy,
                copy_backend=self.copy_backend,
                dummy_compute=self.profile_dummy_compute,
                sync_strategy=self._launch_sync_strategy.name,
                weight_codec=self.weight_codec.name,
            )
            logger.warning(
                f"[SiDP rank{self.dp_rank}] CUDA Graph profiling enabled; "
                "timing events and sampled synchronization perturb performance. "
                f"Diagnostics will be written to {self._graph_profiler.path}"
            )
        if self.enable_debug_logging:
            mode = (
                "cross-forward-cycle-overlap"
                if self.enable_cycle_overlap
                else "serial-graph-safe"
            )
            logger.info(
                f"[SiDP rank{self.dp_rank}] local={len(local_layers)}, "
                f"non_local={len(non_local_layers)}, "
                f"mode={mode}, policy={self.prefetch_policy}, "
                f"copy_backend={self.copy_backend}, "
                f"claim_order={self.dynamic_claim_order}, "
                f"slot_sync={self.slot_sync}, "
                f"dma_slices={self.dma_slices}, "
                f"dma_slice_groups={self.dma_slice_groups}"
            )

        # D3/D10: Materialize every locally retained weight representation;
        # only canonical owners export theirs through IPC. Identity transport
        # returns the original model weight. A real codec must make its encoded
        # main/extra tensors canonical local storage (not retain a second
        # persistent BF16 copy). Static inference weights are encoded once at
        # setup, not once per prefetch.
        torch.cuda.synchronize()  # D11: make model-loader writes IPC-visible.
        codec_stream = torch.cuda.current_stream()
        for lid in local_layers:
            layer = layers[lid]
            self._local_encoded_weights[lid] = {}
            for pname, param in self._get_ffn_params(layer):
                encoded = self.weight_codec.encode_for_storage(
                    layer_id=lid,
                    param_name=pname,
                    weight=param.data,
                    stream=codec_stream,
                )
                self._validate_encoded_weight(
                    encoded, context=f"local layer={lid}, param={pname}"
                )
                # Validate metadata even for a k>1 local replica so every local
                # encoded weight follows one lifecycle contract. CUDA tensor
                # metadata needs a separate IPC lifecycle and is unsupported.
                pickle.dumps(encoded.metadata)
                self._local_encoded_weights[lid][pname] = encoded
                self._local_encoded_refs.append(encoded.tensor)
                self._local_encoded_refs.extend(encoded.extra_tensors.values())

        # D11: all encode kernels must complete before peers can consume the
        # published IPC storage. One setup-time synchronization covers them all.
        codec_stream.synchronize()
        if self.enable_debug_logging:
            logger.info(
                f"[SiDP rank{self.dp_rank}] publishing IPC handles "
                f"(weight_codec={self.weight_codec.name})..."
            )
        for lid, encoded_params in self._local_encoded_weights.items():
            if owner_of(lid, self.dp_size) != self.dp_rank:
                continue
            for pname, encoded in encoded_params.items():
                tensor_reduction = _reduce_tensor(encoded.tensor)
                extra_reductions = {
                    name: _reduce_tensor(tensor)
                    for name, tensor in encoded.extra_tensors.items()
                }
                handle = pickle.dumps(
                    {
                        "version": (
                            3
                            if self._needs_raw_peer_ipc
                            else 2
                        ),
                        "codec": self.weight_codec.name,
                        "tensor": tensor_reduction,
                        "extra_tensors": extra_reductions,
                        "metadata": encoded.metadata,
                        "sm_ipc": (
                            {
                                "<main>": self.memcpy.export_ipc_pointer(
                                    encoded.tensor.data_ptr(), encoded.tensor.nbytes
                                ),
                                **{
                                    name: self.memcpy.export_ipc_pointer(
                                        tensor.data_ptr(), tensor.nbytes
                                    )
                                    for name, tensor in encoded.extra_tensors.items()
                                },
                            }
                            if self._needs_raw_peer_ipc
                            else None
                        ),
                    }
                )
                self.store.set(f"sidp/{self.dp_rank}/{lid}/{pname}", handle)
        if self.enable_debug_logging:
            published_layers = sum(
                1
                for layer_id in local_layers
                if owner_of(layer_id, self.dp_size) == self.dp_rank
            )
            logger.info(
                f"[SiDP rank{self.dp_rank}] published handles for "
                f"{published_layers} layers"
            )

        # D3: Rebuild peer views for non-local layers
        if self.enable_debug_logging:
            logger.info(f"[SiDP rank{self.dp_rank}] fetching peer handles...")
        for lid in non_local_layers:
            src = owner_of(lid, self.dp_size)
            self.peer_views[lid] = {}
            self._peer_sm_ipc[lid] = {}
            for pname, _ in self._get_ffn_params(layers[lid]):
                key = f"sidp/{src}/{lid}/{pname}"
                payload = self.store.get(key)
                wire_payload = pickle.loads(payload)
                # Accept the pre-codec identity payload for easier rolling
                # upgrades, while all new ranks publish the versioned format.
                if isinstance(wire_payload, dict):
                    version = wire_payload.get("version")
                    if version not in (1, 2, 3):
                        raise RuntimeError(
                            "Unsupported SiDP weight payload version: "
                            f"{version}"
                        )
                    if wire_payload.get("codec") != self.weight_codec.name:
                        raise RuntimeError(
                            "SiDP weight codec mismatch across ranks: "
                            f"local={self.weight_codec.name}, "
                            f"remote={wire_payload.get('codec')}"
                        )
                    reduced_tensor = wire_payload["tensor"]
                    reduced_extras = (
                        wire_payload.get("extra_tensors", {})
                        if version in (2, 3)
                        else {}
                    )
                    metadata = wire_payload.get("metadata")
                    sm_ipc = wire_payload.get("sm_ipc")
                else:
                    if self.weight_codec.name != "identity":
                        raise RuntimeError(
                            "Legacy SiDP IPC payload is only valid for identity codec"
                        )
                    reduced_tensor = wire_payload
                    reduced_extras = {}
                    metadata = None
                    sm_ipc = None
                if self._needs_raw_peer_ipc:
                    if not isinstance(sm_ipc, dict):
                        raise RuntimeError(
                            "SiDP copy requires raw requester-context IPC "
                            f"descriptors: layer={lid}, param={pname}"
                        )
                    expected_components = {"<main>", *reduced_extras}
                    if set(sm_ipc) != expected_components:
                        raise RuntimeError(
                            "SiDP IPC component schema mismatch: "
                            f"layer={lid}, param={pname}, "
                            f"expected={sorted(expected_components)}, "
                            f"actual={sorted(sm_ipc)}"
                        )
                    self._peer_sm_ipc[lid][pname] = sm_ipc
                peer_view = _rebuild_tensor(reduced_tensor, src_device=src)
                extra_views = {
                    name: _rebuild_tensor(reduced, src_device=src)
                    for name, reduced in reduced_extras.items()
                }
                encoded_view = EncodedWeight(
                    tensor=peer_view,
                    extra_tensors=extra_views,
                    metadata=metadata,
                )
                self._validate_encoded_weight(
                    encoded_view, context=f"peer layer={lid}, param={pname}"
                )
                self.peer_views[lid][pname] = encoded_view
                self._ipc_refs.append(peer_view)  # D11: prevent GC
                self._ipc_refs.extend(extra_views.values())
        if self.enable_debug_logging:
            logger.info(
                f"[SiDP rank{self.dp_rank}] rebuilt "
                f"{len(non_local_layers)} peer views"
            )

        # Allocate buffers BEFORE releasing weights (their shapes are still
        # needed here). Cycle overlap uses cycle_cache_depth * (D-k) slots;
        # graph-safe fallback keeps the original cache_cycles layer slots.
        self._alloc_buffers(layers, non_local_layers)
        if self.enable_debug_logging:
            logger.info(
                f"[SiDP rank{self.dp_rank}] allocated "
                f"{len(self.buffers)} buffer slots"
            )

        # Key design: rebind weight.data to the rolling buffer, THEN release the
        # original full-weight storage. This way self.mlp(x) automatically reads
        # from the prefetch buffer, and the HBM held by the original non-local
        # weights is handed back to the caching allocator (the whole point of SiDP).
        # Optional diagnostic sampling is deliberately outside the default
        # path. The post-release empty_cache below remains functional: downstream
        # KV profiling uses device-level free memory and must see released pages.
        if self.enable_debug_logging:
            torch.cuda.empty_cache()
            free_before, total_dev = torch.cuda.mem_get_info()
        freed_bytes = 0
        for lid in non_local_layers:
            layer = layers[lid]
            slot = self._layer_to_slot[lid]
            for pname, param in self._get_ffn_params(layer):
                # Grab the original storage BEFORE rebinding (after rebind, param
                # points at the shared buffer and must NOT be resized).
                orig = param.data
                if self.enable_debug_logging:
                    freed_bytes += orig.numel() * orig.element_size()
                # Point weight.data at the buffer (zero-copy rebind)
                param.data = self.buffers[slot][pname]
                # Release the original full weight's HBM back to the allocator.
                orig.untyped_storage().resize_(0)
                del orig
        # Return the freed blocks to the driver so the downstream KV-cache
        # profiling (mem_get_info) actually sees the reclaimed memory.
        torch.cuda.empty_cache()
        if self.enable_debug_logging:
            # If delta ~= logical freed_bytes, HBM physically returned to the
            # device pool. A near-zero delta points to retained IPC mappings or
            # other contexts pinning the pages.
            free_after, _ = torch.cuda.mem_get_info()
            logger.info(
                f"[SiDP rank{self.dp_rank}] released non-local weight storage: "
                f"logical={freed_bytes / (1024 ** 3):.2f} GB | "
                f"device free {free_before / (1024 ** 3):.2f} -> "
                f"{free_after / (1024 ** 3):.2f} GB "
                f"(delta={(free_after - free_before) / (1024 ** 3):+.2f} GB) | "
                f"total={total_dev / (1024 ** 3):.1f} GB"
            )

        # Note: we deliberately do NOT patch model_runner.pre_model_load_memory.
        # sglang's KV budget formula is
        #   rest = available_gpu_memory - pre_model_load_memory * (1 - mem_fraction_static)
        # After the release above, available_gpu_memory naturally rises by ~freed
        # GB while slack (anchored on the pre-load baseline) stays fixed. The KV
        # pool therefore absorbs the freed HBM directly, leaving activation slack
        # untouched — total device usage stays close to baseline, only KV grows.

        # D6: Enable peer access + prime P2P routes
        if self.enable_debug_logging:
            logger.info(
                f"[SiDP rank{self.dp_rank}] enabling peer access + priming routes..."
            )
        for dev in range(self.dp_size):
            if dev != self.dp_rank:
                self.memcpy.enable_peer_access(dev)
        self._prime_routes(non_local_layers)

        if self._uses_conditional_dma:
            from sglang.srt.layers.sidp.dma_graph_backend import SidpDmaGraphBackend

            self._cycle_backend = SidpDmaGraphBackend(self)
        elif self.copy_backend == SidpCopyBackend.SM.value:
            self._cycle_backend = SidpSmBackend(self)
        elif self._uses_flag_sync:
            self._cycle_backend = SidpFixedDmaFlagBackend(self)
        elif self.dma_slices > 1:
            self._cycle_backend = SidpFixedDmaEventBackend(self)

        # Initial WAR state: every slot is safe to write before its first use.
        if not self._uses_flag_sync:
            for evt in self._consume_events:
                evt.record(torch.cuda.current_stream())

        if self.enable_cycle_overlap and self._cycle_layers.get(0):
            self._initialize_cycle_zero()

        # Direction A coordinated_static: allocate + IPC-share the cross-member
        # barrier state after routes are primed and cycle 0 is resident.
        if self.coord_mode:
            self._setup_device_barrier()

        # All layers know the forward boundary manager; the current identity
        # path only needs per-layer RAW/WAR hooks on non-local layers. A real
        # compressed codec must bind *all* layers so local owner weights are
        # also decoded into the one rank-global BF16 materialization buffer
        # immediately before their GEMM.
        for lid, layer in layers.items():
            layer._sidp_bound = lid in self.peer_views
            layer._sidp_mgr = self
            layer._sidp_begin_forward = lid == 0
            layer._sidp_end_forward = lid == self.num_layers - 1
            layer._sidp_profile_enabled = self._graph_profiler is not None
            layer._sidp_dummy_compute = self.profile_dummy_compute

        if self.enable_debug_logging:
            logger.info(f"[SiDP rank{self.dp_rank}] setup complete")

    def _setup_device_barrier(self):
        """Allocate + IPC-share the coordinated_static cross-member barrier.

        The barrier state ``bar`` (int[2]: arrive count, sense/generation) lives
        on the canonical member 0's device and is shared to every member via CUDA
        IPC, so all members atomically arrive/spin on the same words. Each member
        keeps a local ``nptr`` (participant count); the basic version fixes it to
        the full world size. All members must build the barrier component (it JIT
        compiles once) so the kernel node exists in every captured graph.
        """
        from sglang.srt.layers.sidp.device_barrier import SidpDeviceBarrier

        self._device_barrier = SidpDeviceBarrier()

        # Shared bar[2] via raw CUDA IPC (alloc/get/open done natively in the
        # component so the 64-byte cudaIpcMemHandle_t is created and passed by
        # value correctly). The barrier kernel dereferences the peer pointer
        # directly with atomicAdd_system, which needs cudaIpcOpenMemHandle(
        # LazyEnablePeerAccess) -- not torch reduce_tensor (copy-engine only).
        bar_key = "sidp/coord/bar_handle"
        nbytes = self._device_barrier.barrier_state_size_bytes()
        if self.dp_rank == 0:
            self._bar_ptr, hex_handle = self._device_barrier.alloc_shared_bar(nbytes)
            self.store.set(bar_key, hex_handle)
        else:
            raw = self.store.get(bar_key)
            hex_handle = raw.decode("ascii") if isinstance(raw, bytes) else raw
            self._bar_ptr = self._device_barrier.open_shared_bar(hex_handle)

        # Participant count read at kernel runtime (device int, so a future
        # rendezvous can lower it without recapturing the graph). 0 resolves to
        # the full world; a smaller M means only M members run the barrier while
        # the rest stay idle -- validates partial participation before dynamic
        # nptr exists.
        #
        # Ring of depth NPTR_RING_DEPTH (not a single int): with dynamic nptr the
        # scheduler writes the next forward's participant count while the previous
        # forward's barrier may still be in flight on comm_stream. A single slot
        # would be clobbered mid-barrier (the deadlock we hit). Each forward reads
        # its own slot -- kernel index (bar[1] / bpf) % depth -- so writes for a
        # later forward land elsewhere. Every slot starts at resolved_nptr so the
        # static-nptr path (no rendezvous) is correct for any bar[1] the barrier
        # reaches.
        self._barriers_per_forward = self._count_barriers_per_forward()
        resolved_nptr = self.barrier_nptr if self.barrier_nptr > 0 else self.dp_size
        self._nptr = torch.full(
            (self._nptr_ring_depth,), resolved_nptr, dtype=torch.int32, device="cuda"
        )
        self._nptr_ptr = self._nptr.data_ptr()

        # Barrier ordering across members: everyone must have opened the shared
        # bar before any member arrives, or an early arriver could spin on a
        # not-yet-mapped generation word.
        torch.cuda.synchronize()
        try:
            self.store.set(f"sidp/coord/bar_ready/{self.dp_rank}", "1")
            self.store.wait(
                [f"sidp/coord/bar_ready/{r}" for r in range(self.dp_size)]
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"SiDP coordinated barrier rendezvous failed at member "
                f"{self.dp_rank}: {exc}"
            ) from exc
        if self.enable_debug_logging:
            logger.info(
                f"[SiDP rank{self.dp_rank}] coordinated device barrier ready "
                f"(K={self.barrier_interval_cycles}, nptr={resolved_nptr}, "
                f"ring_depth={self._nptr_ring_depth}, "
                f"barriers_per_forward={self._barriers_per_forward})"
            )

        # Dynamic nptr: build the per-forward host rendezvous over the same
        # TCPStore. The live participant count then comes from consensus each
        # forward instead of the fixed resolved_nptr above.
        if self.coord_dynamic_nptr:
            from sglang.srt.layers.sidp.coord_rendezvous import CoordRendezvous

            self._rendezvous = CoordRendezvous(
                store=self.store,
                member_rank=self.dp_rank,
                world_size=self.dp_size,
            )
            if self.enable_debug_logging:
                logger.info(
                    f"[SiDP rank{self.dp_rank}] dynamic nptr rendezvous enabled"
                )

        # Schedule-consistency observer (read-only). Built once here so the
        # scheduler hook can record a row per rendezvous round. Independent of
        # dynamic nptr: useful to observe divergence even in fixed-nptr runs.
        if self.coord_observe:
            from sglang.srt.layers.sidp.coord_observer import CoordScheduleObserver

            self._observer = CoordScheduleObserver(
                member_rank=self.dp_rank,
                world_size=self.dp_size,
                out_dir=self.coord_observe_dir,
            )
            if self.enable_debug_logging:
                logger.info(
                    f"[SiDP rank{self.dp_rank}] schedule observer -> "
                    f"{self._observer.path}"
                )

    @property
    def coord_rendezvous_enabled(self) -> bool:
        """Whether the scheduler should drive the per-forward nptr rendezvous."""
        return self.coord_mode and self.coord_dynamic_nptr and self._rendezvous is not None

    def coord_rendezvous_before_forward(self, wants_forward: bool) -> tuple[bool, int]:
        """Scheduler hook: vote decode/idle, return (participate, live_nptr).

        Called once per scheduler iteration (both event loops) after the batch is
        decided and before run_batch. When participate is True the caller runs the
        forward; live_nptr has already been written to this wave's device barrier
        ring slot.

        Wave alignment (why we advance on live>0, not on participation): the
        device barrier's ring slot is ``(bar[1] / bpf) % depth`` and ``bar[1]`` is
        the SHARED global completed-barrier count. It advances by ``bpf`` in every
        round some member decodes (live>0). Every member sees the SAME ``live``
        each round via the rendezvous, so every member advances an identical wave
        counter -- even in rounds it sits out (prefill). That keeps each member's
        host-side slot index equal to the global ``bar[1]/bpf`` the kernel reads,
        so a member that skips a decode round still writes the right slot when it
        rejoins. Only participating members actually write (they all write the
        same live into the same slot, so the value is consistent regardless).
        """
        participate, live = self._rendezvous.rendezvous(wants_forward)
        if live > 0:
            wave = self._barrier_round
            if participate:
                self.set_live_nptr(wave, live)
            self._barrier_round = wave + 1
        return participate, live

    @property
    def unified_enabled(self) -> bool:
        """Whether the scheduler should drive the unified-schedule rendezvous."""
        return (
            self.coord_mode
            and self.coord_unified_schedule
            and self._rendezvous is not None
        )

    def coord_unified_decide(
        self, *, want_prefill: bool, has_running: bool, chunked_must: bool,
        below_watermark: bool,
    ) -> str:
        """Scheduler hook (inside the prefill decision): agree on ONE group mode.

        Called once per iteration by every member from within the prefill decision
        (after the rank's own intent is known, before any KV is allocated). Runs
        the unified rendezvous and returns M in {"d","p","i"}.

        This decides INTENT ONLY -- it does NOT write the barrier nptr. The nptr
        must reflect how many members ACTUALLY run a decode forward this round,
        which is only known after the batch is finalized (a member with running
        work may still end up with an empty decode batch after filtering, and it
        would then launch no barrier). So the barrier live count is set later by
        the unchanged ``coord_rendezvous_before_forward`` at the event-loop point,
        which votes on the real ``batch.forward_mode.is_decode()``. Decoupling the
        two (intent here, nptr there) keeps live == actual barrier arrivals and
        avoids the count/arrival mismatch that deadlocks the device barrier.

        Cost: two rendezvous rounds per iteration (mode, then nptr), each one
        set + one wait -- negligible next to a forward.
        """
        m = self._rendezvous.rendezvous_unified(
            want_prefill=want_prefill,
            has_running=has_running,
            chunked_must=chunked_must,
            below_watermark=below_watermark,
        )
        self._last_unified_mode = m
        return m

    @property
    def last_unified_mode(self):
        """The most recent unified group mode ("d"/"p"/"i"), or None."""
        return self._last_unified_mode

    @property
    def observe_enabled(self) -> bool:
        """Whether the scheduler should feed per-round observation rows."""
        return self._observer is not None

    def observe_round(
        self,
        *,
        mode: str,
        running: int,
        waiting: int,
        batch_size: int,
        kv_used_frac: float,
        retracted: int,
        eos_prev: int,
        participate: bool,
        live_nptr: int,
    ) -> None:
        """Record one round's scheduling decision (read-only diagnostics).

        Keyed by the rendezvous round id, which is aligned across all members, so
        the offline tool can compare member decisions within the same wave. Called
        by the scheduler right after coord_rendezvous_before_forward, so the round
        id is the one that rendezvous just used. No-op unless observing.
        """
        if self._observer is None:
            return
        # last_round is the aligned wave id; -1 before the first rendezvous (e.g.
        # a pure-observe run with dynamic nptr off) falls back to a local counter.
        round_id = getattr(self._rendezvous, "last_round", -1) if self._rendezvous else -1
        if round_id < 0:
            round_id = getattr(self, "_observe_local_round", 0)
            self._observe_local_round = round_id + 1
        self._observer.record(
            round_id=round_id,
            mode=mode,
            running=running,
            waiting=waiting,
            batch_size=batch_size,
            kv_used_frac=kv_used_frac,
            retracted=retracted,
            eos_prev=eos_prev,
            participate=participate,
            live_nptr=live_nptr,
        )

    def close_observer(self) -> None:
        """Flush and close the schedule observer, if any."""
        if self._observer is not None:
            self._observer.close()
            self._observer = None

    def _count_barriers_per_forward(self) -> int:
        """Barriers a single decode forward launches (bpf, a setup constant).

        _maybe_launch_barrier fires once per cycle c in [1, num_cycles) with
        c % K == 0, so bpf = floor((num_cycles - 1) / K). The device kernel uses
        this to map its shared generation bar[1] back to a per-forward wave index
        (bar[1] / bpf). Clamped to >=1 to keep the kernel's division well-defined
        even in degenerate configs where no barrier ever launches (kernel unused).
        """
        k = max(self.barrier_interval_cycles, 1)
        bpf = max(self._num_cycles - 1, 0) // k
        return max(bpf, 1)

    def _maybe_launch_barrier(self, cycle: int):
        """Enqueue one device barrier on comm_stream at a K-cycle boundary.

        Called from _enqueue_cycle so the barrier is a normal kernel node inside
        the captured comm-stream graph. Every member issues the barrier for the
        same set of cycles (schedule is deterministic across members), so arrive
        counts match. Cycle 0 is resident/refilled outside the in-forward barrier
        cadence and is intentionally not gated here.
        """
        if not self.coord_mode or self._device_barrier is None:
            return
        if self.coord_disable_device_barrier:
            # Ablation: host rendezvous + peak-shifting stay on, but no device
            # barrier is enqueued (measure whether host-only sync suffices).
            return
        if not self._barrier_active_this_forward:
            return
        if cycle <= 0 or cycle % self.barrier_interval_cycles != 0:
            return
        self._device_barrier.launch(
            self._bar_ptr,
            self._nptr_ptr,
            self._nptr_ring_depth,
            self._barriers_per_forward,
            self.comm_stream.cuda_stream,
        )

    def set_live_nptr(self, wave: int, n: int) -> None:
        """Write this wave's participant count into its device barrier ring slot.

        Called by the scheduler (dynamic nptr mode) after the forward-boundary
        rendezvous, before the forward runs. The barrier kernel reads the ring at
        runtime (index ``(bar[1] / bpf) % depth``), so writing the device buffer
        here takes effect for both eager and CUDA Graph replay (the graph captured
        the pointer, not the value). The write goes to slot ``wave % depth`` -- a
        different slot than the previous/next wave -- so a later wave's write can
        never clobber the count an in-flight barrier from this wave still reads.
        The write is enqueued on the compute stream so it is ordered before this
        forward's captured barrier kernels execute.
        """
        if not self.coord_mode or self._nptr is None:
            return
        if self.coord_disable_device_barrier:
            # No device barrier is launched, so nothing ever reads _nptr -- writing
            # it would be a dead H2D. Skip it (this ablation is host-sync only).
            return
        if n < 1:
            raise ValueError(f"SiDP live nptr must be >= 1, got {n}")
        slot = wave % self._nptr_ring_depth
        self._nptr[slot].fill_(n)

    def wait_prefetch(self, layer_id: int):
        """Called BEFORE the MLP GEMM of a non-local layer.

        Cycle 0 is resident before the forward begins, so it has no in-forward
        RAW edge. Later cycles wait only for their own compressed slot's
        prefetch event. Decode is then enqueued on the compute stream directly
        before the GEMM. The serial fallback issues this layer's DMA using
        fork/copy/join and uses the same decode hook.
        """
        slot = self._layer_to_slot[layer_id]
        compute_stream = torch.cuda.current_stream()

        if self.enable_cycle_overlap and self.k < self.dp_size:
            if layer_id // self.dp_size == 0:
                # The previous forward's tail (or setup for the first forward)
                # established the cycle-0-resident invariant.
                self._decode_weight_before_compute(layer_id, slot, compute_stream)
                return
            # RAW: this layer alone waits for its copy. Other cycle copies stay
            # in flight on comm_stream while earlier layers compute.
            if self._graph_profiler is not None:
                self._graph_profiler.record_wait_start(layer_id, compute_stream)
            if self._uses_flag_sync:
                self._cycle_backend.wait_layer(layer_id)
            else:
                compute_stream.wait_event(self._prefetch_events[slot])
            if self._graph_profiler is not None:
                self._graph_profiler.record_wait_end(layer_id, compute_stream)
            self._decode_weight_before_compute(layer_id, slot, compute_stream)
            return

        # fork: comm_stream starts after compute_stream's current point
        self.comm_stream.wait_stream(compute_stream)

        # DMA encoded main/extra tensors into this slot on comm_stream.
        for pname, peer_weight in self.peer_views[layer_id].items():
            self._copy_encoded_weight(layer_id, pname, peer_weight, slot)
        self._prefetch_events[slot].record(self.comm_stream)

        # join: compute waits for the prefetch to finish (RAW)
        compute_stream.wait_event(self._prefetch_events[slot])
        self._decode_weight_before_compute(layer_id, slot, compute_stream)

    def record_compute_and_prefetch_next(self, layer_id: int):
        """Record buffer consumption and advance the cycle window."""
        if not self.enable_cycle_overlap:
            return

        slot = self._layer_to_slot[layer_id]
        if self._uses_flag_sync:
            self._cycle_backend.record_consumed(layer_id)
        else:
            self._consume_events[slot].record(torch.cuda.current_stream())

        cycle = layer_id // self.dp_size
        if self._last_non_local_in_cycle.get(cycle) == layer_id:
            next_cycle = cycle + self._cycle_cache_depth
            if next_cycle < self._num_cycles:
                # With depth=2, compute(c) releases the slot used to prefetch
                # c+2 while c+1 is resident.
                self._enqueue_cycle(next_cycle)
            elif cycle == self._num_cycles - self._cycle_cache_depth:
                # Gemma4 has six cycles. Once c4 releases slot group 0, refill
                # it with the next forward's c0 while c5 computes.
                self._enqueue_next_forward_cycle_zero()

    def record_dummy_consume_and_prefetch_next(self, layer_id: int):
        """Advance the copy pipeline without reading weights in diagnostic mode.

        Dummy-compute profiling intentionally omits the RAW dependency because
        no MLP reads the rolling buffer.  Record an empty wait interval so the
        normal profiler schema remains complete, then release the slot at the
        current compute-stream point and enqueue the next cycle normally.
        """
        compute_stream = torch.cuda.current_stream()
        if self._graph_profiler is not None and layer_id // self.dp_size > 0:
            self._graph_profiler.record_wait_start(layer_id, compute_stream)
            self._graph_profiler.record_wait_end(layer_id, compute_stream)
        self.record_compute_and_prefetch_next(layer_id)

    def begin_forward(self, is_decode: bool = True):
        """Start one eager or captured forward with cycle 0 already resident.

        The graph-start fork orders cycle 1 after the previous forward. No
        cross-forward event is needed because forwards/graph launches are
        serialized on the model stream and the previous forward ended with a
        comm-stream join.

        ``is_decode`` gates the coordinated device barrier: the basic Phase 2
        version only aligns decode forwards. Prefill (extend) batch shapes and
        per-member arrival differ, and without a per-forward rendezvous a
        fixed-nptr barrier there would hang the group. Prefill runs the normal
        unbarriered copy path.
        """
        if not self.enable_cycle_overlap:
            return

        self._barrier_active_this_forward = self.coord_mode and is_decode
        compute_stream = torch.cuda.current_stream()
        if self._graph_profiler is not None:
            self._graph_profiler.record_forward_start(compute_stream)
        self._queued_cycles.clear()
        self._queued_cycles.add(0)  # resident from setup or previous forward
        self._next_forward_cycle_zero_queued = False
        if self._uses_flag_sync:
            # The previous final join makes it safe to rebase bookkeeping while
            # preserving the already resident next-forward cycle 0 data.
            self._cycle_backend.reset_forward(cycle_zero_resident=True)
        self.comm_stream.wait_stream(compute_stream)
        for cycle in range(1, min(self._cycle_cache_depth, self._num_cycles)):
            self._enqueue_cycle(cycle, wait_for_consume=False)

    def end_forward(self):
        """Join the tail prefetch so the next forward starts with c0 resident."""
        if self.enable_cycle_overlap:
            compute_stream = torch.cuda.current_stream()
            if self._graph_profiler is not None:
                self._graph_profiler.record_forward_compute_end(compute_stream)
            compute_stream.wait_stream(self.comm_stream)
            if self._graph_profiler is not None:
                self._graph_profiler.record_forward_end(compute_stream)

    def record_cycle_compute_start(self, layer_id: int):
        """Mark the start of a full decoder cycle for diagnostic captures."""
        if self._graph_profiler is None or layer_id % self.dp_size != 0:
            return
        self._graph_profiler.record_cycle_compute_start(
            layer_id // self.dp_size, torch.cuda.current_stream()
        )

    def record_cycle_compute_end(self, layer_id: int):
        """Mark the end of a full decoder cycle for diagnostic captures."""
        is_cycle_end = (
            layer_id % self.dp_size == self.dp_size - 1
            or layer_id == self.num_layers - 1
        )
        if self._graph_profiler is None or not is_cycle_end:
            return
        self._graph_profiler.record_cycle_compute_end(
            layer_id // self.dp_size, torch.cuda.current_stream()
        )

    def profile_after_cuda_graph_replay(
        self,
        *,
        raw_batch_size: int,
        graph_batch_size: int,
        launch_profile: dict | None = None,
    ):
        """Collect one sampled decode replay when profiling is enabled."""
        if self._graph_profiler is not None:
            self._graph_profiler.collect_after_graph_replay(
                raw_batch_size=raw_batch_size,
                graph_batch_size=graph_batch_size,
                launch_profile=launch_profile,
                trace_provider=(
                    self._cycle_backend.trace_snapshot
                    if self._cycle_backend is not None
                    else None
                ),
            )

    @property
    def graph_profiling_enabled(self) -> bool:
        return self._graph_profiler is not None

    @property
    def needs_cuda_graph_launch_hook(self) -> bool:
        """Whether replay needs functional synchronization or timing metadata."""
        return self.graph_profiling_enabled or self._launch_sync_strategy.name != "none"

    def before_cuda_graph_replay(
        self, *, raw_batch_size: int, graph_batch_size: int
    ) -> dict:
        """Apply the configured peak synchronization strategy before replay."""
        return self._launch_sync_strategy.before_launch(
            raw_batch_size=raw_batch_size,
            graph_batch_size=graph_batch_size,
        )

    def prefetch_first_layers(self):
        """Backward-compatible alias; forward-boundary hooks call begin_forward."""
        self.begin_forward()

    def get_weight_buffer(self, layer_id: int, param_name: str) -> torch.Tensor:
        """Return the local rolling buffer holding the prefetched weight for this layer."""
        slot = self._layer_to_slot[layer_id]
        return self.buffers[slot][param_name]

    def get_encoded_weight(self, layer_id: int, param_name: str) -> EncodedWeight:
        """Return persistent-local or RAW-ready remote encoded compute inputs.

        This is the future FUSED operator adapter boundary: the consumer gets
        the main encoded tensor and every named extra without requiring an HBM
        materialization buffer. Remote callers must first execute the normal
        ``wait_prefetch`` RAW edge.
        """
        if layer_id in self.peer_views:
            slot = self._layer_to_slot[layer_id]
            return self._transfer_buffers[slot][param_name]
        return self._local_encoded_weights[layer_id][param_name]

    def get_materialization_buffer(self, param_name: str) -> torch.Tensor | None:
        """Return the optional rank-shared HBM decode output requested by codec."""
        return self._materialization_buffers.get(param_name)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _validate_encoded_weight(
        self, encoded: EncodedWeight, *, context: str
    ) -> None:
        """Validate the graph/IPC invariants of main and auxiliary tensors."""
        components = [("<main>", encoded.tensor), *encoded.extra_tensors.items()]
        for name, tensor in components:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"SiDP encoded buffer names must be non-empty strings: {context}"
                )
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(
                    "SiDP encoded buffers must be contiguous CUDA tensors: "
                    f"{context}, component={name}, codec={self.weight_codec.name}"
                )

    def _validate_cycle_buffer(
        self,
        receive_buffer: EncodedWeight,
        encoded_weights: List[EncodedWeight],
        *,
        context: str,
    ) -> None:
        """Check one slot can receive every encoded layer assigned to it."""
        if not encoded_weights:
            raise ValueError(f"SiDP cycle slot has no source layers: {context}")
        self._validate_encoded_weight(receive_buffer, context=context)

        extra_schema = set(encoded_weights[0].extra_tensors)
        if set(receive_buffer.extra_tensors) != extra_schema:
            raise ValueError(
                "SiDP receive extra-buffer schema mismatch: "
                f"{context}, receive={sorted(receive_buffer.extra_tensors)}, "
                f"source={sorted(extra_schema)}"
            )
        for encoded in encoded_weights:
            if set(encoded.extra_tensors) != extra_schema:
                raise ValueError(
                    "SiDP encoded extra-buffer schema must be graph-stable for "
                    f"layers sharing a slot: {context}"
                )
            if receive_buffer.tensor.nbytes < encoded.tensor.nbytes:
                raise ValueError(
                    "SiDP main receive buffer is smaller than its source: "
                    f"{context}, receive={receive_buffer.tensor.nbytes}, "
                    f"required={encoded.tensor.nbytes}"
                )
            for name, source in encoded.extra_tensors.items():
                receive = receive_buffer.extra_tensors[name]
                if receive.nbytes < source.nbytes:
                    raise ValueError(
                        "SiDP extra receive buffer is smaller than its source: "
                        f"{context}, extra={name}, receive={receive.nbytes}, "
                        f"required={source.nbytes}"
                    )

    def _copy_encoded_weight(
        self,
        layer_id: int,
        param_name: str,
        peer_weight: EncodedWeight,
        slot: int,
    ) -> None:
        """Pull one encoded main tensor and all extras into a cycle slot.

        The caller records the prefetch RAW event only after this method has
        enqueued every DMA. Decode/compute therefore observes the main tensor,
        scale/zero-point/codebook/etc. as one atomic encoded weight.
        """
        receive_buffer = self._transfer_buffers[slot][param_name]
        self.memcpy.async_copy(
            receive_buffer.tensor.data_ptr(),
            peer_weight.tensor.data_ptr(),
            peer_weight.tensor.nbytes,
            self.comm_stream.cuda_stream,
        )
        self._pull_extra_buffers(peer_weight, receive_buffer)

    def _pull_extra_buffers(
        self,
        peer_weight: EncodedWeight,
        receive_buffer: EncodedWeight,
    ) -> None:
        """Enqueue codec-defined auxiliary tensor pulls on ``comm_stream``."""
        if set(peer_weight.extra_tensors) != set(receive_buffer.extra_tensors):
            raise RuntimeError(
                "SiDP encoded extra-buffer schema changed after setup: "
                f"peer={sorted(peer_weight.extra_tensors)}, "
                f"slot={sorted(receive_buffer.extra_tensors)}"
            )
        for name, peer_extra in peer_weight.extra_tensors.items():
            receive_extra = receive_buffer.extra_tensors[name]
            self.memcpy.async_copy(
                receive_extra.data_ptr(),
                peer_extra.data_ptr(),
                peer_extra.nbytes,
                self.comm_stream.cuda_stream,
            )

    def _decode_weight_before_compute(
        self,
        layer_id: int,
        slot: int,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        """Materialize one RAW-ready encoded layer immediately before GEMM.

        Identity decode is a no-op and ``self.buffers[slot]`` aliases the
        encoded cycle slot. A future MATERIALIZE codec passes one rank-global
        layer-sized HBM buffer; a FUSED codec passes no materialization buffer
        and its custom GEMM consumes the encoded main/extras directly.

        TODO(SiDP codec): MATERIALIZE mode may record compressed-slot consume
        after decode and refill it during GEMM. DIRECT/FUSED modes must retain
        the post-GEMM WAR edge because GEMM itself consumes the cycle slot.
        """
        for param_name, peer_weight in self.peer_views[layer_id].items():
            compute_buffer = self.buffers.get(slot, {}).get(param_name)
            self.weight_codec.decode_before_compute(
                layer_id=layer_id,
                param_name=param_name,
                encoded=self._transfer_buffers[slot][param_name],
                compute_buffer=compute_buffer,
                metadata=peer_weight.metadata,
                stream=compute_stream,
            )

    def _do_prefetch(self, layer_id: int, wait_for_consume: bool = True):
        """Issue one layer's encoded transport copy."""
        slot = self._layer_to_slot[layer_id]

        if wait_for_consume:
            # WAR: wait for previous compute that used this slot to finish.
            self.comm_stream.wait_event(self._consume_events[slot])

        if self.copy_backend == SidpCopyBackend.SM.value:
            if not self.sm_use_event_sync or self._cycle_backend is None:
                raise RuntimeError(
                    "fixed SM Event prefetch requires sm_use_event_sync"
                )
            self._cycle_backend.enqueue_fixed_layer_event(layer_id)
            self._prefetch_events[slot].record(self.comm_stream)
            return

        peer_params = self.peer_views[layer_id]
        if self._graph_profiler is not None:
            self._graph_profiler.record_copy_start(
                layer_id,
                sum(peer_weight.nbytes for peer_weight in peer_params.values()),
                self.comm_stream,
            )
        for pname, peer_weight in peer_params.items():
            self._copy_encoded_weight(layer_id, pname, peer_weight, slot)

        # RAW: the encoded main tensor and every extra are now resident.
        # Decode/compute preparation happens immediately before the layer GEMM.
        if self._graph_profiler is not None:
            self._graph_profiler.record_copy_end(layer_id, self.comm_stream)
        self._prefetch_events[slot].record(self.comm_stream)

    def _enqueue_cycle(self, cycle: int, wait_for_consume: bool = True):
        """Enqueue one cycle exactly once in the current forward."""
        if cycle in self._queued_cycles:
            return
        layers = self._cycle_layers.get(cycle)
        if not layers:
            return
        self._queued_cycles.add(cycle)
        # Direction A: align all members before this cycle's copies begin, every
        # K cycles. Barrier-then-copy (matches the validated probe) puts the
        # sense-reversing kernel node ahead of the cycle's DMAs on comm_stream.
        self._maybe_launch_barrier(cycle)
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_start(cycle, self.comm_stream)
        if self._uses_flag_sync:
            if isinstance(self._cycle_backend, SidpFixedDmaFlagBackend):
                self._cycle_backend.enqueue_cycle(
                    cycle, wait_for_consume=wait_for_consume
                )
            else:
                self._cycle_backend.enqueue_cycle(cycle)
        elif self.dma_slices > 1:
            self._cycle_backend.enqueue_cycle(
                cycle, wait_for_consume=wait_for_consume
            )
        else:
            for layer_id in layers:
                self._do_prefetch(layer_id, wait_for_consume=wait_for_consume)
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_end(cycle, self.comm_stream)

    def _enqueue_next_forward_cycle_zero(self):
        """Refill slot group 0 for the next forward while the tail computes."""
        if self._next_forward_cycle_zero_queued:
            return
        self._next_forward_cycle_zero_queued = True
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_start(0, self.comm_stream)
        if self._uses_flag_sync:
            required_comp_gen, target_fill_gen = next_forward_cycle_zero_generations(
                self._num_cycles, self._cycle_cache_depth
            )
            self._cycle_backend.enqueue_cycle(
                0,
                target_fill_gen=target_fill_gen,
                required_comp_gen=required_comp_gen,
            )
        elif self.dma_slices > 1:
            self._cycle_backend.enqueue_cycle(0, wait_for_consume=True)
        else:
            for layer_id in self._cycle_layers[0]:
                self._do_prefetch(layer_id)
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_end(0, self.comm_stream)

    def _initialize_cycle_zero(self):
        """Materialize the first forward's cycle 0 during model initialization."""
        compute_stream = torch.cuda.current_stream()
        if self._uses_flag_sync:
            self._cycle_backend.reset_forward(cycle_zero_resident=False)
        self.comm_stream.wait_stream(compute_stream)
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_start(0, self.comm_stream)
        if self._uses_flag_sync:
            self._cycle_backend.enqueue_cycle(
                0,
                target_fill_gen=1,
                required_comp_gen=0,
            )
        elif self.dma_slices > 1:
            self._cycle_backend.enqueue_cycle(0, wait_for_consume=False)
        else:
            for layer_id in self._cycle_layers[0]:
                self._do_prefetch(layer_id, wait_for_consume=False)
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_end(0, self.comm_stream)
        compute_stream.wait_stream(self.comm_stream)
        torch.cuda.synchronize()
        if self._cycle_backend is not None and self.enable_debug_logging:
            self._cycle_backend.debug_validate_cycle(0)

    def _build_cycle_schedule(self):
        """Build compute-order cycle membership and stable slot identities."""
        self._num_cycles = (self.num_layers + self.dp_size - 1) // self.dp_size
        self._cycle_cache_depth = min(self.cache_cycles, self._num_cycles)
        if self.enable_cycle_overlap and self.k < self.dp_size:
            if self._cycle_cache_depth != 2:
                raise NotImplementedError(
                    "SiDP cross-forward cycle overlap currently requires "
                    "cache_cycles=2"
                )
            if self._num_cycles % 2 != 0:
                # TODO(SiDP): for an odd cycle count, slot group 0 is still
                # consumed by the final cycle. The future fallback should refill
                # next-forward c0 position-by-position as that final cycle
                # computes. This intentionally uses compute order (and may
                # incast) for only the tail cycle. Current Gemma4 has 6 cycles,
                # so leave this branch explicit but unimplemented for now.
                raise NotImplementedError(
                    "SiDP cross-forward cycle overlap currently requires an even "
                    "number of cycles; odd-cycle tail refill is reserved"
                )
        # Slot identity always follows compute order. The fetch policy may be
        # peak-shifted independently without changing buffer ownership.
        self._remote_positions = remote_positions(
            self.dp_rank, self.dp_size, self.k, peak_shifting=False
        )
        self._remote_position_to_index = {
            pos: index for index, pos in enumerate(self._remote_positions)
        }
        self._fetch_schedule = prefetch_order(
            self.dp_rank,
            self.dp_size,
            self.k,
            self.num_layers,
            peak_shifting=(
                self.prefetch_policy == SidpPrefetchPolicy.STATIC_PEAK.value
            ),
        )
        self._cycle_layers = {}
        for layer_id in self._fetch_schedule:
            cycle = layer_id // self.dp_size
            self._cycle_layers.setdefault(cycle, []).append(layer_id)
        self._last_non_local_in_cycle = {
            cycle: max(layer_ids) for cycle, layer_ids in self._cycle_layers.items()
        }

    def _alloc_buffers(self, layers, non_local_layers):
        """Allocate identity slots and codec cycle-cache staging.

        Today only identity is registered, so every transfer slot aliases its
        compute-dtype model buffer and no extra memory is introduced. A real
        codec keeps only two cycles of encoded main/extra buffers here. It may
        request one rank-global materialization-buffer set outside this loop,
        or request none when a fused operator consumes the encoding directly.
        """
        if self.enable_cycle_overlap:
            num_slots = self._cycle_cache_depth * len(self._remote_positions)
        else:
            num_slots = min(self.cache_cycles, len(non_local_layers))
        if num_slots == 0:
            return

        if not self._uses_flag_sync:
            self._prefetch_events = [torch.cuda.Event() for _ in range(num_slots)]
            self._consume_events = [torch.cuda.Event() for _ in range(num_slots)]

        if self.enable_cycle_overlap:
            remote_count = len(self._remote_positions)
            for lid in non_local_layers:
                cycle = lid // self.dp_size
                position = lid % self.dp_size
                cycle_slot = cycle % self._cycle_cache_depth
                position_slot = self._remote_position_to_index[position]
                self._layer_to_slot[lid] = cycle_slot * remote_count + position_slot
        else:
            # Graph-safe serial fallback: layer slots are reused round-robin.
            for i, lid in enumerate(non_local_layers):
                self._layer_to_slot[lid] = i % num_slots

        # Get shapes from the first non-local layer (weights still intact at this point)
        ref_layer = layers[non_local_layers[0]]
        param_shapes = {}
        for pname, param in self._get_ffn_params(ref_layer):
            param_shapes[pname] = (param.shape, param.dtype)

        # Ask the codec whether decompression needs an HBM output at all. A
        # MATERIALIZE codec gets exactly one rank-shared set, never one per
        # cycle slot. DIRECT/FUSED must return None.
        device = torch.cuda.current_device()
        for pname, (shape, dtype) in param_shapes.items():
            spec = self.weight_codec.materialization_spec(
                param_name=pname,
                original_shape=shape,
                original_dtype=dtype,
            )
            if self.weight_codec.compute_mode == WeightComputeMode.MATERIALIZE:
                if spec is None:
                    raise ValueError(
                        "SiDP MATERIALIZE codec must provide a buffer spec: "
                        f"param={pname}, codec={self.weight_codec.name}"
                    )
                self._materialization_buffers[pname] = torch.empty(
                    spec.shape, dtype=spec.dtype, device=device
                )
            elif spec is not None:
                raise ValueError(
                    "SiDP DIRECT/FUSED codec must not request an HBM "
                    f"materialization buffer: param={pname}, "
                    f"codec={self.weight_codec.name}"
                )

        # Allocate the two-cycle encoded slots. DIRECT mode additionally needs
        # one compute-compatible tensor per slot because GEMM reads it in place.
        for s in range(num_slots):
            self.buffers[s] = {}
            self._transfer_buffers[s] = {}
            for pname, (shape, dtype) in param_shapes.items():
                direct_compute_buffer = None
                if self.weight_codec.compute_mode == WeightComputeMode.DIRECT:
                    direct_compute_buffer = torch.empty(
                        shape, dtype=dtype, device=device
                    )
                    self.buffers[s][pname] = direct_compute_buffer
                elif self.weight_codec.compute_mode == WeightComputeMode.MATERIALIZE:
                    # Every slot intentionally aliases the same one-layer output.
                    self.buffers[s][pname] = self._materialization_buffers[pname]
                slot_layers = [
                    lid for lid in non_local_layers if self._layer_to_slot[lid] == s
                ]
                encoded_weights = [
                    self.peer_views[lid][pname] for lid in slot_layers
                ]
                receive_buffer = self.weight_codec.allocate_cycle_buffers(
                    param_name=pname,
                    encoded_weights=encoded_weights,
                    direct_compute_buffer=direct_compute_buffer,
                )
                self._validate_cycle_buffer(
                    receive_buffer,
                    encoded_weights,
                    context=f"slot={s}, param={pname}",
                )
                self._transfer_buffers[s][pname] = receive_buffer

    def _prime_routes(self, non_local_layers):
        """D6: One real copy per peer device to build peer page mapping."""
        if not non_local_layers:
            return
        device = torch.cuda.current_device()
        primed_devices = set()
        for lid in non_local_layers:
            src_dev = owner_of(lid, self.dp_size)
            if src_dev in primed_devices:
                continue
            # One small copy from this peer's view to trigger page mapping
            for pname, encoded in self.peer_views[lid].items():
                pv = encoded.tensor
                tmp = torch.empty(min(1024, pv.numel()), dtype=pv.dtype, device=device)
                tmp.copy_(pv.flatten()[: tmp.numel()])
                del tmp
                break  # one param per device is enough
            primed_devices.add(src_dev)
        torch.cuda.synchronize()

    def _collect_decoder_layers(self, model) -> Dict[int, Any]:
        """Find all decoder layers that have .mlp and .layer_id."""
        layers = {}
        for _, module in model.named_modules():
            if hasattr(module, "layer_id") and hasattr(module, "mlp"):
                layers[module.layer_id] = module
        return layers

    def _get_ffn_params(self, layer) -> List[Tuple[str, torch.nn.Parameter]]:
        """Return the FFN weight parameters for a decoder layer."""
        result = []
        if hasattr(layer, "mlp"):
            for name, param in layer.mlp.named_parameters():
                if "weight" in name:
                    result.append((name, param))
        return result
