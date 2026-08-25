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

from sglang.srt.layers.sidp.config import SidpConfig
from sglang.srt.layers.sidp.cuda_memcpy import SidpCudaMemcpy
from sglang.srt.layers.sidp.graph_profiler import SidpGraphProfiler
from sglang.srt.layers.sidp.scheduler import (
    is_local_layer,
    owner_of,
    prefetch_order,
    remote_positions,
)
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
        self.k = config.k
        self.cache_cycles = config.cache_cycles
        self.num_layers = config.num_layers
        self.enable_cycle_overlap = config.enable_cycle_overlap
        self.enable_peak_shifting = config.enable_peak_shifting
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
        self._launch_sync_strategy = NoSyncStrategy()

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
            order = "peak-shifting" if self.enable_peak_shifting else "compute"
            logger.info(
                f"[SiDP rank{self.dp_rank}] local={len(local_layers)}, "
                f"non_local={len(non_local_layers)}, "
                f"mode={mode}, order={order}"
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
                handle = pickle.dumps(
                    {
                        "version": 2,
                        "codec": self.weight_codec.name,
                        "tensor": _reduce_tensor(encoded.tensor),
                        "extra_tensors": {
                            name: _reduce_tensor(tensor)
                            for name, tensor in encoded.extra_tensors.items()
                        },
                        "metadata": encoded.metadata,
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
            for pname, _ in self._get_ffn_params(layers[lid]):
                key = f"sidp/{src}/{lid}/{pname}"
                payload = self.store.get(key)
                wire_payload = pickle.loads(payload)
                # Accept the pre-codec identity payload for easier rolling
                # upgrades, while all new ranks publish the versioned format.
                if isinstance(wire_payload, dict):
                    version = wire_payload.get("version")
                    if version not in (1, 2):
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
                        if version == 2
                        else {}
                    )
                    metadata = wire_payload.get("metadata")
                else:
                    if self.weight_codec.name != "identity":
                        raise RuntimeError(
                            "Legacy SiDP IPC payload is only valid for identity codec"
                        )
                    reduced_tensor = wire_payload
                    reduced_extras = {}
                    metadata = None
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

        # Initial WAR state: every slot is safe to write before its first use.
        for evt in self._consume_events:
            evt.record(torch.cuda.current_stream())

        if self.enable_cycle_overlap and self._cycle_layers.get(0):
            self._initialize_cycle_zero()

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

    def begin_forward(self):
        """Start one eager or captured forward with cycle 0 already resident.

        The graph-start fork orders cycle 1 after the previous forward. No
        cross-forward event is needed because forwards/graph launches are
        serialized on the model stream and the previous forward ended with a
        comm-stream join.
        """
        if not self.enable_cycle_overlap:
            return

        compute_stream = torch.cuda.current_stream()
        if self._graph_profiler is not None:
            self._graph_profiler.record_forward_start(compute_stream)
        self._queued_cycles.clear()
        self._queued_cycles.add(0)  # resident from setup or previous forward
        self._next_forward_cycle_zero_queued = False
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
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_start(cycle, self.comm_stream)
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
        for layer_id in self._cycle_layers[0]:
            self._do_prefetch(layer_id)
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_end(0, self.comm_stream)

    def _initialize_cycle_zero(self):
        """Materialize the first forward's cycle 0 during model initialization."""
        compute_stream = torch.cuda.current_stream()
        self.comm_stream.wait_stream(compute_stream)
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_start(0, self.comm_stream)
        for layer_id in self._cycle_layers[0]:
            self._do_prefetch(layer_id, wait_for_consume=False)
        if self._graph_profiler is not None:
            self._graph_profiler.record_cycle_comm_end(0, self.comm_stream)
        compute_stream.wait_stream(self.comm_stream)
        torch.cuda.synchronize()

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
            peak_shifting=self.enable_peak_shifting,
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
