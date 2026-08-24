"""SiDP manager for IPC weight sharing and bounded prefetch buffers.

Eager mode implements cycle-level D5 WAR/RAW overlap and optional D4
peak-shifting. CUDA Graph mode intentionally retains the graph-safe serial
prefetch fallback until the same cycle DAG is captured in a later phase.
"""

import logging
import pickle
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed

from sglang.srt.layers.sidp.config import SidpConfig
from sglang.srt.layers.sidp.cuda_memcpy import SidpCudaMemcpy
from sglang.srt.layers.sidp.scheduler import (
    is_local_layer,
    owner_of,
    prefetch_order,
    remote_positions,
)

logger = logging.getLogger(__name__)


def _reduce_tensor(t: torch.Tensor):
    """Serialize a tensor into an IPC-safe (fn, args) tuple via torch's reduce_tensor."""
    from torch.multiprocessing.reductions import reduce_tensor

    return reduce_tensor(t)


def _rebuild_tensor(payload: bytes, src_device: int) -> torch.Tensor:
    """Rebuild a tensor from pickled (fn, args) on the source device context."""
    fn, args = pickle.loads(payload)
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
        self.enable_eager_overlap = config.enable_eager_overlap
        self.enable_peak_shifting = config.enable_peak_shifting

        # D2: TCPStore is created lazily in setup() so that all ranks
        # have finished load_model() before any rank tries to connect.
        self.store = None
        self._rdzv_host = config.rdzv_host
        self._rdzv_port = config.rdzv_port

        # D7: DMA engine wrapper
        self.memcpy = SidpCudaMemcpy()

        # D7/D8: eager uses this as the asynchronous cycle-fill stream;
        # graph-safe fallback forks/joins it inside each layer's capture scope.
        self.comm_stream = torch.cuda.Stream()

        # Per-slot RAW/WAR events. Eager uses both; graph-safe serial fallback
        # uses RAW only and never overwrites a slot before its immediate GEMM.
        self._prefetch_events: List[torch.cuda.Event] = []
        self._consume_events: List[torch.cuda.Event] = []

        # Populated by setup()
        self.peer_views: Dict[int, Dict[str, torch.Tensor]] = {}
        self.buffers: Dict[int, Dict[str, torch.Tensor]] = {}
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
        self._layers_ref: Dict[int, Any] = {}
        self._ipc_refs: List[torch.Tensor] = []

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
        logger.info(f"[SiDP rank{self.dp_rank}] TCPStore connected")

        layers = self._collect_decoder_layers(model)
        self._layers_ref = layers

        if not layers:
            logger.info(
                f"[SiDP rank{self.dp_rank}] no decoder layers found, skipping setup"
            )
            return

        self.num_layers = max(layers.keys()) + 1
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
        logger.info(
            f"[SiDP rank{self.dp_rank}] local={len(local_layers)}, "
            f"non_local={len(non_local_layers)}, "
            f"mode={'eager-cycle-overlap' if self.enable_eager_overlap else 'serial-graph-safe'}, "
            f"order={'peak-shifting' if self.enable_peak_shifting else 'compute'}"
        )

        # D3: Export owner layers' IPC handles into store
        torch.cuda.synchronize()  # D11: ensure weights visible to IPC
        logger.info(f"[SiDP rank{self.dp_rank}] publishing IPC handles...")
        for lid in local_layers:
            if owner_of(lid, self.dp_size) == self.dp_rank:
                layer = layers[lid]
                for pname, param in self._get_ffn_params(layer):
                    handle = pickle.dumps(_reduce_tensor(param.data))
                    self.store.set(f"sidp/{self.dp_rank}/{lid}/{pname}", handle)
        logger.info(
            f"[SiDP rank{self.dp_rank}] published handles for "
            f"{sum(1 for l in local_layers if owner_of(l, self.dp_size) == self.dp_rank)} layers"
        )

        # D3: Rebuild peer views for non-local layers
        logger.info(f"[SiDP rank{self.dp_rank}] fetching peer handles...")
        for lid in non_local_layers:
            src = owner_of(lid, self.dp_size)
            self.peer_views[lid] = {}
            for pname, _ in self._get_ffn_params(layers[lid]):
                key = f"sidp/{src}/{lid}/{pname}"
                payload = self.store.get(key)
                peer_view = _rebuild_tensor(payload, src_device=src)
                self.peer_views[lid][pname] = peer_view
                self._ipc_refs.append(peer_view)  # D11: prevent GC
        logger.info(
            f"[SiDP rank{self.dp_rank}] rebuilt {len(non_local_layers)} peer views"
        )

        # Allocate buffers BEFORE releasing weights (their shapes are still
        # needed here). Eager overlap uses cycle_cache_depth * (D-k) slots;
        # graph-safe fallback keeps the original cache_cycles layer slots.
        self._alloc_buffers(layers, non_local_layers)
        logger.info(
            f"[SiDP rank{self.dp_rank}] allocated {len(self.buffers)} buffer slots"
        )

        # Key design: rebind weight.data to the rolling buffer, THEN release the
        # original full-weight storage. This way self.mlp(x) automatically reads
        # from the prefetch buffer, and the HBM held by the original non-local
        # weights is handed back to the caching allocator (the whole point of SiDP).
        # DIAG: device-level free BEFORE release (empty cache first so the
        # allocator's own free blocks don't mask the delta we care about).
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
                freed_bytes += orig.numel() * orig.element_size()
                # Point weight.data at the buffer (zero-copy rebind)
                param.data = self.buffers[slot][pname]
                # Release the original full weight's HBM back to the allocator.
                orig.untyped_storage().resize_(0)
                del orig
        # Return the freed blocks to the driver so the downstream KV-cache
        # profiling (mem_get_info) actually sees the reclaimed memory.
        torch.cuda.empty_cache()
        # DIAG: device-level free AFTER release. If delta ~= logical freed_bytes,
        # the HBM physically returned to the device pool (KV profiling should see
        # it). If delta ~= 0, the freed pages are pinned by IPC peer mappings /
        # per-rank CUDA contexts sharing the card.
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

        # Bind hook to non-local layers
        for lid in non_local_layers:
            layer = layers[lid]
            layer._sidp_bound = True
            layer._sidp_mgr = self

        # Local layers: no SiDP intervention needed
        for lid in local_layers:
            layer = layers[lid]
            layer._sidp_bound = False
            layer._sidp_mgr = None

        logger.info(f"[SiDP rank{self.dp_rank}] setup complete")

    def wait_prefetch(self, layer_id: int):
        """Called BEFORE the MLP GEMM of a non-local layer.

        Eager mode only waits on this slot's RAW event; the copy was enqueued
        ahead of time by the cycle pipeline. Graph-safe fallback issues this
        layer's DMA inside the current capture scope using fork/copy/join, so
        its dependency chain never crosses a graph boundary. The MLP then
        reads the fixed slot via the rebound weight.data.
        """
        slot = self._layer_to_slot[layer_id]
        compute_stream = torch.cuda.current_stream()

        if self.enable_eager_overlap:
            # RAW: this layer alone waits for its copy. Other cycle copies stay
            # in flight on comm_stream while earlier layers compute.
            compute_stream.wait_event(self._prefetch_events[slot])
            return

        # fork: comm_stream starts after compute_stream's current point
        self.comm_stream.wait_stream(compute_stream)

        # DMA copy peer weights into this slot's buffer on comm_stream
        for pname, peer_view in self.peer_views[layer_id].items():
            buf = self.buffers[slot][pname]
            self.memcpy.async_copy(
                buf.data_ptr(),
                peer_view.data_ptr(),
                peer_view.nbytes,
                self.comm_stream.cuda_stream,
            )
        self._prefetch_events[slot].record(self.comm_stream)

        # join: compute waits for the prefetch to finish (RAW)
        compute_stream.wait_event(self._prefetch_events[slot])

    def record_compute_and_prefetch_next(self, layer_id: int):
        """Record buffer consumption and advance the eager cycle window."""
        if not self.enable_eager_overlap:
            return

        slot = self._layer_to_slot[layer_id]
        self._consume_events[slot].record(torch.cuda.current_stream())

        cycle = layer_id // self.dp_size
        if self._last_non_local_in_cycle.get(cycle) == layer_id:
            # Reuse this cycle's slot only after all of its per-position
            # consume events have been recorded. With depth=2, compute(c)
            # releases the buffers used to prefetch c+2 while c+1 is resident.
            self._enqueue_cycle(cycle + self._cycle_cache_depth)

    def prefetch_first_layers(self):
        """Prime the eager cycle window before entering layer 0.

        Cycle 0 is demand data; later resident cycles are lookahead. Since the
        copies are queued on one comm stream, compute can begin as soon as its
        first remote layer is ready while the rest of the window keeps filling.
        """
        if not self.enable_eager_overlap:
            return

        self._queued_cycles.clear()
        for cycle in range(min(self._cycle_cache_depth, self._num_cycles)):
            self._enqueue_cycle(cycle)

    def get_weight_buffer(self, layer_id: int, param_name: str) -> torch.Tensor:
        """Return the local rolling buffer holding the prefetched weight for this layer."""
        slot = self._layer_to_slot[layer_id]
        return self.buffers[slot][param_name]

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _do_prefetch(self, layer_id: int):
        """Issue raw cudaMemcpyAsync on comm_stream for one layer's weights."""
        slot = self._layer_to_slot[layer_id]

        # WAR: wait for previous compute that used this slot to finish
        self.comm_stream.wait_event(self._consume_events[slot])

        for pname, peer_view in self.peer_views[layer_id].items():
            buf = self.buffers[slot][pname]
            self.memcpy.async_copy(
                buf.data_ptr(),
                peer_view.data_ptr(),
                peer_view.nbytes,
                self.comm_stream.cuda_stream,
            )

        # RAW: mark prefetch done for this slot
        self._prefetch_events[slot].record(self.comm_stream)

    def _enqueue_cycle(self, cycle: int):
        """Enqueue one cycle exactly once in the current eager forward."""
        if cycle in self._queued_cycles:
            return
        layers = self._cycle_layers.get(cycle)
        if not layers:
            return
        self._queued_cycles.add(cycle)
        for layer_id in layers:
            self._do_prefetch(layer_id)

    def _build_cycle_schedule(self):
        """Build compute-order cycle membership and stable slot identities."""
        self._num_cycles = (self.num_layers + self.dp_size - 1) // self.dp_size
        self._cycle_cache_depth = min(self.cache_cycles, self._num_cycles)
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
        """Allocate serial layer slots or the eager cycle cache."""
        if self.enable_eager_overlap:
            num_slots = self._cycle_cache_depth * len(self._remote_positions)
        else:
            num_slots = min(self.cache_cycles, len(non_local_layers))
        if num_slots == 0:
            return

        self._prefetch_events = [torch.cuda.Event() for _ in range(num_slots)]
        self._consume_events = [torch.cuda.Event() for _ in range(num_slots)]

        # Get shapes from the first non-local layer (weights still intact at this point)
        ref_layer = layers[non_local_layers[0]]
        param_shapes = {}
        for pname, param in self._get_ffn_params(ref_layer):
            param_shapes[pname] = (param.shape, param.dtype)

        # Allocate num_slots buffers
        device = torch.cuda.current_device()
        for s in range(num_slots):
            self.buffers[s] = {}
            for pname, (shape, dtype) in param_shapes.items():
                self.buffers[s][pname] = torch.empty(shape, dtype=dtype, device=device)

        if self.enable_eager_overlap:
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
            for pname, pv in self.peer_views[lid].items():
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
