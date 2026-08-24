"""SiDP manager for IPC weight sharing and graph-compatible serial prefetch.

The current correctness-first path implements D2/D3/D6/D7/D8. D4 scheduling
and D5 cycle-level rolling overlap have scaffolding here but are not active.
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

        # D2: TCPStore is created lazily in setup() so that all ranks
        # have finished load_model() before any rank tries to connect.
        self.store = None
        self._rdzv_host = config.rdzv_host
        self._rdzv_port = config.rdzv_port

        # D7: DMA engine wrapper
        self.memcpy = SidpCudaMemcpy()

        # D8: comm stream for prefetch (will be fork/joined into graph)
        self.comm_stream = torch.cuda.Stream()

        # Per-layer-slot events. The serial path currently uses RAW only;
        # consume events are reserved for the future D5 WAR pipeline.
        self._prefetch_events: List[torch.cuda.Event] = []
        self._consume_events: List[torch.cuda.Event] = []

        # Populated by setup()
        self.peer_views: Dict[int, Dict[str, torch.Tensor]] = {}
        self.buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        self._layer_to_slot: Dict[int, int] = {}
        self._non_local_layers: List[int] = []
        self._fetch_schedule: List[int] = []
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
        logger.info(
            f"[SiDP rank{self.dp_rank}] local={len(local_layers)}, "
            f"non_local={len(non_local_layers)}"
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

        # Allocate the current serial layer-slot buffers BEFORE releasing
        # weights (their shapes are still needed here).
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

        # Initialize consume events for the dormant eager-overlap helper.
        for evt in self._consume_events:
            evt.record(torch.cuda.current_stream())

        # D4 schedule scaffolding; the current serial runtime does not use it.
        self._fetch_schedule = prefetch_order(
            self.dp_rank, self.dp_size, self.k, self.num_layers
        )

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

        Issues the DMA prefetch for THIS layer inside the current (capture)
        stream scope using fork/join, so the whole dependency chain lives in
        one graph:
          fork:  comm_stream.wait_stream(compute_stream)
          copy:  cudaMemcpyAsync(buf <- peer_view) on comm_stream
          record: prefetch_event on comm_stream
          join:  compute_stream.wait_event(prefetch_event)
        The MLP then reads buf via the rebound weight.data.
        """
        slot = self._layer_to_slot[layer_id]
        compute_stream = torch.cuda.current_stream()

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
        """Called AFTER the MLP GEMM. No-op for the self-contained fork/join
        design (kept for API symmetry and future double-buffer overlap)."""
        return

    def prefetch_first_layers(self):
        """No-op: prefetch is issued per-layer inside wait_prefetch (must stay
        inside the capture scope). Kept for eager-path API symmetry."""
        return

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

    def _trigger_next_prefetch(self, just_consumed_layer_id: int):
        """Dormant layer-lookahead helper for the future eager overlap path."""
        try:
            idx = self._non_local_layers.index(just_consumed_layer_id)
        except ValueError:
            return
        next_idx = idx + self.cache_cycles
        if next_idx < len(self._non_local_layers):
            self._do_prefetch(self._non_local_layers[next_idx])

    def _alloc_buffers(self, layers, non_local_layers):
        """Allocate the serial fallback's cache_cycles layer slots.

        These are layer slots, not the final D5 cycle cache, whose target size
        is ``cache_cycles * (dp_size - k)`` layer slots.
        """
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

        # Assign layer → slot mapping (round-robin)
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
