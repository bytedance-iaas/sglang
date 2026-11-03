"""Framework bridge for the fused M64 local-expert operator.

IPC readiness is polled inside the compute kernel. Cross-host readiness uses a
separate NCCL stream and a device stop epoch. The latter is communication/
compute overlap, not a claim that NCCL has been fused into the FFN kernel.
"""

import logging

import torch
import torch.distributed as dist

from sglang.kernels.ops.moe.early_moe import (
    STAT_NAMES,
    ActiveWeights,
    create_early_moe_context,
    early_moe,
)
from sglang.srt.eplb.block_options import options
from sglang.srt.eplb.expert_transport import select_transport
from sglang.srt.eplb.moe_shape import MoeShape

logger = logging.getLogger(__name__)
_states = {}
_scratch = {}


class EarlyState:
    def __init__(self, moe, metadata, group, capacity):
        cfg = options()
        shape = self.shape = MoeShape.from_layer(moe, metadata)
        self.layer, self.group = moe.layer_id, group
        rank = self.rank = dist.get_rank(group.device_group)
        self.transport = select_transport(
            cfg.early_moe_ready_transport, group.cpu_group
        )
        device = metadata.physical_to_logical_map.device
        self.stop = (
            torch.zeros(1, dtype=torch.uint64, device=device)
            if self.transport == "nccl"
            else None
        )
        self.ready_stream = torch.cuda.Stream(device=device)
        self.ready_event = torch.cuda.Event()
        self.barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.weights = ActiveWeights(
            *moe.experts.mega_l1_weights, *moe.experts.mega_l2_weights
        )
        self.local_map = torch.full(
            (shape.physical,), -1, dtype=torch.int32, device=device
        )
        first = rank * shape.slots
        self.local_map[first : first + shape.slots] = torch.arange(
            shape.slots, device=device, dtype=torch.int32
        )
        self.reverse = torch.arange(
            first, first + shape.slots, device=device, dtype=torch.int32
        )
        key = (
            device,
            capacity,
            shape.hidden,
            moe.config.moe_intermediate_size,
            shape.topk,
            shape.physical,
            shape.slots,
            cfg.early_moe_max_inflight,
        )
        self.context = create_early_moe_context(
            group.cpu_group,
            max_tokens=capacity,
            hidden=shape.hidden,
            intermediate=moe.config.moe_intermediate_size,
            topk=shape.topk,
            logical_experts=shape.physical,
            local_slots=shape.slots,
            max_inflight_blocks=cfg.early_moe_max_inflight,
            max_resident_ctas=cfg.early_moe_max_ctas,
            stages=4,
            tile_n=256,
            bulk_prepare=True,
            diagnostic=cfg.block_moe_validate,
            stop_signal=self.stop,
            external_ready=self.transport == "nccl",
            shared_scratch=_scratch.get(key),
        )
        _scratch.setdefault(key, self.context.scratch)
        self.context.bind(self.weights, self.local_map, self.reverse, mapping_version=1)
        self.epoch = 0
        self.effective_weights = torch.empty(
            (capacity, shape.topk), dtype=torch.float32, device=device
        )
        self.host_stats = torch.empty_like(
            self.context.result.stats[0], device="cpu", pin_memory=True
        )
        self.snapshot = torch.cuda.Event()
        dist.all_reduce(self.barrier, group=group.device_group)
        self.ready_stream.wait_stream(torch.cuda.current_stream())

    def run(self, prepared):
        self.epoch += 1
        moe, n = prepared.moe, prepared.num_tokens
        factor = (
            1.0
            if moe.experts.should_fuse_routed_scaling_factor_in_topk
            else float(moe.routed_scaling_factor)
        )
        weights = prepared.topk_weights
        if factor != 1.0:
            weights = torch.mul(weights, factor, out=self.effective_weights[:n])
        if self.transport == "nccl":
            self.ready_event.record()
            with torch.cuda.stream(self.ready_stream):
                self.ready_stream.wait_event(self.ready_event)
                with torch.profiler.record_function("early_ready_nccl"):
                    if self.group.pynccl_comm is not None:
                        with self.group.pynccl_comm.change_state(enable=True):
                            self.group.pynccl_comm.all_reduce(self.barrier)
                    else:
                        dist.all_reduce(self.barrier, group=self.group.device_group)
                    self.context.signal_stop(self.epoch, self.ready_stream)
        with torch.profiler.record_function("early_moe_fused_m64"):
            self.result = early_moe(
                prepared.hidden_states,
                prepared.topk_ids,
                weights,
                self.weights,
                self.local_map,
                1,
                self.epoch,
                context=self.context,
                activation_clamp=getattr(moe.config, "swiglu_limit", None),
            )
        # Status is never optional: a peer timeout must not silently produce output.
        self.host_stats.copy_(self.result.stats[0], non_blocking=True)
        self.snapshot.record()
        return self.result.remaining_physical_ids

    def combine_inputs(self):
        """Borrow the arena until remaining MegaMoE completes on this stream.

        Read completed BF16 contributions in the native top-k reduction; never
        add them to an already rounded remaining sum. Migration may overwrite
        expert weights, but cannot overwrite this per-assignment output arena.
        """
        return {
            "early_values": self.result.early_values.view(
                -1, self.shape.topk, self.shape.hidden
            ),
            "early_done": self.result.early_done,
        }

    def finish(self):
        self.snapshot.synchronize()
        stats = dict(zip(STAT_NAMES, self.host_stats.tolist()))
        if stats["status"]:
            raise RuntimeError(f"Early-MoE failed: {stats}")
        if options().block_moe_audit:
            logger.info(
                "Early-MoE rank=%s layer=%s transport=%s stats=%s",
                self.rank,
                self.layer,
                self.transport,
                stats,
            )


def state_for(moe, metadata, group, capacity):
    key = (id(metadata), moe.layer_id, capacity)
    if key not in _states:
        _states[key] = EarlyState(moe, metadata, group, capacity)
    return _states[key]
