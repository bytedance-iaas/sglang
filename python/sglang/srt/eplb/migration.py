"""Current-forward migration: GPU plan, stage, fence, commit, then route.

All EP ranks execute this protocol, including empty ranks. CPU mirrors are
refreshed after the remaining MegaMoE has been enqueued; they never decide the
current routing. Native LP state is untouched because this is the block path.
"""

import copy
import logging

import torch
import torch.distributed as dist

from sglang.srt.eplb import block_dispatch
from sglang.srt.eplb.balance_config import BLOCK_SIZE
from sglang.srt.eplb.block_options import options
from sglang.srt.eplb.cpu_metadata_reuse import CpuMetadataMirror
from sglang.srt.eplb.expert_transport import ExpertTransport
from sglang.srt.eplb.expert_weights import get_expert_weights
from sglang.srt.eplb.moe_shape import MoeShape
from sglang.srt.eplb.multi_swap import MultiPlanner

logger = logging.getLogger(__name__)
_states = {}


class MigrationState:
    def __init__(self, moe, metadata, group, rank, key):
        self.shape = shape = MoeShape.from_layer(moe, metadata)
        self.layer, self.rank, self.key = moe.layer_id, rank, key
        self.budget = options().expert_migration_budget
        self.group = group
        device = metadata.physical_to_logical_map.device
        self.planner = MultiPlanner(device, self.budget, shape)
        self.weights = get_expert_weights(moe)
        self.transport = ExpertTransport(
            self.weights.tensors,
            group,
            rank,
            self.layer,
            shape,
            options().expert_migration_transport,
            budget=self.budget,
        )
        self.block = copy.copy(block_dispatch._CACHE[key])
        self.block.phy = torch.empty(
            2 * shape.physical, dtype=torch.int32, device=device
        )
        self.block.pairs = torch.empty(
            (shape.pairs, 2), dtype=torch.int32, device=device
        )
        self.lookup = torch.empty(shape.logical, dtype=torch.int32, device=device)
        self.work = torch.empty(shape.device_work, dtype=torch.int32, device=device)
        self.counts = torch.empty(
            (shape.world, shape.physical), dtype=torch.int32, device=device
        )
        self.cursor = torch.empty(shape.logical, dtype=torch.int32, device=device)
        self.weight_ok = torch.empty(1, dtype=torch.int32, device=device)
        self.old_mapping = torch.empty(shape.physical, dtype=torch.int64, device=device)
        self.packet = torch.empty(shape.snapshot_size, dtype=torch.int32, device=device)
        self.host_packet = torch.empty_like(self.packet, device="cpu", pin_memory=True)
        self.mirror = CpuMetadataMirror(
            metadata.physical_to_logical_map_cpu[self.layer],
            metadata.logical_to_all_physical_map_cpu[self.layer],
            rank,
            shape.world,
        )
        self.stream = torch.cuda.Stream(device=device)
        self.ready, self.committed, self.snapshot = (
            torch.cuda.Event() for _ in range(3)
        )
        self.stream.wait_stream(torch.cuda.current_stream())

    def begin(self, metadata):
        """Publish routing inputs before submitting independent shared work."""
        self.old_mapping.copy_(metadata.physical_to_logical_map[self.layer])
        self.weight_ok.fill_(1)
        self.cursor.zero_()
        self.ready.record()

    def enqueue(self, metadata, source_counts, quota, ids):
        p = self.planner
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.ready)
            with torch.profiler.record_function("block_migration_plan"):
                p.plan(self.old_mapping, quota)
            with torch.profiler.record_function("block_migration_read"):
                self.transport.read(p, self.weight_ok, self.budget)
            with torch.profiler.record_function("block_migration_commit"):
                self.transport.commit(
                    p, self.weight_ok, self.budget, options().block_moe_validate
                )
            p.mod.metadata(
                p.mapping,
                metadata.physical_to_logical_map[self.layer],
                metadata.logical_to_all_physical_map[self.layer],
                metadata.logical_to_all_physical_map_num_valid[self.layer],
                self.block.phy,
                self.block.pairs,
                self.lookup,
                self.work,
                self.rank,
            )
            p.mod.counts(p.mapping, source_counts, self.lookup, p.work, self.counts)
            self.committed.record()
            p.mm.snapshot(
                p.info,
                p.copies,
                p.history,
                quota,
                p.mapping,
                p.quota,
                p.counter,
                self.packet,
            )
            self.host_packet.copy_(self.packet, non_blocking=True)
            self.snapshot.record()
        torch.cuda.current_stream().wait_event(self.committed)
        flat = ids.contiguous().view(-1)
        output = torch.empty_like(flat)
        p.mod.route(
            flat,
            output,
            p.route_info,
            self.old_mapping,
            self.lookup,
            self.block.phy,
            self.counts,
            p.quota,
            self.cursor,
            self.rank,
        )
        block_dispatch._CACHE[self.key] = self.block
        if options().block_moe_validate:
            valid = ids >= 0
            before = torch.where(valid, self.old_mapping[ids.clamp_min(0).long()], -1)
            after = torch.where(
                output.reshape_as(ids) >= 0,
                p.mapping[output.reshape_as(ids).clamp_min(0).long()],
                -1,
            )
            self.logical_ok = (before == after).all()
            self.realized = torch.empty_like(p.quota)
            self.block.mod.hist(output, self.realized)
        return output.reshape_as(ids)

    def finish(self):
        """Consume the snapshot only after the remaining MoE launch."""
        self.snapshot.synchronize()
        shape = self.shape
        packet = self.host_packet.tolist()
        n = packet[0]
        offset = shape.snapshot_header
        mapping = packet[offset + shape.physical : offset + 2 * shape.physical]
        if n:
            self.mirror.update(mapping)
        if options().block_moe_validate:
            dist.all_reduce(self.realized, group=self.group.device_group)
            checks = torch.stack(
                [
                    self.logical_ok,
                    self.weight_ok[0].bool(),
                    (self.realized == self.planner.quota).all(),
                ]
            ).to(torch.int32)
            dist.all_reduce(checks, op=dist.ReduceOp.MIN, group=self.group.device_group)
            if not bool(checks.all()):
                raise RuntimeError(
                    f"Migration invariant failed: {checks.cpu().tolist()}"
                )
        if options().block_moe_audit and self.rank == 0:
            from sglang.srt.eplb.capacity_planner import metrics

            before = packet[offset : offset + shape.physical]
            after = packet[offset + 2 * shape.physical : offset + 3 * shape.physical]
            logger.info(
                "Block migration layer=%s copies=%s B/T before=%s after=%s transport=%s",
                self.layer,
                n,
                metrics(before, world=shape.world),
                metrics(after, world=shape.world),
                self.transport.kind,
            )


def state_for(moe, metadata, group):
    phy = metadata.physical_to_logical_map[moe.layer_id]
    key = (moe.layer_id, id(metadata), phy.data_ptr(), BLOCK_SIZE)
    if key not in _states:
        _states[key] = MigrationState(
            moe, metadata, group, dist.get_rank(group.device_group), key
        )
    return _states[key]
