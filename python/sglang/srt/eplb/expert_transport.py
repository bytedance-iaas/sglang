"""Expert-row transfers with explicit read-before-overwrite ordering.

IPC uses GPU-driven pulls on one peer-accessible host. NCCL is the portable
cross-host path: its small command list is read on the host before P2P enqueue.
Both paths stage old source rows and require a collective fence before commit.
"""

import socket
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.srt.eplb.ipc_expert_transfer import ExpertIpcPeers

# Native copy ABI has four staging entries per destination. This is an operator
# capacity, not a model dimension or the configured send/receive budget.
MAX_MIGRATION_BUDGET = 4


@lru_cache(None)
def group_hosts(cpu_group):
    # Hostnames can differ between containers on the same physical machine.
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    host = boot_id.read_text().strip() if boot_id.is_file() else socket.gethostname()
    hosts = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(hosts, host, group=cpu_group)
    return tuple(hosts)


def select_transport(requested, cpu_group):
    same_host = len(set(group_hosts(cpu_group))) == 1
    if requested == "ipc" and not same_host:
        raise ValueError("CUDA IPC cannot address experts on another host; use NCCL")
    return ("ipc" if same_host else "nccl") if requested == "auto" else requested


class ExpertTransport:
    def __init__(self, tensors, group, rank, layer, shape, requested, budget=1):
        if not 1 <= budget <= MAX_MIGRATION_BUDGET:
            raise ValueError("Migration budget exceeds the native staging capacity")
        self.capacity = budget
        self.tensors, self.group, self.rank = tensors, group, rank
        self.shape = shape
        for tensor in tensors:
            row_stride = tensor.stride(0) * tensor.element_size()
            if (
                not tensor.is_contiguous()
                or tensor.shape[0] != shape.slots
                or row_stride % 16
                or tensor[:1].nbytes % 16
            ):
                raise ValueError(
                    "Expert copy requires contiguous rows aligned to 16 bytes"
                )
        self.kind = select_transport(requested, group.cpu_group)
        self.staging = [
            tuple(torch.empty_like(t[:1]) for t in tensors) for _ in range(budget)
        ]
        self.ipc = (
            ExpertIpcPeers(tensors, group.cpu_group, rank)
            if self.kind == "ipc"
            else None
        )
        rows = []
        for j, tensor in enumerate(tensors):
            # NCCL never dereferences the remote entries of this table.
            peers = [
                (
                    tensor.data_ptr()
                    if r == rank
                    else (self.ipc.ptrs[r][j] if self.ipc is not None else 0)
                )
                for r in range(shape.world)
            ]
            rows.append(
                peers
                + [s[j].data_ptr() for s in self.staging]
                + [0] * (MAX_MIGRATION_BUDGET - budget)
                + [tensor.stride(0) * tensor.element_size(), tensor[:1].nbytes]
            )
        self.addresses = torch.tensor(rows, device=tensors[0].device, dtype=torch.int64)
        self.fence = torch.zeros(1, device=tensors[0].device, dtype=torch.int32)
        # P2P's first call is not required to include idle ranks after this fence.
        dist.all_reduce(self.fence, group=group.device_group)

    def read(self, planner, weight_ok, budget):
        if budget > self.capacity:
            raise ValueError(
                "Requested migration exceeds the allocated staging capacity"
            )
        if self.kind == "ipc":
            planner.mm.copy(
                planner.info,
                planner.copies,
                self.addresses,
                weight_ok,
                self.rank,
                0,
                16,
                budget,
            )
            return
        count = int(planner.info[0].item())
        copies = planner.copies[:count].cpu().tolist()
        ordinals = [0] * self.shape.world
        operations = []
        for source, destination in copies:
            src_rank, src_slot = divmod(source, self.shape.slots)
            dst_rank, _ = divmod(destination, self.shape.slots)
            ordinal = ordinals[dst_rank]
            ordinals[dst_rank] += 1
            if ordinal >= budget:
                raise RuntimeError("Migration exceeds the destination budget")
            for j, tensor in enumerate(self.tensors):
                if src_rank == dst_rank == self.rank:
                    self.staging[ordinal][j].copy_(tensor[src_slot : src_slot + 1])
                elif self.rank == src_rank:
                    operations.append(
                        dist.P2POp(
                            dist.isend,
                            tensor[src_slot : src_slot + 1].view(torch.uint8),
                            dist.get_global_rank(self.group.device_group, dst_rank),
                            self.group.device_group,
                        )
                    )
                elif self.rank == dst_rank:
                    operations.append(
                        dist.P2POp(
                            dist.irecv,
                            self.staging[ordinal][j].view(torch.uint8),
                            dist.get_global_rank(self.group.device_group, src_rank),
                            self.group.device_group,
                        )
                    )
        for request in dist.batch_isend_irecv(operations) if operations else []:
            request.wait()

    def commit(self, planner, weight_ok, budget, validate):
        # Even ranks without transfers must fence, otherwise a destination could
        # overwrite a source while a third rank is still reading that source.
        if self.group.pynccl_comm is not None:
            with self.group.pynccl_comm.change_state(enable=True):
                self.group.pynccl_comm.all_reduce(self.fence)
        else:
            dist.all_reduce(self.fence, group=self.group.device_group)
        planner.mm.copy(
            planner.info,
            planner.copies,
            self.addresses,
            weight_ok,
            self.rank,
            1,
            32,
            budget,
        )
        if validate:
            planner.mm.copy(
                planner.info,
                planner.copies,
                self.addresses,
                weight_ok,
                self.rank,
                2,
                32,
                budget,
            )
