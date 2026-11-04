"""SGLang cache and collectives for global Block-LPLB."""

import os

import torch

from sglang.kernels.ops.lplb.block_planner import BlockLPLB
from sglang.srt.eplb.balance_config import BLOCK_SIZE

_CACHE = {}
_LAST_CURRENT = {}


def global_block_route(ids, layer_id, ep_group):
    """Collective MUST be called by active AND idle ranks, once per MoE forward."""
    import torch.distributed as dist

    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Global block routing currently requires eager execution")
    metadata = get_global_expert_location_metadata()
    phy = metadata.physical_to_logical_map[layer_id]
    key = (layer_id, id(metadata), phy.data_ptr(), BLOCK_SIZE)
    # EPLB metadata.update() clears this cache after an in-place mapping change.
    if key not in _CACHE:
        _CACHE[key] = BlockLPLB(phy, dist.get_world_size(ep_group), key[-1])
    planner = _CACHE[key]
    fast = os.getenv("SGLANG_LPLB_BLOCK_FAST", "0") == "1"
    if fast:
        # Per-layer buffers, same-stream sequential use. Audit snapshots must clone.
        if not hasattr(planner, "workspace"):
            from sglang.srt.distributed.parallel_state import get_moe_ep_group

            coordinator = get_moe_ep_group()
            assert coordinator.device_group is ep_group
            planner.comm = coordinator.pynccl_comm
            planner.rank = dist.get_rank(ep_group)
            planner.workspace = (
                torch.empty((planner.p,), dtype=torch.int32, device=ids.device),
                torch.empty(
                    (planner.r, planner.p), dtype=torch.int32, device=ids.device
                ),
                torch.empty((planner.p,), dtype=torch.int32, device=ids.device),
                torch.empty((planner.logical,), dtype=torch.int32, device=ids.device),
            )
            planner.outputs = {}
            print(
                f"BLOCK_LPLB_FAST rank={planner.rank} transport={'pynccl' if planner.comm else 'torch'}",
                flush=True,
            )
        local, gathered, quota, cursor = planner.workspace
        shape = tuple(ids.shape)
        if shape not in planner.outputs:
            planner.outputs[shape] = torch.empty(
                ids.numel(), dtype=torch.int32, device=ids.device
            )
        out = planner.outputs[shape]
    else:
        local = torch.empty((planner.p,), dtype=torch.int32, device=ids.device)
        gathered = torch.empty(
            (planner.r, planner.p), dtype=torch.int32, device=ids.device
        )
        quota = None
        out = None
        cursor = None
    planner.mod.hist(ids.contiguous().view(-1), local)
    if fast and planner.comm is not None:
        # Enqueue NCCL on the caller's CUDA stream, avoiding c10d stream handoffs.
        with planner.comm.change_state(enable=True):
            planner.comm.all_gather(gathered.view(-1), local)
    else:
        dist.all_gather_into_tensor(gathered.view(-1), local, group=ep_group)
    quota = planner.plan(gathered, quota)
    from sglang.srt.eplb.block_options import migration_enabled

    if migration_enabled():
        _LAST_CURRENT[layer_id] = (gathered, quota)
    return planner.route(ids, gathered, quota, dist.get_rank(ep_group), out, cursor)


def invalidate_layers(layer_ids):
    """Invalidate cached replica pairs after an external placement update."""
    changed = set(layer_ids)
    for key in list(_CACHE):
        if key[0] in changed:
            del _CACHE[key]
    for layer in changed:
        _LAST_CURRENT.pop(layer, None)
