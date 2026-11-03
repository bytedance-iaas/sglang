# Block expert balancing

All feature flags default to off. This branch is part of a stacked series; the
next branch adds a separate opt-in feature without changing the native path.

## Block-EPLB

`--enable-block-eplb --eplb-block-size 64` selects per-step Bmax/Tmax placement.
`--enable-eplb` still controls periodic updates. Native replica counts are kept.
Candidates must not increase either tail on any observed step; future inputs
are not covered by that historical bound. Bmax/Bmean is averaged per step and
layer, not computed from aggregate counts.

## Block-LPLB

`--enable-block-lplb --moe-a2a-backend megamoe` enables global integer quotas and
live-map token routing. Use `SGLANG_LPLB_BLOCK_SIZE=64` (or 128), matching EPLB.
All ranks participate, including empty ranks. The original `lp` policy remains.
`SGLANG_LPLB_BLOCK_FAST=1` reuses buffers and the NCCL stream.
`SGLANG_LPLB_BLOCK_PLANNER_VARIANT=boundary` is the optimized quota planner;
`reference` and `warp` are comparison paths. Old `TAIL_*` aliases are removed.
Shared experts stay separate. Model graph capture is not supported.

## Expert migration

`--enable-expert-migration` requires Block-LPLB.

Migration budget is **per rank, per layer**, independently for sends and receives
(`--expert-migration-budget 1`, operator capacity 1..4). All selected copies read
old source rows into staging. Every rank fences before active rows are
replaced. The new mapping and the weights become visible before remaining MoE.
The CPU metadata mirror is updated after the remaining kernel is enqueued.

`--expert-migration-transport auto` uses IPC on one host and NCCL P2P across
hosts. `ipc` requires peer-accessible GPUs; `nccl` is portable but currently reads
a small command list on the CPU, so its overhead is not equivalent to device IPC.
`--block-moe-validate` enables conservation/quota/weight checks;
`--block-moe-audit` logs actual per-layer **remaining-work** B/T counts for each forward. The early-compute bridge is added in a later branch.

Model dimensions and the EP world size come from the model, metadata, and group.
Planner rank masks support up to 32 ranks. Physical expert slots must divide
exactly by EP size; for example, 256 logical experts plus R8 cannot be evenly
sharded over EP16. Choose a compatible redundant-slot count explicitly; the code
does not silently round it or change R8 to a different configuration.


With migration enabled, statistical EPLB counts are resolved to logical IDs at
gate time. This prevents later physical-map updates from relabeling historical
counts. The native distribution recorder's physical counts are provisional
pre-balance counts; use the migration audit for finalized remaining quotas.

## Distributed validation scope

The planner supports rank masks through EP32; model dimensions are explicit.
The migration transport and ready protocol include NCCL fallbacks for multiple
hosts. This does not imply that every selected MegaMoE backend supports RDMA.
`--block-moe-backend iaas` selects its RDMA implementation when the EP group
spans hosts. The installed IAAS wheel must include a compatible RDMA extension,
and its NVSHMEM allocator is selected before communicators are created. A
non-RDMA backend fails explicitly for a multi-host group. Hardware validation currently covers eight local GPUs. EP16/EP32 planner simulations do not establish multi-node performance.
