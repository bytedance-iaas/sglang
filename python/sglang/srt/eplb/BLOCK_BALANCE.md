# Block-based expert balancing

`--eplb-algorithm block --eplb-block-size 64` chooses replica placement from
recorded `[step, layer, expert]` counts. It preserves the native DeepSeek replica
counts. A candidate must not increase Bmax or Tmax on any observed step. This is
a historical estimate, not a guarantee for an unseen request.

`--ep-dispatch-algorithm block --moe-a2a-backend megamoe` selects runtime
Block-LPLB. Gate results first use the live physical map with local-replica
preference. All EP ranks, including ranks with no tokens, then gather counts and
compute exact integer quotas. Routing conserves every `(token, top-k slot)`.
The native `lp` option is unchanged. Shared experts must remain separate.

The common unit is `SGLANG_LPLB_BLOCK_SIZE=64` (also accepts 128). Bmax is the
maximum across ranks of `sum_expert(ceil(assignments / block_size))`; Bsum is not
the objective. Average Bmax/Bmean per **step and layer** before aggregating.
This unit is a load proxy, not a count of CUDA CTAs or a runtime prediction.

`SGLANG_LPLB_BLOCK_FAST=1` reuses per-layer buffers and the caller's NCCL stream.
`SGLANG_LPLB_BLOCK_PLANNER_VARIANT=boundary` selects the optimized planner;
`reference` and `warp` remain available as numerical references. Use only the
`BLOCK_*` names; obsolete `TAIL_*` aliases have been removed.

Current runtime scope: eager SM90 FP8 MegaMoE. CUDA graph capture and
provider-owned packed MoE layouts are not supported by this adapter. The
ordinary SGLang path is used unless the block feature is explicitly enabled.

## Feature switches and migration

Each extension is opt-in. The named switches are `--enable-block-eplb`,
`--enable-block-lplb`, `--enable-expert-migration`,
`--enable-early-expert-compute`, and `--enable-fused-early-moe`.
The first selects the placement policy; `--enable-eplb` still controls periodic
updates, and `--init-expert-location` still controls the initial layout.
Migration requires Block-LPLB. Early computation requires Block-LPLB and the
fused kernel, but does not require migration.

Migration budget is **per rank, per layer**, independently for sends and receives
(`--expert-migration-budget 1`, operator capacity 1..4). All selected copies read
old source rows into staging. Every rank fences before active rows are
replaced. The new mapping and the weights become visible before remaining MoE.
The CPU metadata mirror is updated after the remaining kernel is enqueued.

`--expert-migration-transport auto` uses IPC on one host and NCCL P2P across
hosts. `ipc` requires peer-accessible GPUs; `nccl` is portable but currently reads
a small command list on the CPU, so its overhead is not equivalent to device IPC.
`--block-moe-validate` enables conservation/quota/weight checks;
`--block-moe-audit` logs actual per-layer **remaining-work** B/T counts and
per-rank early-work counters. Early work has already been removed from the
remaining quotas; do not compare these counts with total-work counts without
accounting for it.

Model dimensions and the EP world size come from the model, metadata, and group.
Planner rank masks support up to 32 ranks. Physical expert slots must divide
exactly by EP size; for example, 256 logical experts plus R8 cannot be evenly
sharded over EP16. Choose a compatible redundant-slot count explicitly; the code
does not silently round it or change R8 to a different configuration.

## Early computation and numerical contract

The fused SM90 kernel computes local `(token, top-k slot)` assignments in M64
blocks while polling peer-ready epochs. It stops admitting work when all peers
are ready, drains admitted work, and marks only completed assignments `-1` in
the remaining route. A token may still have other, uncomputed experts. It is
never removed wholesale. Scratch is reused only after native combine finishes.

IPC readiness and FFN computation share a kernel. With
`--early-moe-ready-transport nccl`, the all-reduce runs on a separate stream and
publishes a device epoch; this is overlap, not in-kernel NCCL fusion.
`--early-moe-max-inflight` bounds concurrent tasks, not total admitted work.

The backend now reads completed early contributions in its original top-k slot
order, accumulates in FP32, and writes BF16 once. There is no external merge of
an already rounded remaining sum. The optional backend contract and IAAS patch
are in `python/sglang/kernels/ops/moe/early_moe/backend/`; early mode rejects a
backend without this contract. The regression fixture loads checkpoint weights
through `test/manual/early_moe/checkpoint_weights.py`.

`test/manual/early_moe/check_native_combine.py` requires exact equality for
real-weight contributions, poisoned unread slots and actual fused execution.
The DSV4 8-GPU live-input audit also passed exact checks across all 43 layers.
Other model geometries, quantization settings and distributed backends still
require their own numerical regression; this is not a universal quality claim.

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
