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
