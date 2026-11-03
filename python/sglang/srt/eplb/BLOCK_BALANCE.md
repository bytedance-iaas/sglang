# Block expert balancing

All feature flags default to off. This branch is part of a stacked series; the
next branch adds a separate opt-in feature without changing the native path.

## Block-EPLB

`--enable-block-eplb --eplb-block-size 64` selects per-step Bmax/Tmax placement.
`--enable-eplb` still controls periodic updates. Native replica counts are kept.
Candidates must not increase either tail on any observed step; future inputs
are not covered by that historical bound. Bmax/Bmean is averaged per step and
layer, not computed from aggregate counts.
