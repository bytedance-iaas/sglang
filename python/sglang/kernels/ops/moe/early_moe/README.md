# Fused early expert operator

The CUDA cooperative kernel owns readiness polling, local assignment bucketing,
FP8 preparation, M64 FFN scheduling, stop admission, and completion publication.
After observing all-ready, it finishes admitted M64 work before stopping. Only
completed `(token, top-k slot)` assignments are removed from the residual IDs.
Partial blocks are allowed. The inflight limit bounds storage, not total work.
Only the production BN256, four-stage math path is retained; M remains 64.
The obsolete BN64/BN128 implementations and their unused headers were removed.

`Geometry(H, I, topk, physical_experts, local_slots)` is explicit. H must be a
multiple of 256, I a multiple of 128, and physical slots must evenly cover the
process group. The ready mask supports up to 32 ranks, including rank 31.
The package specializes and caches native code by these dimensions. Model
names/profiles are confined to `test/manual/early_moe/profiles.py`.

Inputs use BF16 activations and already-prepared SM90 FP8 SwiGLU expert tensors:
L1 `[slots, 2*I, H]`, L2 `[slots, H, I]`, FP32 128x128 weight scales, and gran-8
interleaved gate/up rows. Routing IDs are **physical** despite retained legacy
`logical_*` ABI names. Effective top-k weights include routed scaling exactly once.

Results borrow context scratch: consume them before another layer uses that
scratch. Contexts bind one CUDA stream and stable expert tensor addresses.
Migration must not overwrite weights until the local operator completes.
An explicit `stop_signal` supports an external readiness collective; that
collective is outside the fused compute kernel. IPC readiness remains in-kernel.

Validation entrypoints are `test/manual/early_moe/operator_smoke.py` and
`bench_staggered.py`. They require no model weights and use synthetic gate delays.
The 8-GPU fixture is a test configuration, not an EP-size runtime restriction.
The tensor-math oracle uses explicit tolerances. Native-backend integration is
separately checked for exact equality by `check_native_combine.py`: completed
contributions enter the native top-k reduction before its final BF16 store.
See `backend/README.md` for the required optional API and backend patch. Adding
early values to an already rounded remaining sum is no longer supported by
the model bridge. Exactness on a tested model is not a guarantee for all FP8
geometries or quantization settings.
