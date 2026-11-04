# Native early combine

Early assignments must enter the remaining backend's original top-k reduction.
Adding them to an already rounded BF16 remaining sum changes model outputs.

`iaas-early-combine.patch` is the small backend extension required by this
integration. Apply it at the IAAS repository root with `git apply --check`, then
`git apply`, and rebuild/install the native extension with that repository's
normal build process. `source_hashes.json` records the exact source snapshot
and patched file hashes; do not force-apply to an incompatible backend version.
The patch is kept here so the SGLang feature can be restored without copying
the full MegaMoE implementation into this repository.

The Python package and native library must both be updated. The backend exports
`EARLY_COMBINE_API_VERSION = 1` and an additional FFI entry; the existing FFI
entry and calls without early inputs remain supported. The kernel specializes
away early loads when the option is absent. The SGLang early flag remains
opt-in and rejects backends without this contract rather than silently using
the old rounded partial-sum merge.

## Optional inputs

`fp8_mega_moe(..., early_values=values, early_done=done)` accepts:

| Input | Contract |
|---|---|
| `early_values` | Contiguous CUDA BF16 `[capacity, topk, hidden]`, capacity at least current tokens |
| `early_done` | Contiguous CUDA uint8 `[capacity, topk]`, capacity at least current tokens |
| Dispatch IDs | Completed `(token, slot)` assignments are `-1`; unfinished assignments retain their routed physical IDs |
| Values | Each completed slot contains the same BF16 FC2 contribution as native MegaMoE, including routing weight exactly once |
| Lifetime | Producer completes on the launch stream or supplies an event dependency; inputs remain valid until MegaMoE finishes |

Dimensions come from the buffer and weights; there is no fixed rank count,
hidden size, expert count, or top-k in this interface. Both tensors must be
provided together. Unmarked values are never read, and an empty rank still
participates in distributed MegaMoE with zero output rows.

Only the combine load source changes: a completed slot reads local early
storage; another slot reads the native communication buffer. Both follow the
original ascending slot order, FP32 accumulation and final BF16 store. No extra
combine launch, assignment copy or peer communication is introduced. Dispatch
still excludes completed work. Migration may change unfinished physical IDs
but must preserve the logical assignment and top-k slot.

This patch targets the SM90 FP8 NVLink backend. RDMA, FP4 and other backends
must implement the same contract before enabling this integration; the API
version must not be advertised on an unsupported path. Multiple-node numerical
and performance validation has not been performed.

## Regression

`test/manual/early_moe/check_native_combine.py` checks native contributions,
poisoned unread slots, all/none/interleaved completion, uneven partial blocks,
an empty rank and actual fused computation. Supply `--model` for checkpoint
weights. A full-model test must additionally compare repeated identical
forwards and output tokens with early computation disabled. Exact equality on
one model/configuration does not imply quality equivalence on all models.
