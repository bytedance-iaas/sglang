# sglang-scheduler

Rust port of SGLang's CPU-only scheduler (`python/sglang/srt/managers/scheduler_cpu.py`).
Drives GPU `TpWorkerServer` processes over msgpack ZMQ IPC — the same wire format
the Python `CpuScheduler` speaks today.

The original Python `CpuScheduler` runs entirely on CPU and delegates all GPU work to
`TpWorkerServer` subprocesses via a typed object dispatcher. This crate is the Rust
replacement for that process: it talks the same `sock_send` / `sock_recv` protocol
(magic number `0xSG02`, msgspec-tagged msgpack frames) and drives the same `TpWorkerServer`
unchanged on the other side.

## Status — what's ported, what's stub

This crate now drives an end-to-end scheduling loop: tokenized requests come in over a
PULL socket, get admitted through the cache-aware batch builder, are dispatched to the
Python `TpWorkerServer`, replies are processed (KV released, prefixes inserted into the
radix cache), and memory-pressure recovery (cache eviction → retraction) runs when the
KV budget runs out.  The pieces still marked 🟡 are functional skeletons that cover the
common path; their stubbed branches (logprobs, grammar, speculative decoding, multi-DP
all_gather) are listed in the rightmost column.

| Component                              | State | Notes                                                                                       |
|----------------------------------------|-------|---------------------------------------------------------------------------------------------|
| Cargo / pyproject scaffold             | ✅    | Mirrors `rust/sglang-frontend` layout.                                                       |
| Wire types (msgspec.Struct mirrors)    | ✅    | Every RPC the worker exposes. See `src/wire/`.                                               |
| ZMQ transport (sock_send / sock_recv)  | ✅    | Pickle frames are rejected (the Rust scheduler avoids them on every path).                  |
| GPU worker handshake                   | ✅    | `GPUWorkerHandshakeReqInput` → `GPUWorkerHandshakeReqOutput`.                                |
| `CpuPageTracker` port                  | ✅    | Pure-data accounting helper, fully ported with dedup + page-0 strip (`src/memory/`).        |
| `WorkerClient` RPC surface             | ✅    | Typed wrappers for every method (`src/scheduler/worker_client.rs`).                          |
| Core types (`Req`, `SamplingParams`, `ForwardMode`, `ServerArgs`, `ScheduleBatch`) | ✅ | `src/types/`. `Req::check_finished` covers length + stop-token. |
| Waiting queue + scheduling policy      | ✅    | FCFS + LoF policies; `LongestPrefix` / `Priority` fall back to FCFS until their inputs land. |
| Batch builder                          | ✅    | Decode prep + prefill admission with radix-cache prefix scoring.  Admitted reqs `inc_lock_ref` the matched prefix so eviction can't drop it.  `LongestPrefix` policy fully wired. |
| Output processor                       | 🟡    | Token append + finish detection + page-tracker sync + extend `out_cache_loc` writeback + finished-req KV release via `ReqToTokenPool` + cache `insert` + `dec_lock_ref`.  Logprobs / grammar / spec still stub. |
| Scheduler event loop                   | ✅    | End-to-end wiring: tokenizer source → admission → forward → output processing → pressure recovery (cache evict + retract).  Running-side aborts apply `to_finish` on the next iteration. `event_loop.rs`. |
| Radix cache                            | ✅    | Arena tree + `match_prefix`/`insert`/`evict` + `inc_lock_ref`/`dec_lock_ref` with `protected_size` accounting.  Page-aligned eviction is the only follow-up (only matters when `page_size > 1`).  `src/radix_cache.rs`. |
| Retraction (memory pressure)           | ✅    | `retract_decode` + `evict_from_tree_cache` + `reclaim_for_decode` (cache-first, retract-fallback).  Retracted reqs release their `lock_ref`.  No HiSparse / disagg KV offload.  `src/scheduler/retract.rs`. |
| `ReqToTokenPool`                       | ✅    | Full alloc/free/write/read + double-free guard.  `src/memory/req_to_token_pool.rs`. |
| `ModelWorkerBatch` wire encoder        | ✅    | All 40+ fields of the Python dataclass plus `SamplingBatchInfo`.  `src/wire/model_worker_batch.rs`. |
| `tree_cache eviction` hook into pressure | ✅  | `retract::evict_from_tree_cache` + `reclaim_for_decode` (cache-first, retract-fallback).      |
| DP-attention sync                      | 🟡    | Single-DP local fallback (`dp_attn::prepare_local_dp_attention_sync`) stamps coherent `global_num_tokens` / `global_forward_mode`.  Multi-DP `all_gather` not ported (needs a cross-process group on the Rust side). |
| Sampling info / penalizers             | 🟡    | Per-req `temperatures`/`top_ps`/`top_ks`/`min_ps` + `is_all_greedy` / `need_*` flags + `sampling_seed` wired via `SamplingBatchInfoPayload::from_sampling_params`.  Penalizer orchestrator stays `None` — Python disables it via the `rainj-me` TODO. |
| Session controller                     | 🟡    | `SessionController` with `open` / `note_request` / `close`.  Full `SessionReqNode` branching tree not ported. |
| LoRA admin / weight updates / metrics  | 🟡    | Wire types + worker-client wrappers live; no top-level handlers / Python bridge yet.         |
| Tokenizer→scheduler request source     | ✅    | `PullSource` + `scheduler::drain_into` admit `TokenizedGenerate(Req\|Batch)` + `AbortReq`.  Running-side aborts apply `to_finish` on the next iteration via `apply_pending_aborts` in `event_loop.rs`. |
| Profiling, health checks, sleep mode   | ❌    | Lower priority; do after the hot loop is functional.                                         |

Legend: ✅ ported · 🟡 partial / skeleton · ❌ TODO (Python is authoritative)

## Layout

```text
rust/sglang-scheduler/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── src/
│   ├── lib.rs                  # pyo3 module entry
│   ├── main.rs                 # CLI entry — connects, does handshake, runs the loop
│   ├── radix_cache.rs          # Stub; placeholder for the radix port
│   ├── wire/                   # msgpack request/response structs
│   │   ├── mod.rs              # Tagged-union `Wire` enum
│   │   ├── tensor_ipc.rs       # TensorIPC
│   │   ├── handshake.rs        # GPUWorkerHandshakeReq{Input,Output}
│   │   ├── decode.rs           # DecodeStepControlReq + DeferredAllocIPC + DecodeForwardSlimOutput
│   │   ├── forward.rs          # ForwardBatch{Generation,Embedding,SplitPrefill}Req
│   │   ├── lora.rs             # LoRA admin Req/Output
│   │   ├── weights.rs          # Weight-update Req/Output pairs
│   │   └── mem_usage.rs        # GetMemUsageReq{Input,Output}
│   ├── transport/
│   │   ├── mod.rs
│   │   └── zmq_pair.rs         # sock_send / sock_recv equivalents (0xSG02 magic)
│   ├── memory/
│   │   ├── mod.rs
│   │   └── page_tracker.rs     # CpuPageTracker port
│   ├── types/
│   │   ├── mod.rs
│   │   ├── forward_mode.rs     # ForwardMode enum
│   │   ├── req.rs              # Req + FinishReason
│   │   ├── sampling.rs         # SamplingParams subset
│   │   ├── batch.rs            # ScheduleBatch + ModelWorkerBatchView
│   │   └── server_args.rs      # ServerArgs subset
│   ├── queue/
│   │   ├── mod.rs
│   │   ├── policy.rs           # FCFS / LoF / LongestPrefix / Priority
│   │   └── waiting.rs          # WaitingQueue with policy-ordered drain
│   └── scheduler/
│       ├── mod.rs
│       ├── config.rs           # Scheduler-side runtime config
│       ├── worker_client.rs    # TpWorkerClient equivalent; typed per-RPC wrappers
│       ├── batch_builder.rs    # prepare_for_decode + cache-aware prefill admission
│       ├── output_processor.rs # process_batch_result + cache insert + dec_lock_ref
│       ├── request_source.rs   # drain_into → admit Tokenized*ReqInput / AbortReq
│       ├── retract.rs          # retract_decode + evict_from_tree_cache + reclaim_for_decode
│       ├── dp_attn.rs          # single-DP local-fallback all_gather equivalent
│       ├── session_controller.rs # open/close session bookkeeping
│       └── event_loop.rs       # Main loop wiring all of the above
└── tests/
    ├── wire_roundtrip.rs       # rmp-serde + tagged-union sanity checks
    ├── queue_and_req.rs        # Req::check_finished + policy ordering + queue drain
    ├── scheduler_smoke.rs      # radix + retract + reclaim integration
    ├── model_worker_batch.rs   # ModelWorkerBatchPayload encoding
    └── tokenizer_source.rs     # tokenizer PUSH → scheduler PULL roundtrip
```

## Wire compatibility

Each `msgspec.Struct, tag=True` in `python/sglang/srt/managers/io_struct/msgpack_struct.py`
serialises as a msgpack fixmap whose first entry is `"type": "<ClassName>"` followed by
field name → value pairs (with the field order from the Python class). Rust mirrors
declare matching `#[derive(Serialize, Deserialize)]` structs with `#[serde(tag = "type")]`
internally tagged enums OR explicit `"type"` fields — the chosen approach is consistent
within each module; see the comments in `src/wire/`.

The magic-number framing on the wire is unchanged:
* `[b"0xSG02", msgpack_bytes]` for typed msgspec.Struct payloads.
* `[b"0xSG01", pickle_bytes]` for dataclass / pickled fallbacks — this crate currently
  rejects pickle frames since the Rust scheduler should only need msgpack-typed traffic.

### Model-config piggyback on the handshake

Instead of having the Rust scheduler load HuggingFace configs itself (which would
require pulling in `transformers`-equivalent shape derivation for every model
family), the Python GPU worker stamps a handful of scalars off its already-loaded
`model_runner.model_config` into the handshake reply:

| Field | Source | Used by Rust for |
|---|---|---|
| `vocab_size` | `model_config.vocab_size` | `SamplingBatchInfoPayload.vocab_size` + `logit_bias` scratch |
| `context_len` | `model_config.context_len` | truncating oversized prompts in `request_source::build_req` |
| `is_generation` | `model_config.is_generation` | gating embedding-mode admission (today: warn-and-drop) |
| `hf_eos_token_ids` | normalised list of `model_config.hf_eos_token_id` | default `stop_token_ids` for reqs that didn't ship any |
| `think_end_id` | `model_config.think_end_id` | reasoning models — currently logged only |

These are stamped into `GPUWorkerHandshakeReqOutput` (Python:
[`msgpack_struct.py`](../../python/sglang/srt/managers/io_struct/msgpack_struct.py),
Rust mirror: [`src/wire/handshake.rs`](src/wire/handshake.rs)) and surfaced inside
the scheduler via `WorkerSnapshot::from_handshake`.

**Backward compatibility**: every new field is defaulted (`vocab_size: int = 0`,
`hf_eos_token_ids: Optional[List[int]] = None`, …) and `#[serde(default)]` on
the Rust side, so:
* an *older* Python worker connecting to a *newer* Rust scheduler decodes the
  reply with `vocab_size = 0`, no EOS fallback, no context-len cap — exactly the
  pre-Option-B behaviour;
* a *newer* Python worker connecting to an *older* Rust scheduler has its extra
  fields ignored (serde's default is "discard unknown keys").

The `handshake_older_worker_decodes_with_defaults` test in
[`tests/wire_roundtrip.rs`](tests/wire_roundtrip.rs) pins this contract.

## Building

```bash
cd rust/sglang-scheduler
cargo build --release           # CLI binary at target/release/sglang-scheduler
maturin develop --release       # Python extension (sglang_scheduler) in current venv
```

## Running against a Python TpWorkerServer

The crate exposes two entry points — a standalone CLI and a Python extension
module — so you can drive the scheduler however the surrounding orchestration
prefers (same shape as the sibling `sglang-frontend` crate).

### From Python (preferred)

```python
from sglang_scheduler import SchedulerConfig, start_scheduler

# TP=1 (convenience: `worker_ipc=` instead of a one-element list)
cfg = SchedulerConfig(
    worker_ipc="ipc:///tmp/tp_worker_0.ipc",
    tokenizer_ipc="ipc:///tmp/scheduler_input.ipc",
)

# TP=N: one endpoint per rank, rank 0 is the leader
cfg = SchedulerConfig(
    worker_ipcs=[
        "ipc:///tmp/tp_worker_0.ipc",
        "ipc:///tmp/tp_worker_1.ipc",
        "ipc:///tmp/tp_worker_2.ipc",
        "ipc:///tmp/tp_worker_3.ipc",
    ],
    tokenizer_ipc="ipc:///tmp/scheduler_input.ipc",
)

start_scheduler(cfg)   # blocks until the loop exits; releases the GIL
```

`start_scheduler` drops the GIL via `py.detach` so the calling Python process
can keep doing other work (matching the `sglang_frontend.start_engine` pattern).

### From the CLI

```bash
# In each GPU process (one per TP rank, Python side, unchanged):
python -m sglang.srt.managers.tp_worker_server --ipc ipc:///tmp/tp_worker_0.ipc ...

# In the Rust scheduler process, list every rank's endpoint
# (repeat --worker-ipc per rank, or pass a comma list via SGLANG_WORKER_IPC):
sglang-scheduler \
    --worker-ipc ipc:///tmp/tp_worker_0.ipc \
    --worker-ipc ipc:///tmp/tp_worker_1.ipc \
    --tokenizer-ipc ipc:///tmp/scheduler_input.ipc

# Equivalent env-var form:
SGLANG_WORKER_IPC=ipc:///tmp/tp_worker_0.ipc,ipc:///tmp/tp_worker_1.ipc \
SGLANG_TOKENIZER_IPC=ipc:///tmp/scheduler_input.ipc \
    sglang-scheduler
```

Either entry point performs the handshake, binds the tokenizer PULL socket, and
then runs the prefill→decode→output-process loop driven by
`TokenizedGenerateReqInput` frames arriving on the tokenizer IPC.  Without a
tokenizer IPC the scheduler idles and exits after the first iteration
(handshake-only smoke mode).

### TP fan-out — `WorkerClientGroup`

For TP>1 every RPC fans out over a PAIR socket per rank.  The scheduler picks
one of two dispatch patterns per RPC type
([`worker_client.rs`](src/scheduler/worker_client.rs)):

* **broadcast_leader_only** — send to every rank, return the leader's typed
  reply, drain (and at most warn on) every other rank's reply.  Used for the
  forward path (`forward_batch_generation`, `decode_step`, …), `handshake`,
  `get_mem_usage`, and `get_weights_by_name` — RPCs where every worker
  produces an equivalent result and only the leader's is read.
* **broadcast_all_confirm** — send to every rank, collect every reply.  Used
  for admin RPCs (`load_lora_adapter`, `update_weights_from_*`,
  `init/destroy_weights_update_group`, …) where partial success would be a
  bug; the leader's reply is returned and rank disagreement is logged.

The PAIR-socket lockstep is load-bearing: skipping a non-leader drain on a
broadcast_leader_only RPC would leave a stale frame in that socket's buffer
that the next RPC's `recv` would mis-pick.  `tests/worker_group.rs` pins
this with a "consecutive broadcast leader-only" regression test.

Handshake parity: every rank's handshake reply is read and checked against the
leader's sizing fields (`max_total_num_tokens`, `vocab_size`, `context_len`,
…).  Disagreements are surfaced at `warn!` level — they mean the workers were
booted with mismatched configs and the leader's snapshot still drives the
loop.

## Remaining work

The hot loop is now end-to-end functional: tokenized requests in → typed `ModelWorkerBatch`
to the GPU worker → output processing → cache reuse / retraction → next iteration.  What's
still on the list, in rough order of leverage:

1. **Output-processor branches**: logprobs streaming, grammar / structured output integration,
   hidden-states return, speculative-decode accept counting.  See the TODOs in
   `output_processor.rs`.
2. **Multi-DP attention sync**: replace the local fallback in `dp_attn::prepare_local_dp_attention_sync`
   with a real cross-process `all_gather` (gloo/NCCL bridge) once the Rust scheduler runs
   in a multi-DP topology.
3. **Penalizer orchestrator**: the Python equivalent is currently disabled
   (`rainj-me` TODO); revive it once upstream re-enables the path.
4. **Top-level LoRA / weight-update handlers**: the wire types + worker-client wrappers
   exist; need a `process_input_requests` dispatcher that maps the admin RPCs to them.
5. **Profiling, health checks, sleep mode** — lower priority.
6. **`SessionReqNode` branching tree** — the current `SessionController` is flat
   open/close bookkeeping; the full parent/child tree from Python is a follow-up.

Every step keeps the wire contract stable, so the Python `TpWorkerServer` doesn't need to
change.
