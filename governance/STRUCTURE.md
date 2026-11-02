# Repository structure

Top-level layout with the responsibility of each directory. Module owners and
boundaries are authoritative in [manifest.yaml](manifest.yaml) (`modules`).

```
.
├── AGENTS.md                 Agent entry point (commands, boundaries, routing)
├── CLAUDE.md                 Imports AGENTS.md for Claude Code
├── Makefile                  Standard commands (bootstrap, lint, test, verify, agent-*)
├── README.md                 Human entry point
├── .claude/                  Agent rules (rules/) and task skills (skills/)
├── .github/                  CI workflows, PR template, CODEOWNERS, CI permissions
├── 3rdparty/                 Vendored third-party code (AMD); not a submodule
├── assets/                   Logos and images used by README/docs
├── benchmark/                Benchmarks and accuracy evaluations (gsm8k, mmlu, ...)
├── docker/                   Dockerfiles per platform, compose.yaml, k8s manifests
├── docs/                     Mintlify documentation site and cookbook
├── examples/                 Usage examples (runtime, frontend language, monitoring)
├── experimental/sgl-router/  Experimental router (Rust)
├── governance/               Manifest, status, ledger, policy, ADRs, agent change log
├── proto/                    Protobuf definitions (gRPC)
├── python/                   The `sglang` Python package (pyproject*.toml per platform)
│   └── sglang/
│       ├── srt/              Serving runtime
│       │   ├── entrypoints/      HTTP/gRPC/OpenAI-compatible servers, Engine
│       │   ├── managers/         Scheduler, tokenizer manager, detokenizer
│       │   ├── model_executor/   ModelRunner, forward batch, CUDA graphs
│       │   ├── mem_cache/        Radix/hierarchical KV cache and memory pools
│       │   ├── models/           Model implementations
│       │   ├── layers/           Attention backends, MoE, quantization, sampling
│       │   ├── speculative/      Speculative decoding
│       │   ├── disaggregation/   Prefill/decode disaggregation
│       │   ├── distributed/      Parallel groups and communicators
│       │   ├── server_args.py    CLI surface
│       │   └── environ.py        SGLANG_* environment variable registry
│       ├── kernels/          Kernel wrappers and JIT kernels
│       ├── multimodal_gen/   Diffusion / multimodal generation
│       ├── lang/             Frontend language (DSL)
│       └── test/             Shared test utilities and CI registration helpers
├── rust/                     Rust workspace (grpc, mm, radix tree, server); Cargo.lock pinned
├── scripts/                  CI, release, lint (scripts/lint) and agent (scripts/agent) scripts
├── sgl-model-gateway/        Model gateway / router (Rust)
├── test/                     CI tests: registered/ (auto-discovered), manual/, run_suite.py
└── tools/                    Standalone tools (sglang-simulator)
```

Submodules: none.
