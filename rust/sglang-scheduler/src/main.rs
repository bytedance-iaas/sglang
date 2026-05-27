// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! CLI entry point — connects to a running `TpWorkerServer`, performs the
//! handshake, and drives the scheduler event loop.
//!
//! For Python launching see [`lib.rs`] (`sglang_scheduler.start_scheduler`).
//! The CLI is the standalone-binary path; both end up calling
//! `run_event_loop` with the same `SchedulerConfig`.

use clap::Parser;

use sglang_scheduler::scheduler::{run_event_loop, SchedulerConfig};

/// Rust port of SGLang's CPU-only scheduler.
#[derive(Debug, Parser)]
#[command(name = "sglang-scheduler", version, about)]
struct Args {
    /// ZMQ PAIR endpoint for one TP rank's GPU worker.  Repeat the flag
    /// once per rank; rank 0 is the leader.  Equivalent: pass a single
    /// comma-separated list via `SGLANG_WORKER_IPC`.
    ///
    /// Examples:
    ///   --worker-ipc ipc:///tmp/tp_worker_0.ipc
    ///   --worker-ipc ipc:///tmp/tp_worker_0.ipc --worker-ipc ipc:///tmp/tp_worker_1.ipc
    ///   SGLANG_WORKER_IPC=ipc:///tmp/w0.ipc,ipc:///tmp/w1.ipc sglang-scheduler
    #[arg(
        long,
        env = "SGLANG_WORKER_IPC",
        value_delimiter = ',',
        num_args = 1..,
        required = true,
    )]
    worker_ipc: Vec<String>,

    /// ZMQ endpoint the scheduler PULL-connects to for tokenized
    /// requests (the tokenizer manager binds it as PUSH).  Matches
    /// Python's `port_args.scheduler_input_ipc_name`.  Empty disables
    /// the request source (useful for handshake-only smoke runs).
    #[arg(long, env = "SGLANG_TOKENIZER_IPC", default_value = "")]
    tokenizer_ipc: String,

    /// ZMQ endpoint the scheduler PUSH-connects to for output frames
    /// (the detokenizer manager binds it as PULL).  Matches Python's
    /// `port_args.detokenizer_ipc_name`.  Empty disables output
    /// emission (the scheduler still runs and consumes tokens).
    #[arg(long, env = "SGLANG_DETOKENIZER_IPC", default_value = "")]
    detokenizer_ipc: String,

    /// Disable the radix prefix cache — fall back to ChunkedCache
    /// semantics (no prefix matching, no cross-req KV sharing).
    /// Mirrors Python's `--disable-radix-cache`.
    #[arg(long, env = "SGLANG_DISABLE_RADIX_CACHE", default_value_t = false)]
    disable_radix_cache: bool,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let cfg = SchedulerConfig {
        worker_ipcs: args.worker_ipc,
        tokenizer_ipc: args.tokenizer_ipc,
        detokenizer_ipc: args.detokenizer_ipc,
        disable_radix_cache: args.disable_radix_cache,
    };

    if let Err(err) = run_event_loop(&cfg) {
        log::error!("scheduler exited with error: {err}");
        std::process::exit(1);
    }
}
