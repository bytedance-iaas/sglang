// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! CPU-only scheduler — Rust port of
//! `python/sglang/srt/managers/scheduler_cpu.py` + the bulk of
//! `scheduler.py` it inherits from.
//!
//! The current cut establishes the wire contract and the worker handshake
//! plumbing; the actual scheduling policy (request queue, batching,
//! retraction, radix cache) is stubbed.  Each stub has a `TODO(rust-port)`
//! tag pointing at the Python source of truth.

pub mod batch_builder;
pub mod config;
pub mod dp_attn;
pub mod event_loop;
pub mod metrics;
pub mod output_processor;
pub mod request_source;
pub mod retract;
pub mod session_controller;
pub mod worker_client;
pub mod worker_snapshot;

pub use batch_builder::BatchBuilder;
pub use config::SchedulerConfig;
pub use dp_attn::{DpSync, prepare_local_dp_attention_sync};
pub use event_loop::run_event_loop;
pub use metrics::SchedulerMetrics;
pub use output_processor::{StepStats, process_batch_result, process_batch_result_with_cache};
pub use request_source::{RecvOutcome, drain_into};
pub use retract::{
    RetractionOutcome, evict_from_tree_cache, reclaim_for_decode, retract_decode,
    retract_decode_with_cache,
};
pub use session_controller::{Session, SessionController};
pub use worker_client::{WorkerClient, WorkerClientError, WorkerClientGroup};
pub use worker_snapshot::WorkerSnapshot;
