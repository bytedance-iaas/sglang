// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Core scheduler data model — `Req`, `ForwardMode`, `SamplingParams`,
//! `ScheduleBatch`, `ModelWorkerBatch`, plus the `ServerArgs` subset the
//! scheduler reads from.
//!
//! All of these mirror Python types in:
//! * `python/sglang/srt/managers/schedule_batch.py`
//! * `python/sglang/srt/sampling/sampling_params.py`
//! * `python/sglang/srt/server_args.py`
//!
//! What's ported here is the **shape** the scheduler needs to track —
//! enough to walk the batch through the wire and the queue.  Semantic
//! behaviour (radix-cache integration, retraction, sampling penalties)
//! lives in the modules that own those concerns.

pub mod batch;
pub mod forward_mode;
pub mod req;
pub mod sampling;
pub mod server_args;

pub use batch::{ModelWorkerBatchView, ScheduleBatch};
pub use forward_mode::ForwardMode;
pub use req::{FinishReason, Req};
pub use sampling::SamplingParams;
pub use server_args::ServerArgs;
