// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! CPU-side KV memory bookkeeping.  Mirrors the data-only helpers from
//! `python/sglang/srt/managers/scheduler_cpu.py`.

pub mod page_tracker;
pub mod req_to_token_pool;

pub use page_tracker::CpuPageTracker;
pub use req_to_token_pool::{ReqToTokenPool, ReqToTokenPoolError};
