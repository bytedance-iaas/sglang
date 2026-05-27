// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Request queue + admission.
//!
//! Source: `Scheduler.waiting_queue` + `process_input_requests` +
//! `policies/` in `python/sglang/srt/managers/`.

pub mod policy;
pub mod waiting;

pub use policy::{SchedulePolicy, SchedulePolicyKind};
pub use waiting::WaitingQueue;
