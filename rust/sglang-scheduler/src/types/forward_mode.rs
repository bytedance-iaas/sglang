// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! `ForwardMode` — mirrors the enum of the same name in
//! `python/sglang/srt/model_executor/forward_batch_info.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForwardMode {
    /// Prefill (any length, possibly multi-req).
    Extend,
    /// Single-token-per-req decode.
    Decode,
    /// Mixed prefill + decode in one batch.
    Mixed,
    /// No-op batch — used when DP sync requires all ranks to step even
    /// though this rank has nothing to do.
    Idle,
    /// Replayed-from-cache prefill (chunked-prefill resume).
    Prebuilt,
    /// Split-prefill phase (multi-stage prefill).
    SplitPrefill,
    /// Spec-decoding target verify.
    TargetVerify,
    /// Spec-decoding draft extend.
    DraftExtend,
}

impl ForwardMode {
    pub fn is_decode(self) -> bool {
        matches!(self, ForwardMode::Decode)
    }

    pub fn is_extend(self) -> bool {
        matches!(self, ForwardMode::Extend | ForwardMode::Mixed)
    }

    pub fn is_idle(self) -> bool {
        matches!(self, ForwardMode::Idle)
    }

    pub fn is_prebuilt(self) -> bool {
        matches!(self, ForwardMode::Prebuilt)
    }

    pub fn is_split_prefill(self) -> bool {
        matches!(self, ForwardMode::SplitPrefill)
    }
}
