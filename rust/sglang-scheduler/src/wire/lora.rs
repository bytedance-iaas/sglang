// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! LoRA adapter admin wire types.
//!
//! Source: `msgpack_struct.py` (`LoadLoRAAdapterReqInput`,
//! `UnloadLoRAAdapterReqInput`, `LoadLoRAAdapterFromTensorsReqInput`)
//! and the shared `LoRAUpdateOutput` reply type.

use serde::{Deserialize, Serialize};

use crate::wire::tensor_ipc::TensorIPC;

/// Shared output for all three LoRA admin RPCs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAUpdateOutput {
    pub success: bool,
    pub error_message: Option<String>,
    pub loaded_adapters: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadLoRAAdapterReqInput {
    pub lora_name: String,
    pub lora_path: String,
    #[serde(default)]
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnloadLoRAAdapterReqInput {
    pub lora_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadLoRAAdapterFromTensorsReqInput {
    pub lora_name: String,
    /// Map of weight name → tensor blob.  Opaque on the Rust side.
    pub tensors: Vec<(String, TensorIPC)>,
    #[serde(default)]
    pub pinned: bool,
}
