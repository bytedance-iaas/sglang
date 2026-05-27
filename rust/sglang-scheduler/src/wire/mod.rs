// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0
//
//! Wire-format types for the Rust scheduler ↔ Python `TpWorkerServer` IPC.
//!
//! Every message on the wire is a msgspec-encoded fixmap with `"type"`
//! as the discriminator (Python: `class Foo(msgspec.Struct, tag=True)`).
//! The Rust mirror uses serde's internally-tagged enum representation,
//! which `rmp-serde` encodes to the same map shape.
//!
//! Wire schema source of truth:
//! `python/sglang/srt/managers/io_struct/msgpack_struct.py`.
//!
//! Tensors ride the wire as the 3-tuple `(shape, dtype_str, raw_bytes)` —
//! see `enc_hook` / `dec_hook` in the Python file.  We don't model torch
//! semantics on the Rust side; tensors are opaque `TensorIPC` blobs that
//! the scheduler routes around without inspecting (except for KV-page
//! bookkeeping fields, which we promote to typed integer arrays).

pub mod decode;
pub mod detokenizer;
pub mod forward;
pub mod handshake;
pub mod lora;
pub mod mem_usage;
pub mod model_worker_batch;
pub mod tensor_ipc;
pub mod tokenizer;
pub mod weights;

use serde::{Deserialize, Serialize};

pub use decode::{DecodeForwardSlimOutput, DecodeStepControlReq, DeferredAllocIPC};
pub use detokenizer::BatchTokenIDOutput;
pub use forward::{
    ForwardBatchEmbeddingReq, ForwardBatchGenerationReq, ForwardBatchSplitPrefillReq,
};
pub use handshake::{GPUWorkerHandshakeReqInput, GPUWorkerHandshakeReqOutput};
pub use lora::{
    LoRAUpdateOutput, LoadLoRAAdapterFromTensorsReqInput, LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
};
pub use mem_usage::{GetMemUsageReqInput, GetMemUsageReqOutput};
pub use model_worker_batch::{
    capture_hidden_mode_wire, forward_mode_wire, ModelWorkerBatchPayload,
    SamplingBatchInfoPayload, SamplingParamsView,
};
pub use tensor_ipc::TensorIPC;
pub use tokenizer::{
    AbortReq, BatchTokenizedEmbeddingReqInput, BatchTokenizedGenerateReqInput,
    SamplingParamsIPC, TokenizedEmbeddingReqInput, TokenizedGenerateReqInput,
};
pub use weights::{
    DestroyWeightsUpdateGroupReqInput, DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput, GetWeightsByNameReqOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput, InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput, SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput, UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput, UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput, UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput, UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput, WeightOpOutput,
};

/// Discriminated union of every wire-level request/response the Rust scheduler
/// may produce or consume. The variant name MUST match the Python class name
/// (msgspec's tag value is `cls.__name__` by default), so don't rename without
/// the matching change on the Python side.
///
/// Add new variants here as you port more handlers — `sock_recv` returns
/// `Wire`, and the scheduler matches on the variant to route.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Wire {
    // Handshake.
    GPUWorkerHandshakeReqInput(GPUWorkerHandshakeReqInput),
    GPUWorkerHandshakeReqOutput(GPUWorkerHandshakeReqOutput),

    // Decode hot path.
    DecodeStepControlReq(DecodeStepControlReq),
    DecodeForwardSlimOutput(DecodeForwardSlimOutput),

    // Full forward paths.
    ForwardBatchGenerationReq(ForwardBatchGenerationReq),
    ForwardBatchEmbeddingReq(ForwardBatchEmbeddingReq),
    ForwardBatchSplitPrefillReq(ForwardBatchSplitPrefillReq),

    // LoRA admin.
    LoadLoRAAdapterReqInput(LoadLoRAAdapterReqInput),
    UnloadLoRAAdapterReqInput(UnloadLoRAAdapterReqInput),
    LoadLoRAAdapterFromTensorsReqInput(LoadLoRAAdapterFromTensorsReqInput),
    LoRAUpdateOutput(LoRAUpdateOutput),

    // Weight updates.
    UpdateWeightFromDiskReqInput(UpdateWeightFromDiskReqInput),
    UpdateWeightFromDiskReqOutput(UpdateWeightFromDiskReqOutput),
    InitWeightsUpdateGroupReqInput(InitWeightsUpdateGroupReqInput),
    InitWeightsUpdateGroupReqOutput(InitWeightsUpdateGroupReqOutput),
    DestroyWeightsUpdateGroupReqInput(DestroyWeightsUpdateGroupReqInput),
    DestroyWeightsUpdateGroupReqOutput(DestroyWeightsUpdateGroupReqOutput),
    InitWeightsSendGroupForRemoteInstanceReqInput(
        InitWeightsSendGroupForRemoteInstanceReqInput,
    ),
    InitWeightsSendGroupForRemoteInstanceReqOutput(
        InitWeightsSendGroupForRemoteInstanceReqOutput,
    ),
    SendWeightsToRemoteInstanceReqInput(SendWeightsToRemoteInstanceReqInput),
    SendWeightsToRemoteInstanceReqOutput(SendWeightsToRemoteInstanceReqOutput),
    UpdateWeightsFromDistributedReqInput(UpdateWeightsFromDistributedReqInput),
    UpdateWeightsFromDistributedReqOutput(UpdateWeightsFromDistributedReqOutput),
    UpdateWeightsFromTensorReqInput(UpdateWeightsFromTensorReqInput),
    UpdateWeightsFromTensorReqOutput(UpdateWeightsFromTensorReqOutput),
    UpdateWeightsFromIPCReqInput(UpdateWeightsFromIPCReqInput),
    UpdateWeightsFromIPCReqOutput(UpdateWeightsFromIPCReqOutput),

    // Param introspection.
    GetWeightsByNameReqInput(GetWeightsByNameReqInput),
    GetWeightsByNameReqOutput(GetWeightsByNameReqOutput),

    // Memory usage.
    GetMemUsageReqInput(GetMemUsageReqInput),
    GetMemUsageReqOutput(GetMemUsageReqOutput),

    // Tokenizer → scheduler.
    TokenizedGenerateReqInput(TokenizedGenerateReqInput),
    BatchTokenizedGenerateReqInput(BatchTokenizedGenerateReqInput),
    TokenizedEmbeddingReqInput(TokenizedEmbeddingReqInput),
    BatchTokenizedEmbeddingReqInput(BatchTokenizedEmbeddingReqInput),
    AbortReq(AbortReq),

    // Scheduler → detokenizer.
    BatchTokenIDOutput(BatchTokenIDOutput),
}

impl Wire {
    /// Stable string name of the wire variant — used for logging and the
    /// `WorkerClient::rpc` mismatch error message.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Wire::GPUWorkerHandshakeReqInput(_) => "GPUWorkerHandshakeReqInput",
            Wire::GPUWorkerHandshakeReqOutput(_) => "GPUWorkerHandshakeReqOutput",
            Wire::DecodeStepControlReq(_) => "DecodeStepControlReq",
            Wire::DecodeForwardSlimOutput(_) => "DecodeForwardSlimOutput",
            Wire::ForwardBatchGenerationReq(_) => "ForwardBatchGenerationReq",
            Wire::ForwardBatchEmbeddingReq(_) => "ForwardBatchEmbeddingReq",
            Wire::ForwardBatchSplitPrefillReq(_) => "ForwardBatchSplitPrefillReq",
            Wire::LoadLoRAAdapterReqInput(_) => "LoadLoRAAdapterReqInput",
            Wire::UnloadLoRAAdapterReqInput(_) => "UnloadLoRAAdapterReqInput",
            Wire::LoadLoRAAdapterFromTensorsReqInput(_) => {
                "LoadLoRAAdapterFromTensorsReqInput"
            }
            Wire::LoRAUpdateOutput(_) => "LoRAUpdateOutput",
            Wire::UpdateWeightFromDiskReqInput(_) => "UpdateWeightFromDiskReqInput",
            Wire::UpdateWeightFromDiskReqOutput(_) => "UpdateWeightFromDiskReqOutput",
            Wire::InitWeightsUpdateGroupReqInput(_) => "InitWeightsUpdateGroupReqInput",
            Wire::InitWeightsUpdateGroupReqOutput(_) => "InitWeightsUpdateGroupReqOutput",
            Wire::DestroyWeightsUpdateGroupReqInput(_) => {
                "DestroyWeightsUpdateGroupReqInput"
            }
            Wire::DestroyWeightsUpdateGroupReqOutput(_) => {
                "DestroyWeightsUpdateGroupReqOutput"
            }
            Wire::InitWeightsSendGroupForRemoteInstanceReqInput(_) => {
                "InitWeightsSendGroupForRemoteInstanceReqInput"
            }
            Wire::InitWeightsSendGroupForRemoteInstanceReqOutput(_) => {
                "InitWeightsSendGroupForRemoteInstanceReqOutput"
            }
            Wire::SendWeightsToRemoteInstanceReqInput(_) => {
                "SendWeightsToRemoteInstanceReqInput"
            }
            Wire::SendWeightsToRemoteInstanceReqOutput(_) => {
                "SendWeightsToRemoteInstanceReqOutput"
            }
            Wire::UpdateWeightsFromDistributedReqInput(_) => {
                "UpdateWeightsFromDistributedReqInput"
            }
            Wire::UpdateWeightsFromDistributedReqOutput(_) => {
                "UpdateWeightsFromDistributedReqOutput"
            }
            Wire::UpdateWeightsFromTensorReqInput(_) => "UpdateWeightsFromTensorReqInput",
            Wire::UpdateWeightsFromTensorReqOutput(_) => "UpdateWeightsFromTensorReqOutput",
            Wire::UpdateWeightsFromIPCReqInput(_) => "UpdateWeightsFromIPCReqInput",
            Wire::UpdateWeightsFromIPCReqOutput(_) => "UpdateWeightsFromIPCReqOutput",
            Wire::GetWeightsByNameReqInput(_) => "GetWeightsByNameReqInput",
            Wire::GetWeightsByNameReqOutput(_) => "GetWeightsByNameReqOutput",
            Wire::GetMemUsageReqInput(_) => "GetMemUsageReqInput",
            Wire::GetMemUsageReqOutput(_) => "GetMemUsageReqOutput",
            Wire::TokenizedGenerateReqInput(_) => "TokenizedGenerateReqInput",
            Wire::BatchTokenizedGenerateReqInput(_) => "BatchTokenizedGenerateReqInput",
            Wire::TokenizedEmbeddingReqInput(_) => "TokenizedEmbeddingReqInput",
            Wire::BatchTokenizedEmbeddingReqInput(_) => "BatchTokenizedEmbeddingReqInput",
            Wire::AbortReq(_) => "AbortReq",
            Wire::BatchTokenIDOutput(_) => "BatchTokenIDOutput",
        }
    }
}
