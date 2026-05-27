// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Weight-update admin wire types.
//!
//! Source: `msgpack_struct.py` for each `Update*ReqInput` /
//! `Update*ReqOutput` pair plus `GetWeightsByName*`.  All updates share
//! the same `(success, message)` reply shape.

use serde::{Deserialize, Serialize};

use crate::wire::tensor_ipc::TensorIPC;

/// Common reply shape across weight-update RPCs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightOpOutput {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateWeightFromDiskReqInput {
    pub model_path: String,
    pub load_format: String,
}
pub type UpdateWeightFromDiskReqOutput = WeightOpOutput;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitWeightsUpdateGroupReqInput {
    pub master_address: String,
    pub master_port: i64,
    pub rank_offset: i64,
    pub world_size: i64,
    pub group_name: String,
    pub backend: String,
}
pub type InitWeightsUpdateGroupReqOutput = WeightOpOutput;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DestroyWeightsUpdateGroupReqInput {
    pub group_name: String,
}
pub type DestroyWeightsUpdateGroupReqOutput = WeightOpOutput;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitWeightsSendGroupForRemoteInstanceReqInput {
    pub master_address: String,
    pub master_port: i64,
    pub rank_offset: i64,
    pub world_size: i64,
    pub group_name: String,
    pub backend: String,
}
pub type InitWeightsSendGroupForRemoteInstanceReqOutput = WeightOpOutput;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendWeightsToRemoteInstanceReqInput {
    pub group_name: String,
    pub dst_rank: i64,
}
pub type SendWeightsToRemoteInstanceReqOutput = WeightOpOutput;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateWeightsFromDistributedReqInput {
    pub names: Vec<String>,
    pub dtypes: Vec<String>,
    pub shapes: Vec<Vec<i64>>,
    pub group_name: String,
    pub flush_cache: bool,
}
pub type UpdateWeightsFromDistributedReqOutput = WeightOpOutput;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateWeightsFromTensorReqInput {
    pub named_tensors: Vec<(String, TensorIPC)>,
    pub flush_cache: bool,
}
pub type UpdateWeightsFromTensorReqOutput = WeightOpOutput;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateWeightsFromIPCReqInput {
    pub handles: Vec<serde_json::Value>,
    pub flush_cache: bool,
}
pub type UpdateWeightsFromIPCReqOutput = WeightOpOutput;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetWeightsByNameReqInput {
    pub name: String,
    #[serde(default)]
    pub truncate_size: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetWeightsByNameReqOutput {
    pub parameter: Option<TensorIPC>,
}
