// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Memory-usage query wire types.  Source: `GetMemUsageReqInput` /
//! `GetMemUsageReqOutput` in `msgpack_struct.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GetMemUsageReqInput {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetMemUsageReqOutput {
    pub weight_load_mem_usage: f64,
    pub graph_mem_usage: f64,
}
