// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Minimal config for the scheduler.  This is the subset of `ServerArgs`
//! + `PortArgs` the Rust scheduler currently consumes; grow it as more
//! Python paths get ported.
//!
//! See `python/sglang/srt/server_args.py` for the full Python schema.
//!
//! Mirrors the sibling `sglang-frontend` pattern: the struct doubles as
//! a `#[pyclass]` so Python can build it with a keyword constructor and
//! pass it straight into `start_scheduler`.
//!
//! ## Field minimalism
//!
//! Anything that *would* be config in the Python scheduler but isn't yet
//! consumed by any Rust code path lives on the relevant feature module
//! once it ports, not here.  Past dead-config fields that lived in this
//! struct purely for parity (`tp_rank`, `pp_rank`, `pipeline_2ahead`,
//! `recv_timeout_ms`) have been removed — they had zero consumers and
//! would reappear as constructor arguments to the module that actually
//! uses them (e.g. multi-DP sync, the pipelined event loop) when those
//! port.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(from_py_object, module = "sglang_scheduler")]
#[derive(Debug, Clone, Default)]
pub struct SchedulerConfig {
    /// One ZMQ PAIR endpoint per TP rank.  Rank 0 is the leader (see
    /// `WorkerClientGroup`).  TP=1 callers pass a one-element list, or
    /// equivalently use the `worker_ipc=` keyword in the Python ctor.
    ///
    /// The Rust scheduler doesn't derive endpoints from TP ranks the
    /// way Python does — the caller is expected to pass the fully-
    /// formed ipc:// or tcp:// addresses for every worker process.
    #[pyo3(get, set)]
    pub worker_ipcs: Vec<String>,

    /// ZMQ endpoint the scheduler PULLs (connects to) for tokenized
    /// requests from the tokenizer manager.  Mirrors Python's
    /// `port_args.scheduler_input_ipc_name`; the tokenizer manager
    /// binds it as PUSH and the scheduler connects.  Empty disables
    /// the request source (useful for handshake-only smoke tests).
    #[pyo3(get, set)]
    pub tokenizer_ipc: String,

    /// ZMQ endpoint the scheduler PUSHes (connects to) for output
    /// frames sent to the detokenizer manager.  Mirrors Python's
    /// `port_args.detokenizer_ipc_name`; the detokenizer binds it as
    /// PULL.  Empty disables the output emission (handshake-only
    /// smoke mode also leaves it empty).
    #[pyo3(get, set)]
    pub detokenizer_ipc: String,
}

#[pymethods]
impl SchedulerConfig {
    /// Construct a SchedulerConfig.
    ///
    /// Exactly one of `worker_ipc` (single rank, TP=1 convenience) or
    /// `worker_ipcs` (list, one entry per TP rank) must be provided.
    /// Both at once raises `ValueError`.
    #[new]
    #[pyo3(signature = (
        worker_ipc = None,
        worker_ipcs = None,
        tokenizer_ipc = String::new(),
        detokenizer_ipc = String::new(),
    ))]
    fn new(
        worker_ipc: Option<String>,
        worker_ipcs: Option<Vec<String>>,
        tokenizer_ipc: String,
        detokenizer_ipc: String,
    ) -> PyResult<Self> {
        let ipcs = match (worker_ipc, worker_ipcs) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "pass exactly one of worker_ipc=<str> or worker_ipcs=<list[str]>, not both",
                ));
            }
            (Some(single), None) => vec![single],
            (None, Some(list)) => list,
            (None, None) => {
                return Err(PyValueError::new_err(
                    "SchedulerConfig requires either worker_ipc=<str> or worker_ipcs=<list[str]>",
                ));
            }
        };
        if ipcs.is_empty() {
            return Err(PyValueError::new_err(
                "worker_ipcs must contain at least one endpoint",
            ));
        }
        Ok(Self {
            worker_ipcs: ipcs,
            tokenizer_ipc,
            detokenizer_ipc,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SchedulerConfig(worker_ipcs={:?}, tokenizer_ipc={:?}, detokenizer_ipc={:?})",
            self.worker_ipcs, self.tokenizer_ipc, self.detokenizer_ipc,
        )
    }
}
