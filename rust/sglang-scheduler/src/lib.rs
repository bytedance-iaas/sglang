// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! `sglang-scheduler` — Rust port of SGLang's CPU-only scheduler.
//!
//! See `README.md` in this crate's root for the porting status and the
//! correspondence between Rust modules and the Python source files
//! they're derived from.
//!
//! Public modules:
//! * [`wire`] — msgspec.Struct mirrors (the IPC wire schema).
//! * [`transport`] — ZMQ PAIR + magic-number framing.
//! * [`memory`] — CPU-side KV bookkeeping (`CpuPageTracker`).
//! * [`scheduler`] — config, worker client, event loop.
//!
//! ## Python entry point
//!
//! The crate exposes a `sglang_scheduler` Python extension module
//! (same pattern as the sibling `sglang-frontend` crate):
//!
//! ```python
//! from sglang_scheduler import SchedulerConfig, start_scheduler
//!
//! cfg = SchedulerConfig(
//!     worker_ipc="ipc:///tmp/tp_worker.ipc",
//!     tokenizer_ipc="ipc:///tmp/scheduler_input.ipc",
//! )
//! start_scheduler(cfg)   # blocks until the loop exits
//! ```
//!
//! `start_scheduler` releases the GIL while the event loop runs so the
//! launching Python process can keep doing other work.

pub mod memory;
pub mod queue;
pub mod radix_cache;
pub mod scheduler;
pub mod transport;
pub mod types;
pub mod wire;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Run the scheduler event loop under the supplied config.
///
/// Drops the GIL via `py.detach` so the calling Python thread doesn't
/// block other Python work (sibling pattern: `sglang_frontend.start_engine`).
/// Blocks until the loop exits — either because the tokenizer IPC was
/// not set (handshake-only smoke mode) or because the worker socket
/// failed.
#[pyfunction]
fn start_scheduler(py: Python<'_>, config: scheduler::SchedulerConfig) -> PyResult<()> {
    // Initialise env_logger once on the first call.  Subsequent calls
    // re-use the same global logger.
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    py.detach(|| scheduler::run_event_loop(&config))
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err}")))
}

/// One-shot handshake against the first endpoint in `worker_ipcs`.
///
/// Opens a temporary PAIR connection to rank 0, sends a
/// `GPUWorkerHandshakeReqInput`, reads the reply, and returns the
/// scheduler-relevant fields as a Python dict.  The connection is
/// dropped before returning so a subsequent `start_scheduler` call can
/// reconnect cleanly.
///
/// Intended for Python launchers that need to forward `max_total_num_tokens`
/// / `max_req_input_len` to their parent process via a multiprocessing
/// pipe *before* the long-running event loop starts blocking.  See
/// `_run_rust_scheduler_process` in `sglang/srt/entrypoints/http_server.py`.
#[pyfunction]
fn peek_worker_handshake<'py>(
    py: Python<'py>,
    worker_ipcs: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    use scheduler::WorkerClientGroup;

    if worker_ipcs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "peek_worker_handshake requires at least one worker_ipc",
        ));
    }

    // Connect, handshake, drop.  Done under `py.detach` because the ZMQ
    // recv blocks until the worker replies.
    let snapshot = py
        .detach(|| {
            let ctx = zmq::Context::new();
            let group = WorkerClientGroup::connect_all(&ctx, &worker_ipcs)?;
            // Clone the leader handshake before dropping the group so
            // the sockets close before we return.
            Ok::<_, scheduler::WorkerClientError>(group.handshake().clone())
        })
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err}")))?;

    let dict = PyDict::new(py);
    dict.set_item("max_total_num_tokens", snapshot.max_total_num_tokens)?;
    dict.set_item("max_prefill_tokens", snapshot.max_prefill_tokens)?;
    dict.set_item("max_running_requests", snapshot.max_running_requests)?;
    dict.set_item("max_req_len", snapshot.max_req_len)?;
    dict.set_item("max_req_input_len", snapshot.max_req_input_len)?;
    dict.set_item("device", snapshot.device.clone())?;
    dict.set_item("req_to_token_pool_size", snapshot.req_to_token_pool_size)?;
    dict.set_item(
        "req_to_token_pool_max_context_len",
        snapshot.req_to_token_pool_max_context_len,
    )?;
    dict.set_item("token_to_kv_pool_size", snapshot.token_to_kv_pool_size)?;
    dict.set_item("vocab_size", snapshot.vocab_size)?;
    dict.set_item("context_len", snapshot.context_len)?;
    dict.set_item("is_generation", snapshot.is_generation)?;
    Ok(dict)
}

#[pymodule]
fn sglang_scheduler(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<scheduler::SchedulerConfig>()?;
    m.add_function(wrap_pyfunction!(start_scheduler, m)?)?;
    m.add_function(wrap_pyfunction!(peek_worker_handshake, m)?)?;
    Ok(())
}
