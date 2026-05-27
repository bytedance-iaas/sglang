// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Scheduler main loop.
//!
//! Source: `CpuScheduler.event_loop_normal` / `event_loop_pipelined` in
//! `scheduler_cpu.py`, plus the `Scheduler` base class's
//! `event_loop_normal` it inherits.
//!
//! The loop assembles a `ScheduleBatch` per iteration, sends it through
//! the worker client, and routes the reply through the output processor.
//!
//! ## Wiring status
//!
//! * Worker connection + handshake + page-tracker bootstrap: ✅
//! * Iteration loop with batch building + output processing: ✅
//! * Request source (tokenizer → scheduler ZMQ socket): ❌ stub —
//!   the loop exits cleanly when the waiting queue is empty so the
//!   binary doesn't busy-spin against no work.
//! * `ForwardBatchGenerationReq.batch` field serialisation: ✅
//!   `ScheduleBatch::to_model_worker_batch_payload` builds the wire
//!   shape `ModelWorkerBatch` expects.  The payload covers every field
//!   msgspec will try to decode; feature-specific fields the Rust
//!   scheduler hasn't ported (multimodal, encoder-decoder, spec, …) ride
//!   out as the same `None` / default values the Python class uses when
//!   no one sets them.

use std::collections::HashSet;

use crate::memory::{CpuPageTracker, ReqToTokenPool};
use crate::queue::{SchedulePolicy, SchedulePolicyKind, WaitingQueue};
use crate::radix_cache::RadixCache;
use crate::scheduler::batch_builder::BatchBuilder;
use crate::scheduler::metrics::SchedulerMetrics;
use crate::scheduler::output_processor::process_batch_result_with_cache;
use crate::scheduler::request_source::drain_into;
use crate::scheduler::retract::reclaim_for_decode;
use crate::scheduler::worker_snapshot::WorkerSnapshot;
use crate::scheduler::{SchedulerConfig, WorkerClientError, WorkerClientGroup};
use crate::transport::{PullSource, PushSink, TransportError};
use crate::types::{FinishReason, ForwardMode, ScheduleBatch, ServerArgs};
use crate::wire::{ForwardBatchGenerationReq, Wire};

#[derive(Debug, thiserror::Error)]
pub enum EventLoopError {
    #[error(transparent)]
    WorkerClient(#[from] WorkerClientError),

    #[error(transparent)]
    Transport(#[from] TransportError),

    #[error("failed to encode ModelWorkerBatch payload: {0}")]
    EncodeBatch(#[from] rmp_serde::encode::Error),
}

pub fn run_event_loop(cfg: &SchedulerConfig) -> Result<(), EventLoopError> {
    // Start from defaults; overlay the SchedulerConfig + env vars.
    let mut server_args = ServerArgs::default();
    // ChunkedCache toggle — either field on SchedulerConfig (Python
    // entry point) or `SGLANG_DISABLE_RADIX_CACHE=1` env var (CLI).
    if cfg.disable_radix_cache
        || std::env::var("SGLANG_DISABLE_RADIX_CACHE")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
    {
        server_args.disable_radix_cache = true;
    }
    let policy = SchedulePolicy::new(SchedulePolicyKind::parse(&server_args.schedule_policy));

    let ctx = zmq::Context::new();
    let workers = WorkerClientGroup::connect_all(&ctx, &cfg.worker_ipcs)?;
    let hs = workers.handshake();
    let snapshot = WorkerSnapshot::from_handshake(hs);

    log::info!(
        "Worker model_config: vocab_size={}, context_len={}, is_generation={}, eos_tokens={:?}, think_end_id={:?}",
        snapshot.vocab_size,
        snapshot.context_len,
        snapshot.is_generation,
        snapshot.default_stop_token_ids,
        snapshot.think_end_id,
    );

    let page_size = server_args.page_size as i64;
    let total_pages = hs.token_to_kv_pool_size / page_size;
    let mut page_tracker = CpuPageTracker::new(total_pages, page_size);

    let mut req_pool = ReqToTokenPool::new(
        hs.req_to_token_pool_size as u32,
        hs.req_to_token_pool_max_context_len as u32,
    );

    let builder = BatchBuilder::new(
        page_size,
        server_args.max_running_requests as usize,
        server_args.max_prefill_tokens as i64,
        policy,
    );

    let mut waiting = WaitingQueue::new();
    let mut running = ScheduleBatch::new(ForwardMode::Decode);
    let mut radix_cache = RadixCache::new();
    let mut metrics = SchedulerMetrics::new();

    if server_args.disable_radix_cache {
        log::info!(
            "disable_radix_cache=true → using ChunkedCache semantics \
             (no prefix matching, no cross-req KV sharing)"
        );
    } else {
        log::info!("Radix prefix cache enabled");
    }
    // Pending aborts: rids whose AbortReq arrived while they were
    // running.  Applied on the next loop iteration via
    // `apply_pending_aborts`.
    let mut pending_aborts: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Optional tokenizer-request source.  When configured we bind the
    // PULL socket and pump it on every iteration; when empty the
    // scheduler runs in handshake-only mode and exits after one no-op
    // pass.
    let tokenizer_source = if cfg.tokenizer_ipc.is_empty() {
        log::warn!(
            "no tokenizer IPC configured; scheduler will idle and exit \
             once the running batch drains.  Pass --tokenizer-ipc to \
             drive real traffic."
        );
        None
    } else {
        // The Python TokenizerManager binds the PUSH socket at
        // `scheduler_input_ipc_name`; the scheduler is the PULL
        // *connect* side (see `get_zmq_socket(..., bind=False)` in
        // `scheduler.py::__init__`).
        log::info!("Connecting tokenizer source: {}", cfg.tokenizer_ipc);
        Some(PullSource::connect(&ctx, &cfg.tokenizer_ipc)?)
    };

    // Detokenizer push sink — symmetric with the tokenizer source: the
    // detokenizer manager binds the PULL socket at
    // `detokenizer_ipc_name`, the scheduler PUSH-connects.  Empty
    // disables emission (scheduler still consumes tokens, just doesn't
    // stream them out — handy for benchmarking / smoke tests).
    let detokenizer_sink = if cfg.detokenizer_ipc.is_empty() {
        log::warn!(
            "no detokenizer IPC configured; scheduler will not emit \
             BatchTokenIDOutput frames.  Pass --detokenizer-ipc to \
             stream output tokens."
        );
        None
    } else {
        log::info!("Connecting detokenizer sink: {}", cfg.detokenizer_ipc);
        Some(PushSink::connect(&ctx, &cfg.detokenizer_ipc)?)
    };

    log::info!(
        "Scheduler initialised: device={}, total_pages={}, page_size={}, available={}",
        hs.device,
        page_tracker.total_pages(),
        page_tracker.page_size(),
        page_tracker.available_size(),
    );

    let mut iter = 0u64;
    let mut idle_iters_without_source = 0u64;

    loop {
        iter += 1;

        // 1. Pull new requests from the tokenizer (non-blocking drain).
        if let Some(source) = tokenizer_source.as_ref() {
            let running_rids: HashSet<String> = running
                .reqs
                .iter()
                .map(|r| r.read().unwrap().rid.clone())
                .collect();
            match drain_into(source, &mut waiting, &running_rids, &snapshot) {
                Ok(outcome) => {
                    if outcome.admitted + outcome.aborted + outcome.dropped_unsupported > 0 {
                        log::debug!(
                            "iter {iter}: tokenizer recv admitted={} aborted={} dropped={}",
                            outcome.admitted,
                            outcome.aborted,
                            outcome.dropped_unsupported,
                        );
                    }
                    if outcome.abort_all {
                        // Flag every running rid so the next pass aborts them.
                        for rid in &running_rids {
                            pending_aborts
                                .insert(rid.clone(), "aborted by user (abort_all)".into());
                        }
                    }
                    for (rid, msg) in outcome.running_aborts {
                        pending_aborts.insert(rid, msg);
                    }
                }
                Err(err) => log::warn!("tokenizer recv error: {err}"),
            }
        }

        // 1b. Apply any pending aborts to running reqs.
        if !pending_aborts.is_empty() {
            apply_pending_aborts(&mut running, &mut pending_aborts);
        }

        // 1c. Memory pressure check before building the next batch.
        //     If decoding the current running batch would exhaust the
        //     KV budget, try cache eviction first then retraction.
        //     With `disable_radix_cache`, skip eviction (no cache to
        //     evict from) — ChunkedCache semantics.
        if !running.is_empty() {
            let need = running.batch_size() as i64 * page_size;
            if page_tracker.available_size() < need {
                let cache_opt = if server_args.disable_radix_cache {
                    None
                } else {
                    Some(&mut radix_cache)
                };
                let outcome = reclaim_for_decode(
                    &mut running,
                    &mut waiting,
                    &mut page_tracker,
                    &mut req_pool,
                    cache_opt,
                    need,
                );
                if outcome.retracted_reqs + outcome.aborted > 0
                    || outcome.evicted_cache_tokens > 0
                {
                    log::info!(
                        "iter {iter}: reclaim evicted={} tokens, retracted={} reqs, aborted={} reqs",
                        outcome.evicted_cache_tokens,
                        outcome.retracted_reqs,
                        outcome.aborted,
                    );
                }
            }
        }

        // 2. Choose the next batch.  With `disable_radix_cache`, pass
        //    `None` so admission skips prefix matching and slot
        //    planting — matches Python's `ChunkCache.match_prefix`
        //    returning an empty match.
        let batch_to_send = if !waiting.is_empty() {
            let cache_opt = if server_args.disable_radix_cache {
                None
            } else {
                Some(&mut radix_cache)
            };
            builder.get_new_batch_prefill_with_cache(
                &running,
                &mut waiting,
                &page_tracker,
                &mut req_pool,
                cache_opt,
            )
        } else if !running.is_empty() {
            builder.prepare_for_decode(&mut running);
            if running.is_empty() {
                None
            } else {
                Some(swap_running(&mut running))
            }
        } else {
            None
        };

        let Some(mut batch) = batch_to_send else {
            // No work this tick.  If we have no source we bail; if we
            // do, sleep briefly and poll again.
            if tokenizer_source.is_none() {
                log::info!("scheduler idle and no tokenizer source — exiting after {iter} iter(s)");
                return Ok(());
            }
            idle_iters_without_source = idle_iters_without_source.saturating_add(1);
            // Adaptive back-off: ramp from ~1ms to ~10ms after a string
            // of idle ticks so we don't burn CPU spinning.
            let sleep_ms = (idle_iters_without_source.min(10)) as u64;
            std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
            continue;
        };
        idle_iters_without_source = 0;

        // 3. Drain pending frees into the wire payload.
        batch.indices_to_free = page_tracker.drain_pending_free();

        // 4. Build the wire payload and send it.  `vocab_size` and the
        //    worker `device` come off the cached handshake snapshot
        //    (see `WorkerSnapshot::from_handshake`).
        let payload =
            batch.to_model_worker_batch_payload(
                snapshot.vocab_size,
                &snapshot.device,
                Some(&req_pool),
            );
        let req = ForwardBatchGenerationReq {
            batch: payload.to_msgpack_value()?,
            pp_proxy_tensors: None,
            is_verify: false,
            skip_attn_backend_init: false,
        };
        let reply = workers.forward_batch_generation(req)?;

        // 5. Apply the reply to the scheduler-side state.  With
        //    `disable_radix_cache`, pass `None` so the finish path
        //    frees ALL the req's KV slots back to the worker instead
        //    of grafting them into the cache — matches Python's
        //    `ChunkCache.cache_finished_req`.
        let cache_opt = if server_args.disable_radix_cache {
            None
        } else {
            Some(&mut radix_cache)
        };
        let mut stats = process_batch_result_with_cache(
            &mut batch,
            &reply,
            &mut page_tracker,
            &mut req_pool,
            cache_opt,
        );

        // 5b. Stream the per-iteration output to the detokenizer.
        if let (Some(sink), Some(output)) =
            (detokenizer_sink.as_ref(), stats.detokenizer_output.take())
        {
            if let Err(err) = sink.send(&Wire::BatchTokenIDOutput(output)) {
                log::warn!("failed to push BatchTokenIDOutput to detokenizer: {err}");
            }
        }
        log::debug!(
            "iter {iter}: generated={} finished={} cuda_graph={}",
            stats.num_generated_tokens,
            stats.num_finished_reqs,
            stats.can_run_cuda_graph,
        );

        // 5c. Throughput logging — emits the Python-shaped
        //     `Prefill batch …` / `Decode batch …` lines.  Periodic
        //     for decode (every `decode_log_interval` steps), every-iter
        //     for prefill.
        match batch.forward_mode {
            ForwardMode::Extend | ForwardMode::Mixed => {
                let cached_tokens: u64 = batch
                    .reqs
                    .iter()
                    .map(|r| r.read().unwrap().prefix_len_from_cache as u64)
                    .sum();
                let new_input_tokens = batch.input_ids.len() as u64;
                if let Some(line) = metrics.report_prefill_step(
                    &batch,
                    new_input_tokens,
                    cached_tokens,
                    &page_tracker,
                    waiting.len(),
                ) {
                    log::info!("{line}");
                }
            }
            ForwardMode::Decode => {
                if let Some(line) = metrics.report_decode_step(
                    &batch,
                    stats.num_generated_tokens,
                    stats.can_run_cuda_graph,
                    &page_tracker,
                    waiting.len(),
                    server_args.decode_log_interval,
                ) {
                    log::info!("{line}");
                }
            }
            _ => {}
        }

        // 6. Promote the just-run batch back into `running` for the
        //    next iteration.  The Python equivalent is `self.last_batch
        //    = batch`.
        running = batch;
    }
}

fn swap_running(running: &mut ScheduleBatch) -> ScheduleBatch {
    let placeholder = ScheduleBatch::new(running.forward_mode);
    std::mem::replace(running, placeholder)
}

/// Inject `FinishReason::Aborted` into running reqs whose rids show
/// up in `pending`.  The output processor picks it up on the next
/// `check_finished` and releases their KV.
fn apply_pending_aborts(
    running: &mut ScheduleBatch,
    pending: &mut std::collections::HashMap<String, String>,
) {
    if pending.is_empty() || running.is_empty() {
        return;
    }
    for req_arc in running.reqs.iter() {
        let mut req = req_arc.write().unwrap();
        if let Some(msg) = pending.remove(&req.rid) {
            req.to_finish = Some(FinishReason::Aborted { message: msg });
        }
    }
}
