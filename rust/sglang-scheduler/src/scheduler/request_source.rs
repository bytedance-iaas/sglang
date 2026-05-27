// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Tokenizer → scheduler request source.
//!
//! Source: `Scheduler.recv_requests` + `Scheduler.process_input_requests`
//! in `scheduler.py`.  The Python side has a giant `TypeBasedDispatcher`
//! that routes every request type to a `handle_*` method.  This crate
//! only owns the handful of types the Rust scheduler actually drives
//! today (generate / batch-generate / abort); embedding mode is
//! recognised but dropped with a warning, and admin RPCs (LoRA, weight
//! update, …) belong on the worker-client side.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::queue::WaitingQueue;
use crate::scheduler::worker_snapshot::WorkerSnapshot;
use crate::transport::{PullSource, TransportError};
use crate::types::{Req, SamplingParams};
use crate::wire::{
    AbortReq, BatchTokenizedGenerateReqInput, SamplingParamsIPC, TokenizedGenerateReqInput, Wire,
};

#[derive(Debug, Default)]
pub struct RecvOutcome {
    pub admitted: u32,
    pub aborted: u32,
    pub dropped_unsupported: u32,
    /// True when an `AbortReq { abort_all: true }` arrived.  The
    /// caller should treat every running req as pending-abort too.
    pub abort_all: bool,
    /// Aborts for rids that were *running* at the time the AbortReq
    /// arrived.  Each entry is `(rid, message)`; the event loop applies
    /// them by setting `Req::to_finish = FinishReason::Aborted{..}`.
    pub running_aborts: HashMap<String, String>,
}

/// Pulls every queued request frame off `source`, decodes them, and
/// applies them to scheduler state.
///
/// Mirrors the Python while-NOBLOCK drain loop — the function never
/// blocks; an idle socket simply returns an empty outcome.
///
/// `snapshot` carries the worker-side model config so we can inject
/// sensible defaults (EOS stop tokens, context-length cap) onto reqs
/// that didn't ship their own.
pub fn drain_into(
    source: &PullSource,
    waiting: &mut WaitingQueue,
    running_rids: &HashSet<String>,
    snapshot: &WorkerSnapshot,
) -> Result<RecvOutcome, TransportError> {
    let mut outcome = RecvOutcome::default();
    let frames = source.drain_nonblocking()?;
    for frame in frames {
        match frame {
            Wire::TokenizedGenerateReqInput(req) => {
                let r = build_req(req, snapshot);
                waiting.push(Arc::new(RwLock::new(r)));
                outcome.admitted += 1;
            }
            Wire::BatchTokenizedGenerateReqInput(BatchTokenizedGenerateReqInput {
                reqs, ..
            }) => {
                for req in reqs {
                    let r = build_req(req, snapshot);
                    waiting.push(Arc::new(RwLock::new(r)));
                    outcome.admitted += 1;
                }
            }
            Wire::AbortReq(abort) => {
                apply_abort(&abort, waiting, running_rids, &mut outcome);
                outcome.aborted += 1;
            }
            Wire::TokenizedEmbeddingReqInput(_) | Wire::BatchTokenizedEmbeddingReqInput(_) => {
                log::warn!("embedding requests not yet supported by the Rust scheduler — dropping");
                outcome.dropped_unsupported += 1;
            }
            other => {
                log::warn!(
                    "unexpected wire variant on the tokenizer socket: {}",
                    other.variant_name()
                );
                outcome.dropped_unsupported += 1;
            }
        }
    }
    Ok(outcome)
}

fn build_req(input: TokenizedGenerateReqInput, snapshot: &WorkerSnapshot) -> Req {
    let rid = input.rid.clone().unwrap_or_else(|| {
        // Match Python's behaviour: rid is supposed to be set, but be
        // defensive — if the upstream is buggy we'd rather log a noisy
        // synthetic id than panic.
        let synthetic = format!("anon-{:p}", &input as *const _);
        log::warn!(
            "TokenizedGenerateReqInput without rid; synthesised {}",
            synthetic
        );
        synthetic
    });

    let mut sampling_params = sampling_from_wire(&input.sampling_params, input.return_logprob);

    // EOS fallback: if the tokenized request didn't ship any
    // stop_token_ids and the user didn't ask for `ignore_eos`, fall
    // back to the model's HuggingFace EOS list (carried in the worker
    // handshake).  Mirrors `Scheduler.handle_generate_request` in
    // `scheduler.py`, which passes `eos_token_ids=self.model_config.hf_eos_token_id`
    // into the `Req` constructor.
    if !sampling_params.ignore_eos
        && sampling_params
            .stop_token_ids
            .as_ref()
            .map(|v| v.is_empty())
            .unwrap_or(true)
        && !snapshot.default_stop_token_ids.is_empty()
    {
        sampling_params.stop_token_ids = Some(snapshot.default_stop_token_ids.clone());
    }

    let mut req = Req::new(
        rid,
        input.input_ids.iter().map(|&i| i as i32).collect(),
        sampling_params,
    );
    // The IPC carries logprob settings on the outer struct, not on
    // the `SamplingParams` Python dataclass; mirror that.
    req.sampling_params.return_logprob = input.return_logprob;
    req.sampling_params.logprob_start_len = input.logprob_start_len;
    req.sampling_params.top_logprobs_num = input.top_logprobs_num;

    // Hidden-states-return flag.  Drives the per-batch
    // `capture_hidden_mode` field on `ModelWorkerBatch`.
    req.return_hidden_states = input.return_hidden_states;

    // Context-length cap.  Python's `handle_generate_request` truncates
    // the prompt to fit `context_len - 1 - max_new_tokens` and warns;
    // we mirror the warning + truncate, but only when the snapshot
    // actually reports a non-zero context length (older workers ship 0).
    if snapshot.context_len > 0 {
        let prompt_len = req.origin_input_ids.len() as u32;
        let max_new = req.sampling_params.max_new_tokens;
        // Reserve at least one token for the first decode step.
        let budget = snapshot.context_len.saturating_sub(1);
        if prompt_len.saturating_add(max_new) > budget {
            let allow = budget.saturating_sub(max_new) as usize;
            let allow = allow.min(req.origin_input_ids.len());
            if allow < req.origin_input_ids.len() {
                log::warn!(
                    "rid={} prompt_len={} + max_new={} exceeds context_len={}; \
                     truncating prompt to {} tokens",
                    req.rid,
                    prompt_len,
                    max_new,
                    snapshot.context_len,
                    allow,
                );
                req.origin_input_ids.truncate(allow);
            }
        }
    }

    req
}

fn sampling_from_wire(p: &SamplingParamsIPC, return_logprob: bool) -> SamplingParams {
    let mut sp = SamplingParams {
        max_new_tokens: p.max_new_tokens,
        temperature: p.temperature as f64,
        top_p: p.top_p as f64,
        top_k: p.top_k,
        min_p: p.min_p as f64,
        frequency_penalty: p.frequency_penalty as f64,
        presence_penalty: p.presence_penalty as f64,
        repetition_penalty: p.repetition_penalty as f64,
        min_new_tokens: p.min_new_tokens,
        n: p.n,
        stop: None, // string-based stops are handled by the detokenizer
        stop_token_ids: p
            .stop_token_ids
            .as_ref()
            .map(|v| v.iter().map(|&t| t as i32).collect()),
        ignore_eos: p.ignore_eos,
        skip_special_tokens: p.skip_special_tokens,
        spaces_between_special_tokens: p.spaces_between_special_tokens,
        no_stop_trim: p.no_stop_trim,
        seed: p.sampling_seed,
        return_logprob,
        logprob_start_len: -1,
        top_logprobs_num: 0,
    };
    sp.normalize();
    sp
}

fn apply_abort(
    abort: &AbortReq,
    waiting: &mut WaitingQueue,
    running_rids: &HashSet<String>,
    outcome: &mut RecvOutcome,
) {
    if abort.abort_all {
        // Drop everything still queued, and tell the event loop to
        // mark every running rid as pending-abort.
        waiting.clear();
        outcome.abort_all = true;
        log::info!(
            "abort_all: cleared waiting queue ({} running requests will be aborted on next iteration)",
            running_rids.len()
        );
        return;
    }
    let Some(rid) = abort.rid.as_deref() else {
        log::warn!("AbortReq without rid and without abort_all — dropping");
        return;
    };
    let removed = waiting.drop_by_rid(rid) > 0;
    if removed {
        log::info!("aborted waiting req {}", rid);
    } else if running_rids.contains(rid) {
        let msg = abort
            .abort_message
            .clone()
            .unwrap_or_else(|| "aborted by user".into());
        outcome.running_aborts.insert(rid.to_string(), msg);
        log::info!(
            "aborted running req {} — to_finish will be applied on next iteration",
            rid
        );
    } else {
        log::warn!("AbortReq for unknown rid {}", rid);
    }
}
