// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! axum HTTP server exposing OpenAI-compatible API endpoints.
//!
//! Endpoints:
//!   GET  /health
//!   GET  /v1/models
//!   POST /v1/completions          (streaming + non-streaming)
//!   POST /v1/chat/completions     (streaming + non-streaming)

pub mod protocol;

use crate::http_server::protocol::*;
use crate::ipc::wire::SamplingParams;
use crate::tokenizer::{ResponseChunk, TmConfig, TokenizerManager};
use crate::HttpServerConfig;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use log::info;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;

// ──────────────────────────── shared state ──────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub tm: Arc<TokenizerManager>,
    pub model_name: String,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gen_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

// ─────────────────────────── router factory ─────────────────────────────────

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/model_info", get(model_info))
        .route("/get_model_info", get(get_model_info))
        .route("/get_server_info", get(get_server_info))
        .route("/server_info", get(server_info))
        .route("/get_weight_version", get(get_weight_version))
        .route("/weight_version", get(weight_version))
        .route("/ping", get(sagemaker_health))
        .route("/get_load", get(get_load))
        .route("/generate", post(generate).put(generate))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/health/generate", get(health_generate).post(health_generate).put(health_generate).delete(health_generate))
        .route("/get/server/info", get(get_server_info).post(get_server_info).put(get_server_info).delete(get_server_info))
        .route("/server/info", get(server_info).post(server_info).put(server_info).delete(server_info))
        .route("/get/weight/version", get(get_weight_version).post(get_weight_version).put(get_weight_version).delete(get_weight_version))
        .route("/weight/version", get(weight_version).post(weight_version).put(weight_version).delete(weight_version))
        .route("/encode", get(encode).post(encode).put(encode).delete(encode))
        .route("/tokenize", get(tokenize).post(tokenize).put(tokenize).delete(tokenize))
        .route("/v1/tokenize", get(v1_tokenize).post(v1_tokenize).put(v1_tokenize).delete(v1_tokenize))
        .route("/detokenize", get(detokenize).post(detokenize).put(detokenize).delete(detokenize))
        .route("/v1/detokenize", get(v1_detokenize).post(v1_detokenize).put(v1_detokenize).delete(v1_detokenize))
        .route("/v1/embeddings", get(v1_embeddings).post(v1_embeddings).put(v1_embeddings).delete(v1_embeddings))
        .route("/v1/classify", get(v1_classify).post(v1_classify).put(v1_classify).delete(v1_classify))
        .route("/classify", get(classify).post(classify).put(classify).delete(classify))
        .route("/v1/rerank", get(v1_rerank).post(v1_rerank).put(v1_rerank).delete(v1_rerank))
        .route("/v1/messages", get(v1_messages).post(v1_messages).put(v1_messages).delete(v1_messages))
        .route("/v1/messages/count/tokens", get(v1_messages_count_tokens).post(v1_messages_count_tokens).put(v1_messages_count_tokens).delete(v1_messages_count_tokens))
        .route("/get/load", get(get_load).post(get_load).put(get_load).delete(get_load))
        .route("/ping", get(ping).post(ping).put(ping).delete(ping))
        .route("/v1/models/model", get(v1_models_model).post(v1_models_model).put(v1_models_model).delete(v1_models_model))
        .route("/vertex/generate", get(vertex_generate).post(vertex_generate).put(vertex_generate).delete(vertex_generate))
        .route("/invocations", get(invocations).post(invocations).put(invocations).delete(invocations))
        .route("/api/tags", get(api_tags).post(api_tags).put(api_tags).delete(api_tags))
        .route("/api/chat", get(api_chat).post(api_chat).put(api_chat).delete(api_chat))
        .route("/api/generate", get(api_generate).post(api_generate).put(api_generate).delete(api_generate))
        .route("/api/show", get(api_show).post(api_show).put(api_show).delete(api_show))
        .route("/", get(root_route).post(root_route).put(root_route).delete(root_route))
        .route("/abort/request", get(abort_request).post(abort_request).put(abort_request).delete(abort_request))
        .route("/pause/generation", get(pause_generation).post(pause_generation).put(pause_generation).delete(pause_generation))
        .route("/continue/generation", get(continue_generation).post(continue_generation).put(continue_generation).delete(continue_generation))
        .route("/open/session", get(open_session).post(open_session).put(open_session).delete(open_session))
        .route("/close/session", get(close_session).post(close_session).put(close_session).delete(close_session))
        .route("/v1/responses", get(v1_responses).post(v1_responses).put(v1_responses).delete(v1_responses))
        .route("/v1/responses/id", get(v1_responses_id).post(v1_responses_id).put(v1_responses_id).delete(v1_responses_id))
        .route("/v1/responses/id/cancel", get(v1_responses_id_cancel).post(v1_responses_id_cancel).put(v1_responses_id_cancel).delete(v1_responses_id_cancel))
        .route("/flush/cache", get(flush_cache).post(flush_cache).put(flush_cache).delete(flush_cache))
        .route("/set/internal/state", get(set_internal_state).post(set_internal_state).put(set_internal_state).delete(set_internal_state))
        .route("/add/external/corpus", get(add_external_corpus).post(add_external_corpus).put(add_external_corpus).delete(add_external_corpus))
        .route("/remove/external/corpus", get(remove_external_corpus).post(remove_external_corpus).put(remove_external_corpus).delete(remove_external_corpus))
        .route("/list/external/corpora", get(list_external_corpora).post(list_external_corpora).put(list_external_corpora).delete(list_external_corpora))
        .route("/clear/hicache/storage/backend", get(clear_hicache_storage_backend).post(clear_hicache_storage_backend).put(clear_hicache_storage_backend).delete(clear_hicache_storage_backend))
        .route("/hicache/storage/backend", get(hicache_storage_backend).post(hicache_storage_backend).put(hicache_storage_backend).delete(hicache_storage_backend))
        .route("/start/profile", get(start_profile).post(start_profile).put(start_profile).delete(start_profile))
        .route("/stop/profile", get(stop_profile).post(stop_profile).put(stop_profile).delete(stop_profile))
        .route("/set/trace/level", get(set_trace_level).post(set_trace_level).put(set_trace_level).delete(set_trace_level))
        .route("/freeze/gc", get(freeze_gc).post(freeze_gc).put(freeze_gc).delete(freeze_gc))
        .route("/configure/logging", get(configure_logging).post(configure_logging).put(configure_logging).delete(configure_logging))
        .route("/dumper", get(dumper).post(dumper).put(dumper).delete(dumper))
        .route("/start/expert/distribution/record", get(start_expert_distribution_record).post(start_expert_distribution_record).put(start_expert_distribution_record).delete(start_expert_distribution_record))
        .route("/stop/expert/distribution/record", get(stop_expert_distribution_record).post(stop_expert_distribution_record).put(stop_expert_distribution_record).delete(stop_expert_distribution_record))
        .route("/dump/expert/distribution/record", get(dump_expert_distribution_record).post(dump_expert_distribution_record).put(dump_expert_distribution_record).delete(dump_expert_distribution_record))
        .route("/update/weights/from/disk", get(update_weights_from_disk).post(update_weights_from_disk).put(update_weights_from_disk).delete(update_weights_from_disk))
        .route("/update/weights/from/tensor", get(update_weights_from_tensor).post(update_weights_from_tensor).put(update_weights_from_tensor).delete(update_weights_from_tensor))
        .route("/update/weight/version", get(update_weight_version).post(update_weight_version).put(update_weight_version).delete(update_weight_version))
        .route("/update/weights/from/distributed", get(update_weights_from_distributed).post(update_weights_from_distributed).put(update_weights_from_distributed).delete(update_weights_from_distributed))
        .route("/update/weights/from/ipc", get(update_weights_from_ipc).post(update_weights_from_ipc).put(update_weights_from_ipc).delete(update_weights_from_ipc))
        .route("/get/weights/by/name", get(get_weights_by_name).post(get_weights_by_name).put(get_weights_by_name).delete(get_weights_by_name))
        .route("/weights/checker", get(weights_checker).post(weights_checker).put(weights_checker).delete(weights_checker))
        .route("/init/weights/send/group/for/remote/instance", get(init_weights_send_group_for_remote_instance).post(init_weights_send_group_for_remote_instance).put(init_weights_send_group_for_remote_instance).delete(init_weights_send_group_for_remote_instance))
        .route("/send/weights/to/remote/instance", get(send_weights_to_remote_instance).post(send_weights_to_remote_instance).put(send_weights_to_remote_instance).delete(send_weights_to_remote_instance))
        .route("/get/remote/instance/transfer/engine/info", get(get_remote_instance_transfer_engine_info).post(get_remote_instance_transfer_engine_info).put(get_remote_instance_transfer_engine_info).delete(get_remote_instance_transfer_engine_info))
        .route("/remote/instance/transfer/engine/info", get(remote_instance_transfer_engine_info).post(remote_instance_transfer_engine_info).put(remote_instance_transfer_engine_info).delete(remote_instance_transfer_engine_info))
        .route("/init/weights/update/group", get(init_weights_update_group).post(init_weights_update_group).put(init_weights_update_group).delete(init_weights_update_group))
        .route("/destroy/weights/update/group", get(destroy_weights_update_group).post(destroy_weights_update_group).put(destroy_weights_update_group).delete(destroy_weights_update_group))
        .route("/release/memory/occupation", get(release_memory_occupation).post(release_memory_occupation).put(release_memory_occupation).delete(release_memory_occupation))
        .route("/resume/memory/occupation", get(resume_memory_occupation).post(resume_memory_occupation).put(resume_memory_occupation).delete(resume_memory_occupation))
        .route("/slow/down", get(slow_down).post(slow_down).put(slow_down).delete(slow_down))
        .route("/load/lora/adapter", get(load_lora_adapter).post(load_lora_adapter).put(load_lora_adapter).delete(load_lora_adapter))
        .route("/load/lora/adapter/from/tensors", get(load_lora_adapter_from_tensors).post(load_lora_adapter_from_tensors).put(load_lora_adapter_from_tensors).delete(load_lora_adapter_from_tensors))
        .route("/unload/lora/adapter", get(unload_lora_adapter).post(unload_lora_adapter).put(unload_lora_adapter).delete(unload_lora_adapter))
        .route("/v1/audio/transcriptions", get(v1_audio_transcriptions).post(v1_audio_transcriptions).put(v1_audio_transcriptions).delete(v1_audio_transcriptions))
        .route("/parse/function/call", get(parse_function_call).post(parse_function_call).put(parse_function_call).delete(parse_function_call))
        .route("/separate/reasoning", get(separate_reasoning).post(separate_reasoning).put(separate_reasoning).delete(separate_reasoning))
        .route("/v1/score", get(v1_score).post(v1_score).put(v1_score).delete(v1_score))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ─────────────────────────────── handlers ───────────────────────────────────

async fn health() -> &'static str {
    "OK"
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelCard {
            id: state.model_name.clone(),
            object: "model".into(),
            created: now_secs(),
            owned_by: "sglang".into(),
        }],
    })
}

// ──────────────────────────── /model_info ───────────────────────────────────

async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    let cfg = &state.tm.config;
    Json(ModelInfo {
        model_path: cfg.model_name.clone(),
        tokenizer_path: cfg.tokenizer_path.clone(),
        is_generation: true,
        preferred_sampling_params: None,
        weight_version: None,
        has_image_understanding: false,
        has_audio_understanding: false,
        model_type: None,
        architectures: None,
    })
}

/// Deprecated alias for `/model_info`. Kept for backwards compatibility with
/// clients that still hit the old Python endpoint.
async fn get_model_info(state: State<Arc<AppState>>) -> Json<ModelInfo> {
    log::warn!(
        "Endpoint '/get_model_info' is deprecated and will be removed in a future version. \
         Please use '/model_info' instead."
    );
    model_info(state).await
}

// ─────────────────────────────── /generate ─────────────────────────────────

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    // Resolve input: either text (tokenize) or pre-supplied input_ids.
    let input_ids = match (&req.text, &req.input_ids) {
        (Some(GenerateText::Single(s)), _) => match state.tm.tokenize(s) {
            Ok(ids) => ids,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e),
        },
        (Some(GenerateText::Batch(v)), _) => {
            // Take the first prompt only — batched /generate is not yet supported.
            let s = v.first().map(String::as_str).unwrap_or("");
            match state.tm.tokenize(s) {
                Ok(ids) => ids,
                Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e),
            }
        }
        (None, Some(GenerateTokenIds::Single(ids))) => ids.clone(),
        (None, Some(GenerateTokenIds::Batch(v))) => v.first().cloned().unwrap_or_default(),
        (None, None) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Either `text` or `input_ids` must be provided",
            );
        }
    };

    let sp = sampling_params_from_dict(req.sampling_params.as_ref());
    let prompt_tokens = input_ids.len() as u32;
    let stream_mode = req.stream;
    let (rid, mut rx) = state.tm.submit(input_ids, sp, stream_mode).await;

    if stream_mode {
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let rid2 = rid.clone();
        tokio::spawn(async move {
            let mut accumulated = String::new();
            let mut completion_tokens: u32 = 0;
            while let Some(chunk) = rx.recv().await {
                let (data, done) = match chunk {
                    ResponseChunk::Delta { text } => {
                        accumulated.push_str(&text);
                        completion_tokens += 1;
                        let body = GenerateResponse {
                            text: accumulated.clone(),
                            meta_info: GenerateMetaInfo {
                                id: rid2.clone(),
                                finish_reason: None,
                                prompt_tokens,
                                completion_tokens,
                            },
                        };
                        (serde_json::to_string(&body).unwrap_or_default(), false)
                    }
                    ResponseChunk::Done { text, finish_reason } => {
                        accumulated.push_str(&text);
                        let body = GenerateResponse {
                            text: accumulated.clone(),
                            meta_info: GenerateMetaInfo {
                                id: rid2.clone(),
                                finish_reason: Some(FinishReason { kind: finish_reason }),
                                prompt_tokens,
                                completion_tokens,
                            },
                        };
                        (serde_json::to_string(&body).unwrap_or_default(), true)
                    }
                    ResponseChunk::Error(e) => (format!("{{\"error\":{{\"message\":\"{e}\"}}}}"), true),
                };
                if sse_tx.send(Ok(Event::default().data(data))).is_err() || done {
                    break;
                }
            }
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
        });

        Sse::new(UnboundedReceiverStream::new(sse_rx)).into_response()
    } else {
        let (full_text, finish_reason) = collect_text(rx).await;
        let resp = GenerateResponse {
            text: full_text,
            meta_info: GenerateMetaInfo {
                id: rid,
                finish_reason: Some(FinishReason { kind: finish_reason }),
                prompt_tokens,
                completion_tokens: 0, // not tracked in non-stream path; client can recount
            },
        };
        Json(resp).into_response()
    }
}

// ─────────────────────────── /v1/completions ────────────────────────────────

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let stream_mode = req.stream.unwrap_or(false);

    // Build text prompt from request
    let text = match &req.prompt {
        PromptInput::Text(s) => s.clone(),
        PromptInput::Batch(v) => v.first().cloned().unwrap_or_default(),
        PromptInput::Tokens(_) => {
            return error_response(StatusCode::BAD_REQUEST, "Token ID prompts not supported");
        }
        PromptInput::Empty => String::new(),
    };

    let input_ids = match state.tm.tokenize(&text) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e),
    };

    let sp = sampling_params_from_completion(&req);
    let model = req.model.clone();
    let (rid, mut rx) = state.tm.submit(input_ids, sp, stream_mode).await;

    if stream_mode {
        let id = gen_id("cmpl");
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let id2 = id.clone();
        let model2 = model.clone();
        tokio::spawn(async move {
            while let Some(chunk) = rx.recv().await {
                let (data, done) = match &chunk {
                    ResponseChunk::Delta { text } => {
                        let c = CompletionChunk {
                            id: id2.clone(),
                            object: "text_completion".into(),
                            created: now_secs(),
                            model: model2.clone(),
                            choices: vec![CompletionChunkChoice {
                                text: text.clone(),
                                index: 0,
                                finish_reason: None,
                                logprobs: None,
                            }],
                        };
                        (serde_json::to_string(&c).unwrap_or_default(), false)
                    }
                    ResponseChunk::Done { text, finish_reason } => {
                        let c = CompletionChunk {
                            id: id2.clone(),
                            object: "text_completion".into(),
                            created: now_secs(),
                            model: model2.clone(),
                            choices: vec![CompletionChunkChoice {
                                text: text.clone(),
                                index: 0,
                                finish_reason: Some(finish_reason.clone()),
                                logprobs: None,
                            }],
                        };
                        (serde_json::to_string(&c).unwrap_or_default(), true)
                    }
                    ResponseChunk::Error(e) => (format!("{{\"error\":\"{e}\"}}"), true),
                };
                if sse_tx.send(Ok(Event::default().data(data))).is_err() || done {
                    break;
                }
            }
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
        });

        let _ = rid;
        Sse::new(UnboundedReceiverStream::new(sse_rx)).into_response()
    } else {
        let (full_text, finish_reason) = collect_text(rx).await;
        let resp = CompletionResponse {
            id: gen_id("cmpl"),
            object: "text_completion".into(),
            created: now_secs(),
            model: req.model.clone(),
            choices: vec![CompletionChoice {
                text: full_text,
                index: 0,
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: UsageInfo::default(),
        };
        let _ = rid;
        Json(resp).into_response()
    }
}

// ──────────────────────── /v1/chat/completions ──────────────────────────────

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let stream_mode = req.stream.unwrap_or(false);

    // Build message list for template application
    let msgs: Vec<(&str, &str)> = req
        .messages
        .iter()
        .map(|m| {
            let role: &str = m.role.as_str();
            let content: &str = m
                .content
                .as_ref()
                .map(|c| match c {
                    ChatContent::Text(s) => s.as_str(),
                    ChatContent::Parts(_) => "",
                })
                .unwrap_or("");
            (role, content)
        })
        .collect();

    // We need owned strings because template may not live long enough
    let msg_owned: Vec<(String, String)> = msgs
        .iter()
        .map(|(r, c)| (r.to_string(), c.to_string()))
        .collect();

    let msg_refs: Vec<(&str, &str)> = msg_owned
        .iter()
        .map(|(r, c)| (r.as_str(), c.as_str()))
        .collect();

    let prompt = state.tm.apply_chat_template(&msg_refs, true);

    let input_ids = match state.tm.tokenize(&prompt) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e),
    };

    let sp = sampling_params_from_chat(&req);
    let model = req.model.clone();
    let (rid, mut rx) = state.tm.submit(input_ids, sp, stream_mode).await;

    if stream_mode {
        let id = gen_id("chatcmpl");
        // Bridge: send all SSE events (role delta + text chunks + [DONE]) into one channel.
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();

        let id2 = id.clone();
        let model2 = model.clone();
        tokio::spawn(async move {
            // First chunk: role delta (required by OpenAI spec for streaming)
            let role_chunk = ChatChunk {
                id: id2.clone(),
                object: "chat.completion.chunk".into(),
                created: now_secs(),
                model: model2.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".into()),
                        content: Some(String::new()),
                        tool_calls: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
            };
            let _ = sse_tx.send(Ok(Event::default().data(
                serde_json::to_string(&role_chunk).unwrap_or_default(),
            )));

            // Content chunks from the detokenizer
            while let Some(chunk) = rx.recv().await {
                let data = match &chunk {
                    ResponseChunk::Delta { text } => {
                        let c = ChatChunk {
                            id: id2.clone(),
                            object: "chat.completion.chunk".into(),
                            created: now_secs(),
                            model: model2.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(text.clone()),
                                    tool_calls: None,
                                },
                                finish_reason: None,
                                logprobs: None,
                            }],
                            usage: None,
                        };
                        serde_json::to_string(&c).unwrap_or_default()
                    }
                    ResponseChunk::Done { text, finish_reason } => {
                        let c = ChatChunk {
                            id: id2.clone(),
                            object: "chat.completion.chunk".into(),
                            created: now_secs(),
                            model: model2.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(text.clone()),
                                    tool_calls: None,
                                },
                                finish_reason: Some(finish_reason.clone()),
                                logprobs: None,
                            }],
                            usage: None,
                        };
                        let _ = sse_tx.send(Ok(Event::default().data(
                            serde_json::to_string(&c).unwrap_or_default(),
                        )));
                        break;
                    }
                    ResponseChunk::Error(e) => format!("{{\"error\":\"{e}\"}}"),
                };
                if sse_tx.send(Ok(Event::default().data(data))).is_err() {
                    break;
                }
            }
            // [DONE] sentinel
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
        });

        let _ = rid;
        Sse::new(UnboundedReceiverStream::new(sse_rx)).into_response()
    } else {
        let (full_text, finish_reason) = collect_text(rx).await;
        let resp = ChatCompletionResponse {
            id: gen_id("chatcmpl"),
            object: "chat.completion".into(),
            created: now_secs(),
            model: req.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant".into(),
                    content: Some(full_text),
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: UsageInfo::default(),
        };
        let _ = rid;
        Json(resp).into_response()
    }
}

// ─────────────────────────── helper utilities ───────────────────────────────

/// Drain a response channel to completion, accumulating text and finish reason.
async fn collect_text(mut rx: mpsc::UnboundedReceiver<ResponseChunk>) -> (String, String) {
    let mut full = String::new();
    let mut reason = "stop".to_string();
    while let Some(chunk) = rx.recv().await {
        match chunk {
            ResponseChunk::Delta { text } => full.push_str(&text),
            ResponseChunk::Done { text, finish_reason } => {
                full.push_str(&text);
                reason = finish_reason;
                break;
            }
            ResponseChunk::Error(e) => {
                reason = format!("error: {e}");
                break;
            }
        }
    }
    (full, reason)
}

fn error_response(status: StatusCode, msg: &str) -> Response {
    let body = ErrorResponse::bad_request(msg);
    (status, Json(body)).into_response()
}

fn sampling_params_from_completion(req: &CompletionRequest) -> SamplingParams {
    let mut sp = SamplingParams::default();
    if let Some(v) = req.max_tokens { sp.max_new_tokens = v; }
    if let Some(v) = req.temperature { sp.temperature = v; }
    if let Some(v) = req.top_p { sp.top_p = v; }
    if let Some(v) = req.top_k { sp.top_k = v; }
    if let Some(v) = req.min_p { sp.min_p = v; }
    if let Some(v) = req.presence_penalty { sp.presence_penalty = v; }
    if let Some(v) = req.frequency_penalty { sp.frequency_penalty = v; }
    if let Some(v) = req.repetition_penalty { sp.repetition_penalty = v; }
    if let Some(v) = req.seed { sp.seed = Some(v); }
    if let Some(v) = req.ignore_eos { sp.ignore_eos = v; }
    if let Some(v) = req.skip_special_tokens { sp.skip_special_tokens = v; }
    if let Some(stop) = &req.stop { sp.stop = Some(stop.as_vec()); }
    if let Some(rf) = &req.response_format {
        sp.json_schema = rf.json_schema_str();
    }
    sp
}

fn sampling_params_from_chat(req: &ChatCompletionRequest) -> SamplingParams {
    let mut sp = SamplingParams::default();
    let max_tokens = req.max_completion_tokens.or(req.max_tokens);
    if let Some(v) = max_tokens { sp.max_new_tokens = v; }
    if let Some(v) = req.temperature { sp.temperature = v; }
    if let Some(v) = req.top_p { sp.top_p = v; }
    if let Some(v) = req.top_k { sp.top_k = v; }
    if let Some(v) = req.min_p { sp.min_p = v; }
    if let Some(v) = req.presence_penalty { sp.presence_penalty = v; }
    if let Some(v) = req.frequency_penalty { sp.frequency_penalty = v; }
    if let Some(v) = req.repetition_penalty { sp.repetition_penalty = v; }
    if let Some(v) = req.seed { sp.seed = Some(v); }
    if let Some(v) = req.ignore_eos { sp.ignore_eos = v; }
    if let Some(v) = req.skip_special_tokens { sp.skip_special_tokens = v; }
    if let Some(stop) = &req.stop { sp.stop = Some(stop.as_vec()); }
    if let Some(rf) = &req.response_format {
        sp.json_schema = rf.json_schema_str();
    }
    sp
}

/// Build SamplingParams from the free-form `sampling_params` dict accepted by
/// `/generate`. Unknown keys are ignored. Mirrors the subset of fields the
/// Python SamplingParams understands that we forward over IPC.
fn sampling_params_from_dict(
    params: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> SamplingParams {
    let mut sp = SamplingParams::default();
    let Some(p) = params else { return sp };

    let as_u32 = |v: &serde_json::Value| v.as_u64().and_then(|n| u32::try_from(n).ok());
    let as_i32 = |v: &serde_json::Value| v.as_i64().and_then(|n| i32::try_from(n).ok());
    let as_f64 = |v: &serde_json::Value| v.as_f64();
    let as_bool = |v: &serde_json::Value| v.as_bool();
    let as_string_list = |v: &serde_json::Value| -> Option<Vec<String>> {
        match v {
            serde_json::Value::String(s) => Some(vec![s.clone()]),
            serde_json::Value::Array(arr) => Some(
                arr.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect(),
            ),
            _ => None,
        }
    };

    if let Some(v) = p.get("max_new_tokens").and_then(as_u32) { sp.max_new_tokens = v; }
    if let Some(v) = p.get("temperature").and_then(as_f64) { sp.temperature = v; }
    if let Some(v) = p.get("top_p").and_then(as_f64) { sp.top_p = v; }
    if let Some(v) = p.get("top_k").and_then(as_i32) { sp.top_k = v; }
    if let Some(v) = p.get("min_p").and_then(as_f64) { sp.min_p = v; }
    if let Some(v) = p.get("frequency_penalty").and_then(as_f64) { sp.frequency_penalty = v; }
    if let Some(v) = p.get("presence_penalty").and_then(as_f64) { sp.presence_penalty = v; }
    if let Some(v) = p.get("repetition_penalty").and_then(as_f64) { sp.repetition_penalty = v; }
    if let Some(v) = p.get("min_new_tokens").and_then(as_u32) { sp.min_new_tokens = v; }
    if let Some(v) = p.get("n").and_then(as_u32) { sp.n = v; }
    if let Some(v) = p.get("seed").and_then(|v| v.as_i64()) { sp.seed = Some(v); }
    if let Some(v) = p.get("ignore_eos").and_then(as_bool) { sp.ignore_eos = v; }
    if let Some(v) = p.get("skip_special_tokens").and_then(as_bool) { sp.skip_special_tokens = v; }
    if let Some(v) = p.get("spaces_between_special_tokens").and_then(as_bool) {
        sp.spaces_between_special_tokens = v;
    }
    if let Some(v) = p.get("no_stop_trim").and_then(as_bool) { sp.no_stop_trim = v; }
    if let Some(v) = p.get("stop").and_then(as_string_list) { sp.stop = Some(v); }
    if let Some(v) = p.get("stop_regex").and_then(as_string_list) { sp.stop_regex = Some(v); }
    if let Some(arr) = p.get("stop_token_ids").and_then(|v| v.as_array()) {
        let ids: Vec<i32> = arr.iter().filter_map(|x| x.as_i64().and_then(|n| i32::try_from(n).ok())).collect();
        if !ids.is_empty() { sp.stop_token_ids = Some(ids); }
    }
    if let Some(v) = p.get("json_schema").and_then(|v| v.as_str()) {
        sp.json_schema = Some(v.to_string());
    }
    sp
}

// ─────────────────────────────────── start ──────────────────────────────────

pub fn start(config: HttpServerConfig) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.worker_threads.unwrap_or(8) as usize)
        .enable_all()
        .thread_name("sglang-http")
        .build()
        .expect("Failed to build tokio runtime");

    rt.block_on(async move {
        let tm_cfg = TmConfig {
            tokenizer_ipc_name: config.tokenizer_ipc_name.clone(),
            scheduler_ipc_name: config.scheduler_ipc_name.clone(),
            tokenizer_path: config.tokenizer_path.clone(),
            skip_tokenizer_init: config.skip_tokenizer_init,
            model_name: config.model_name.clone(),
        };

        let tm = TokenizerManager::start(tm_cfg);

        let state = Arc::new(AppState {
            tm,
            model_name: config.model_name.clone(),
        });

        let app = build_router(state);
        let addr = format!("{}:{}", config.host, config.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|e| panic!("Failed to bind HTTP on {addr}: {e}"));

        info!("HTTP server listening on {addr}");
        axum::serve(listener, app)
            .await
            .expect("HTTP server error");
    });
}

async fn health_generate() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn get_server_info() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn server_info() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn get_weight_version() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn weight_version() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn encode() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn tokenize() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_tokenize() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn detokenize() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_detokenize() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_embeddings() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_classify() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn classify() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_rerank() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_messages() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_messages_count_tokens() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn get_load(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    log::warn!("Endpoint '/get_load' is deprecated and will be removed in a future version. Please use '/v1/loads' instead.");
    // In actual implementation, we would query the tokenizer manager / loads
    Json(serde_json::json!([]))
}

async fn ping() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_models_model() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn vertex_generate() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn invocations() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn api_tags() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn api_chat() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn api_generate() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn api_show() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn root_route() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn abort_request() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn pause_generation() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn continue_generation() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn open_session() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn close_session() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_responses() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_responses_id() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_responses_id_cancel() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn flush_cache() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn set_internal_state() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn add_external_corpus() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn remove_external_corpus() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn list_external_corpora() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn clear_hicache_storage_backend() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn hicache_storage_backend() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn start_profile() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn stop_profile() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn set_trace_level() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn freeze_gc() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn configure_logging() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn dumper() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn start_expert_distribution_record() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn stop_expert_distribution_record() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn dump_expert_distribution_record() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn update_weights_from_disk() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn update_weights_from_tensor() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn update_weight_version() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn update_weights_from_distributed() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn update_weights_from_ipc() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn get_weights_by_name() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn weights_checker() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn init_weights_send_group_for_remote_instance() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn send_weights_to_remote_instance() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn get_remote_instance_transfer_engine_info() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn remote_instance_transfer_engine_info() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn init_weights_update_group() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn destroy_weights_update_group() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn release_memory_occupation() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn resume_memory_occupation() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn slow_down() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn load_lora_adapter() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn load_lora_adapter_from_tensors() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn unload_lora_adapter() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_audio_transcriptions() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn parse_function_call() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn separate_reasoning() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn v1_score() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"error": "Not implemented yet"}))
}

async fn sagemaker_health() -> Response {
    Response::new("".into())
}
