// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Rust port of `TpWorkerClient` + `TpWorkerClientGroup` from
//! `python/sglang/srt/managers/tp_worker_client.py`.
//!
//! Two layers:
//!
//!   * [`WorkerClient`] — owns a single ZMQ PAIR connection to one GPU
//!     worker (one TP rank).  Provides only the raw send / recv / rpc
//!     primitives; no per-RPC wrappers.
//!   * [`WorkerClientGroup`] — owns `Vec<WorkerClient>` (one per TP rank),
//!     does the initial handshake fan-out, and exposes the per-RPC
//!     wrappers the scheduler actually calls.  Every RPC picks one of
//!     the two dispatch patterns below.
//!
//! ## Dispatch matrix
//!
//! | RPC | Pattern |
//! |---|---|
//! | `handshake` | broadcast_leader_only |
//! | `forward_batch_generation` | broadcast_leader_only |
//! | `forward_batch_embedding` | broadcast_leader_only |
//! | `forward_batch_split_prefill` | broadcast_leader_only |
//! | `decode_step` | broadcast_leader_only |
//! | `get_mem_usage` | broadcast_leader_only |
//! | `get_weights_by_name` | broadcast_leader_only |
//! | `update_weights_from_disk` | broadcast_all_confirm |
//! | `update_weights_from_distributed` | broadcast_all_confirm |
//! | `update_weights_from_tensor` | broadcast_all_confirm |
//! | `update_weights_from_ipc` | broadcast_all_confirm |
//! | `init_weights_update_group` | broadcast_all_confirm |
//! | `destroy_weights_update_group` | broadcast_all_confirm |
//! | `init_weights_send_group_for_remote_instance` | broadcast_all_confirm |
//! | `send_weights_to_remote_instance` | broadcast_all_confirm |
//! | `load_lora_adapter` | broadcast_all_confirm |
//! | `unload_lora_adapter` | broadcast_all_confirm |
//! | `load_lora_adapter_from_tensors` | broadcast_all_confirm |
//!
//! **broadcast_leader_only**: send the request to every rank, return
//! the typed reply from rank 0, drain (and at most warn on) the
//! stragglers from ranks 1..N.  The drain is **load-bearing** — PAIR
//! sockets stay in lockstep, so the next RPC's reply has to find the
//! socket buffer empty.
//!
//! **broadcast_all_confirm**: send the request to every rank, collect
//! every reply, return them as a Vec in rank order so the caller can
//! validate they agree (or surface partial-success bugs).

use crate::transport::{Transport, TransportError};
use crate::wire::{
    DecodeForwardSlimOutput, DecodeStepControlReq, DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput, ForwardBatchEmbeddingReq,
    ForwardBatchGenerationReq, ForwardBatchSplitPrefillReq, GPUWorkerHandshakeReqInput,
    GPUWorkerHandshakeReqOutput, GetMemUsageReqInput, GetMemUsageReqOutput,
    GetWeightsByNameReqInput, GetWeightsByNameReqOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput, InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput, LoRAUpdateOutput, LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput, SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput, UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput, UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput, UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput, UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput, UpdateWeightsFromTensorReqOutput, Wire,
};

#[derive(Debug, thiserror::Error)]
pub enum WorkerClientError {
    #[error(transparent)]
    Transport(#[from] TransportError),

    #[error("expected reply variant {expected}, got {got}")]
    UnexpectedReply {
        expected: &'static str,
        got: &'static str,
    },

    #[error("worker client group needs at least one endpoint")]
    EmptyGroup,
}

/// Single per-rank ZMQ PAIR connection.
pub struct WorkerClient {
    transport: Transport,
    endpoint: String,
}

/// Helper — turn an unexpected `Wire` variant into a typed error.
macro_rules! expect_variant {
    ($reply:expr, $variant:ident) => {{
        let r = $reply;
        let got = r.variant_name();
        match r {
            Wire::$variant(inner) => Ok(inner),
            _ => Err(WorkerClientError::UnexpectedReply {
                expected: stringify!($variant),
                got,
            }),
        }
    }};
}

/// For `broadcast_all_confirm` callsites: pop the leader's reply,
/// surface any non-leader rank whose variant disagrees.  Variant-level
/// only for now; a future revision can extend to field-level equality
/// once the output structs grow `Eq` impls.
macro_rules! expect_first_and_check_all_match {
    ($replies:expr, $variant:ident, $rpc_name:literal) => {{
        let mut replies = $replies;
        if replies.is_empty() {
            return Err(WorkerClientError::UnexpectedReply {
                expected: stringify!($variant),
                got: "<no replies>",
            });
        }
        for (i, r) in replies.iter().enumerate().skip(1) {
            if !matches!(r, Wire::$variant(_)) {
                log::warn!(
                    "{}: rank{} returned {} (expected {})",
                    $rpc_name,
                    i,
                    r.variant_name(),
                    stringify!($variant),
                );
            }
        }
        let leader = replies.remove(0);
        expect_variant!(leader, $variant)
    }};
}

impl WorkerClient {
    /// Open the PAIR socket to a single worker.  The handshake itself is
    /// orchestrated by [`WorkerClientGroup`] (it's a fan-out concern, not
    /// a per-rank concern).
    pub fn connect(ctx: &zmq::Context, endpoint: &str) -> Result<Self, WorkerClientError> {
        let transport = Transport::connect(ctx, endpoint)?;
        Ok(Self {
            transport,
            endpoint: endpoint.to_string(),
        })
    }

    pub fn transport(&self) -> &Transport {
        &self.transport
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Send + recv as a single RPC.  Equivalent to `TpWorkerClient._rpc`
    /// in Python.  Most callers should use [`WorkerClientGroup`]'s
    /// per-RPC wrappers; this is exposed for low-level testing.
    pub fn rpc(&self, req: &Wire) -> Result<Wire, WorkerClientError> {
        self.transport.send(req)?;
        Ok(self.transport.recv()?)
    }

    /// Non-blocking send half — pair with [`Self::recv`] for the
    /// pipelined event loop.
    pub fn send(&self, req: &Wire) -> Result<(), WorkerClientError> {
        Ok(self.transport.send(req)?)
    }

    pub fn recv(&self) -> Result<Wire, WorkerClientError> {
        Ok(self.transport.recv()?)
    }
}

/// Fan-out client over one PAIR socket per TP rank.
///
/// Rank 0 is the leader for `broadcast_leader_only` RPCs.  All ranks
/// participate in every RPC (no leader-only dispatch exists in the
/// matrix — every request is sent to every worker).
pub struct WorkerClientGroup {
    clients: Vec<WorkerClient>,
    leader_idx: usize,
    leader_handshake: GPUWorkerHandshakeReqOutput,
}

impl WorkerClientGroup {
    /// Open one PAIR socket per endpoint, broadcast the handshake to
    /// every rank, and cache rank 0's reply as the group's snapshot.
    ///
    /// Empty `endpoints` is an error — TP=1 callers pass a one-element
    /// vec.
    pub fn connect_all(
        ctx: &zmq::Context,
        endpoints: &[String],
    ) -> Result<Self, WorkerClientError> {
        if endpoints.is_empty() {
            return Err(WorkerClientError::EmptyGroup);
        }
        let mut clients = Vec::with_capacity(endpoints.len());
        for ep in endpoints {
            clients.push(WorkerClient::connect(ctx, ep)?);
        }
        let leader_idx = 0;
        log::info!(
            "WorkerClientGroup connecting TP={} ranks (leader=rank{} @ {})",
            clients.len(),
            leader_idx,
            clients[leader_idx].endpoint(),
        );

        // Broadcast the handshake.  Send to every rank, then collect the
        // leader's reply *and* every other rank's reply (drained-with-
        // parity-check to keep the PAIR sockets in lockstep).
        for client in &clients {
            client.send(&Wire::GPUWorkerHandshakeReqInput(
                GPUWorkerHandshakeReqInput::default(),
            ))?;
        }
        let mut replies: Vec<GPUWorkerHandshakeReqOutput> = Vec::with_capacity(clients.len());
        for (i, client) in clients.iter().enumerate() {
            let reply = expect_variant!(client.recv()?, GPUWorkerHandshakeReqOutput)?;
            log::info!(
                "  rank{}: device={}, max_total_num_tokens={}, vocab_size={}, context_len={}",
                i,
                reply.device,
                reply.max_total_num_tokens,
                reply.vocab_size,
                reply.context_len,
            );
            replies.push(reply);
        }

        // Parity-check: every non-leader rank should agree on the
        // scheduler-visible sizing.  A mismatch is a worker-side
        // configuration bug — surface it loudly but don't fail-open,
        // since the leader's view is still usable.
        let leader = &replies[leader_idx];
        for (i, r) in replies.iter().enumerate() {
            if i == leader_idx {
                continue;
            }
            check_handshake_parity(leader, r, i);
        }

        let leader_handshake = replies.into_iter().nth(leader_idx).unwrap();
        Ok(Self {
            clients,
            leader_idx,
            leader_handshake,
        })
    }

    pub fn len(&self) -> usize {
        self.clients.len()
    }

    pub fn is_tp1(&self) -> bool {
        self.clients.len() == 1
    }

    /// Cached leader handshake reply.  Source of truth for scheduler-
    /// visible sizing and model config (see `WorkerSnapshot`).
    pub fn handshake(&self) -> &GPUWorkerHandshakeReqOutput {
        &self.leader_handshake
    }

    // ------------------------------------------------------------------
    // Dispatch primitives
    // ------------------------------------------------------------------

    /// Send `req` to every rank, return the typed reply from the leader,
    /// drain (and at most warn on) every other rank's reply.
    fn broadcast_leader_only(&self, req: Wire) -> Result<Wire, WorkerClientError> {
        for client in &self.clients {
            client.send(&req)?;
        }
        let mut leader_reply: Option<Wire> = None;
        for (i, client) in self.clients.iter().enumerate() {
            let reply = client.recv()?;
            if i == self.leader_idx {
                leader_reply = Some(reply);
            } else if let Wire::DecodeForwardSlimOutput(out) = &reply {
                // Non-leader forward replies carry the same shape but
                // their tensors aren't read.  Log at debug only.
                log::debug!(
                    "rank{}: drained DecodeForwardSlimOutput (can_run_cuda_graph={})",
                    i,
                    out.can_run_cuda_graph
                );
            } else {
                log::debug!(
                    "rank{}: drained {} reply on broadcast_leader_only",
                    i,
                    reply.variant_name()
                );
            }
        }
        leader_reply.ok_or(WorkerClientError::UnexpectedReply {
            expected: "leader reply",
            got: "<missing>",
        })
    }

    /// Send `req` to every rank, return every reply in rank order.
    /// Caller can then check they agree.
    fn broadcast_all_confirm(&self, req: Wire) -> Result<Vec<Wire>, WorkerClientError> {
        for client in &self.clients {
            client.send(&req)?;
        }
        let mut replies = Vec::with_capacity(self.clients.len());
        for client in &self.clients {
            replies.push(client.recv()?);
        }
        Ok(replies)
    }

    // ------------------------------------------------------------------
    // Forward-pass RPCs (broadcast_leader_only)
    // ------------------------------------------------------------------

    pub fn forward_batch_generation(
        &self,
        req: ForwardBatchGenerationReq,
    ) -> Result<DecodeForwardSlimOutput, WorkerClientError> {
        let reply = self.broadcast_leader_only(Wire::ForwardBatchGenerationReq(req))?;
        expect_variant!(reply, DecodeForwardSlimOutput)
    }

    pub fn decode_step(
        &self,
        req: DecodeStepControlReq,
    ) -> Result<DecodeForwardSlimOutput, WorkerClientError> {
        let reply = self.broadcast_leader_only(Wire::DecodeStepControlReq(req))?;
        expect_variant!(reply, DecodeForwardSlimOutput)
    }

    pub fn forward_batch_embedding(
        &self,
        req: ForwardBatchEmbeddingReq,
    ) -> Result<Wire, WorkerClientError> {
        // TODO(rust-port): port the embedding reply schema and return a
        // typed result instead of the raw `Wire`.
        self.broadcast_leader_only(Wire::ForwardBatchEmbeddingReq(req))
    }

    pub fn forward_batch_split_prefill(
        &self,
        req: ForwardBatchSplitPrefillReq,
    ) -> Result<DecodeForwardSlimOutput, WorkerClientError> {
        let reply = self.broadcast_leader_only(Wire::ForwardBatchSplitPrefillReq(req))?;
        expect_variant!(reply, DecodeForwardSlimOutput)
    }

    // ------------------------------------------------------------------
    // Memory usage (broadcast_leader_only — every rank reports the same KV pool size)
    // ------------------------------------------------------------------

    pub fn get_mem_usage(&self) -> Result<GetMemUsageReqOutput, WorkerClientError> {
        let reply = self
            .broadcast_leader_only(Wire::GetMemUsageReqInput(GetMemUsageReqInput::default()))?;
        expect_variant!(reply, GetMemUsageReqOutput)
    }

    // ------------------------------------------------------------------
    // Weight introspection (broadcast_leader_only — weights are replicated)
    // ------------------------------------------------------------------

    pub fn get_weights_by_name(
        &self,
        req: GetWeightsByNameReqInput,
    ) -> Result<GetWeightsByNameReqOutput, WorkerClientError> {
        let reply = self.broadcast_leader_only(Wire::GetWeightsByNameReqInput(req))?;
        expect_variant!(reply, GetWeightsByNameReqOutput)
    }

    // ------------------------------------------------------------------
    // LoRA admin (broadcast_all_confirm)
    // ------------------------------------------------------------------

    pub fn load_lora_adapter(
        &self,
        req: LoadLoRAAdapterReqInput,
    ) -> Result<LoRAUpdateOutput, WorkerClientError> {
        let replies = self.broadcast_all_confirm(Wire::LoadLoRAAdapterReqInput(req))?;
        expect_first_and_check_all_match!(replies, LoRAUpdateOutput, "load_lora_adapter")
    }

    pub fn unload_lora_adapter(
        &self,
        req: UnloadLoRAAdapterReqInput,
    ) -> Result<LoRAUpdateOutput, WorkerClientError> {
        let replies = self.broadcast_all_confirm(Wire::UnloadLoRAAdapterReqInput(req))?;
        expect_first_and_check_all_match!(replies, LoRAUpdateOutput, "unload_lora_adapter")
    }

    pub fn load_lora_adapter_from_tensors(
        &self,
        req: LoadLoRAAdapterFromTensorsReqInput,
    ) -> Result<LoRAUpdateOutput, WorkerClientError> {
        let replies =
            self.broadcast_all_confirm(Wire::LoadLoRAAdapterFromTensorsReqInput(req))?;
        expect_first_and_check_all_match!(
            replies,
            LoRAUpdateOutput,
            "load_lora_adapter_from_tensors"
        )
    }

    // ------------------------------------------------------------------
    // Weight updates (broadcast_all_confirm)
    // ------------------------------------------------------------------

    pub fn update_weights_from_disk(
        &self,
        req: UpdateWeightFromDiskReqInput,
    ) -> Result<UpdateWeightFromDiskReqOutput, WorkerClientError> {
        let replies = self.broadcast_all_confirm(Wire::UpdateWeightFromDiskReqInput(req))?;
        expect_first_and_check_all_match!(
            replies,
            UpdateWeightFromDiskReqOutput,
            "update_weights_from_disk"
        )
    }

    pub fn init_weights_update_group(
        &self,
        req: InitWeightsUpdateGroupReqInput,
    ) -> Result<InitWeightsUpdateGroupReqOutput, WorkerClientError> {
        let replies = self.broadcast_all_confirm(Wire::InitWeightsUpdateGroupReqInput(req))?;
        expect_first_and_check_all_match!(
            replies,
            InitWeightsUpdateGroupReqOutput,
            "init_weights_update_group"
        )
    }

    pub fn destroy_weights_update_group(
        &self,
        req: DestroyWeightsUpdateGroupReqInput,
    ) -> Result<DestroyWeightsUpdateGroupReqOutput, WorkerClientError> {
        let replies =
            self.broadcast_all_confirm(Wire::DestroyWeightsUpdateGroupReqInput(req))?;
        expect_first_and_check_all_match!(
            replies,
            DestroyWeightsUpdateGroupReqOutput,
            "destroy_weights_update_group"
        )
    }

    pub fn init_weights_send_group_for_remote_instance(
        &self,
        req: InitWeightsSendGroupForRemoteInstanceReqInput,
    ) -> Result<InitWeightsSendGroupForRemoteInstanceReqOutput, WorkerClientError> {
        let replies = self
            .broadcast_all_confirm(Wire::InitWeightsSendGroupForRemoteInstanceReqInput(req))?;
        expect_first_and_check_all_match!(
            replies,
            InitWeightsSendGroupForRemoteInstanceReqOutput,
            "init_weights_send_group_for_remote_instance"
        )
    }

    pub fn send_weights_to_remote_instance(
        &self,
        req: SendWeightsToRemoteInstanceReqInput,
    ) -> Result<SendWeightsToRemoteInstanceReqOutput, WorkerClientError> {
        let replies =
            self.broadcast_all_confirm(Wire::SendWeightsToRemoteInstanceReqInput(req))?;
        expect_first_and_check_all_match!(
            replies,
            SendWeightsToRemoteInstanceReqOutput,
            "send_weights_to_remote_instance"
        )
    }

    pub fn update_weights_from_distributed(
        &self,
        req: UpdateWeightsFromDistributedReqInput,
    ) -> Result<UpdateWeightsFromDistributedReqOutput, WorkerClientError> {
        let replies =
            self.broadcast_all_confirm(Wire::UpdateWeightsFromDistributedReqInput(req))?;
        expect_first_and_check_all_match!(
            replies,
            UpdateWeightsFromDistributedReqOutput,
            "update_weights_from_distributed"
        )
    }

    pub fn update_weights_from_tensor(
        &self,
        req: UpdateWeightsFromTensorReqInput,
    ) -> Result<UpdateWeightsFromTensorReqOutput, WorkerClientError> {
        let replies =
            self.broadcast_all_confirm(Wire::UpdateWeightsFromTensorReqInput(req))?;
        expect_first_and_check_all_match!(
            replies,
            UpdateWeightsFromTensorReqOutput,
            "update_weights_from_tensor"
        )
    }

    pub fn update_weights_from_ipc(
        &self,
        req: UpdateWeightsFromIPCReqInput,
    ) -> Result<UpdateWeightsFromIPCReqOutput, WorkerClientError> {
        let replies = self.broadcast_all_confirm(Wire::UpdateWeightsFromIPCReqInput(req))?;
        expect_first_and_check_all_match!(
            replies,
            UpdateWeightsFromIPCReqOutput,
            "update_weights_from_ipc"
        )
    }
}

/// Validate that every non-leader rank's handshake reply matches the
/// leader's on scheduler-visible sizing.  Mismatches are worker-side
/// configuration bugs (every rank should be the same model with the
/// same TP layout), so we surface them loudly but keep going.
fn check_handshake_parity(
    leader: &GPUWorkerHandshakeReqOutput,
    rank: &GPUWorkerHandshakeReqOutput,
    rank_idx: usize,
) {
    macro_rules! check {
        ($field:ident) => {
            if leader.$field != rank.$field {
                log::warn!(
                    "handshake mismatch on rank{}: {}: leader={:?}, rank{}={:?}",
                    rank_idx,
                    stringify!($field),
                    leader.$field,
                    rank_idx,
                    rank.$field,
                );
            }
        };
    }
    check!(max_total_num_tokens);
    check!(max_prefill_tokens);
    check!(max_running_requests);
    check!(max_req_len);
    check!(req_to_token_pool_size);
    check!(req_to_token_pool_max_context_len);
    check!(token_to_kv_pool_size);
    check!(vocab_size);
    check!(context_len);
    check!(is_generation);
}

