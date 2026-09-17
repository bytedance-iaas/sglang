from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import quote

import torch
import torch.distributed

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import poll_and_all_reduce_attn_cp_tp_group
from sglang.srt.distributed.communication_op import attn_cp_tp_broadcast_pyobj
from sglang.srt.distributed.parallel_state import (
    P2PWork,
    P2PWorkGroup,
    TensorDictRecvHandle,
    get_vpp_pp_reverse_group,
)
from sglang.srt.distributed.pipeline_layout import (
    PipelineControlEnvelope,
    PipelineControlKind,
    PipelineLayout,
    PipelinePrefixRegistry,
    PipelineRankSchedule,
    PipelineReplicaIdentity,
    PipelineReplicaRegistry,
    PipelineResourceGate,
    PipelineResourceSnapshot,
    PipelineWavefrontAction,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    max_prefill_buffer_tokens,
)
from sglang.srt.sampling.sampling_observer_pp import (
    add_auxiliary_output_to_pp_tensors,
    pop_auxiliary_output_from_pp_tensors,
)
from sglang.srt.utils import DynamicGradMode, point_to_point_pyobj
from sglang.srt.utils.common import is_xpu

logger = logging.getLogger(__name__)

_VPP_ACTIVATION_TAG = 1
_VPP_CONTROL_TAG = 2
_VPP_PROTOCOL_VERSION = 1

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _pp_can_skip_output_comm(batch: ScheduleBatch) -> bool:
    """Check if output send/recv can be skipped for this batch."""
    return (
        envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get()
        and batch is not None
        and batch.forward_mode == ForwardMode.EXTEND
        and len(batch.reqs) == 1
        and not batch.contains_last_prefill_chunk
        and not batch.return_logprob
    )


@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool


class SchedulerPPMixin:
    # #region debug-point H1-H3:prefix-timeline
    def _pp_vpp_timeline_active(self: Scheduler) -> bool:
        return getattr(self, "_pp_vpp_timeline_enabled", False) and getattr(
            getattr(self, "profiler_manager", None), "torch_profiler", None
        ) is not None

    def _pp_vpp_timeline(self: Scheduler, event: str, **data) -> None:
        if not self._pp_vpp_timeline_active():
            return
        profile_id = self.profiler_manager.profile_id
        if getattr(self, "_pp_vpp_timeline_profile_id", None) != profile_id:
            self._pp_vpp_timeline_profile_id = profile_id
            self._pp_vpp_timeline_count = 0
        if self._pp_vpp_timeline_count >= 10000:
            return
        self._pp_vpp_timeline_count += 1
        record = json.dumps(
            {
                "event": event,
                "pp": self.ps.pp_rank,
                "tp": self.ps.tp_rank,
                "profile": profile_id,
                "wall_ns": time.time_ns(),
                "mono_ns": time.monotonic_ns(),
                **data,
            },
            separators=(",", ":"),
        )
        with torch.profiler.record_function(
            f"vpp_timeline/{event} {quote(record, safe=':,[]')}"
        ):
            pass
        if self.ps.tp_rank == 0:
            logger.info("[VPP_TIMELINE] %s", record)
        if self._pp_vpp_timeline_count == 10000:
            logger.warning("[VPP_TIMELINE] event limit reached; later markers omitted")

    # #endregion

    def _pp_vpp_enabled(self: Scheduler) -> bool:
        return get_parallel().pp_virtual_stages > 1

    def _pp_vpp_max_inflight(self: Scheduler) -> int:
        burst_size = get_parallel().pp_vpp_prefill_burst_size
        if burst_size == 1:
            return self.ps.pp_size
        return burst_size + self.ps.pp_size

    def _pp_prewarm_vpp_device_group(self: Scheduler) -> None:
        if not self._pp_vpp_enabled():
            return
        warmup_tensor = torch.zeros(
            1,
            dtype=torch.int32,
            device=self.pp_group.device,
        )
        torch.distributed.all_reduce(
            warmup_tensor,
            group=self.pp_group.device_group,
        )
        self.pp_group.device_module.synchronize()
        logger.info("VPP pipeline device group prewarm completed")

    @DynamicGradMode()
    def event_loop_pp(self: Scheduler):
        """
        A scheduler loop for pipeline parallelism.
        Notes:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. We use async send but sync recv to avoid desynchronization while minimizing the communication overhead.
        3. We can use async batch depth to buffer the outputs in the last stage for to allow overlapping the GPU computation and CPU processing and avoid last PP rank staggler.

        Unified Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1)% mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage(can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.

        ====================================================================
        """
        self.init_pp_loop_state()
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.ingest_requests()
                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)
                    with torch.profiler.record_function("send_reqs_to_next_stage"):
                        self.send_req_work = self._pp_send_pyobj_to_next_stage(
                            recv_reqs,
                            async_send=True,
                        )
                with torch.profiler.record_function("get_next_batch_to_run"):
                    plan = self.get_next_batch_to_run(
                        running_batch=self.running_batch, last_batch=self.last_batch
                    )
                    self.running_batch = plan.running_batch
                    self.mbs[mb_id] = plan.batch_to_run
                self.running_mbs[mb_id] = self.running_batch
                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = (
                        self._pp_recv_vpp_proxy_tensors(first_visit=True)
                        if self._pp_vpp_enabled()
                        else self._pp_recv_proxy_tensors()
                    )
                next_pp_outputs = None
                next_batch_result = None
                d2h_event = None
                if get_parallel().pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if cur_batch:
                    if self._pp_vpp_enabled():
                        result, self.launch_event = self._pp_launch_vpp_batch(
                            mb_id,
                            cur_batch,
                            pp_proxy_tensors,
                            self.mb_metadata,
                            self.last_rank_comm_queue,
                        )
                    else:
                        result, self.launch_event = self._pp_launch_batch(
                            mb_id,
                            cur_batch,
                            pp_proxy_tensors,
                            self.mb_metadata,
                            self.last_rank_comm_queue,
                        )
                if get_parallel().pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    with torch.profiler.record_function("process_batch_result"):
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]
                if not self.pp_group.is_last_rank and not self._pp_vpp_enabled():
                    if cur_batch:
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        with torch.profiler.record_function(
                            "send_proxy_dict_to_next_stage"
                        ):
                            self.send_proxy_work = self._pp_send_dict_to_next_stage(
                                result.pp_hidden_states_proxy_tensors.tensors,
                                async_send=True,
                                msg_type="proxy",
                            )

                self.pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                self.on_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        This is the prefill server event loop for pipeline parallelism.

        Notes:
        1. Following the same rules as the event_loop_pp.
        2. Adds extra steps for KV transfer process: bootstrap + release.

        Prefill Server Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith bootstrap req from previous stage
        recv ith transferred req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1) % mb_size th consensus bootstrapped req from previous stage
        local consensus on bootstrapped req
        recv prev (i+1) % mb_size th release req from previous stage
        local consensus on release req
        recv prev (i+1) % mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith bootstrap req to next stage
        send ith transferred req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage (can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.
        ====================================================================

        There are two additional elements compared to the regular schedule:

        Bootstrap Requests + Release Requests:
        - Both can have local failure and need to be consensus on. PP needs to guarantee eventual consistency of local failure and flush malfunc requests out as soft error.

        """
        if self._pp_vpp_enabled():
            self._event_loop_pp_disagg_prefill_vpp()
            return

        self.init_pp_loop_state()

        # PD additional state initialization
        bmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_bootstrapped_rids: Optional[List[str]] = None
        transferred_rids: List[str] = []
        release_rids: Optional[List[str]] = None
        send_bootstrapped_work = []
        send_transfer_work = []
        send_consensus_bootstrapped_work = []
        send_release_work = []
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_release_rids = None
                next_consensus_bootstrapped_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.ingest_requests()

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                bootstrapped_rids = self._pp_pd_get_bootstrapped_ids()
                bmbs[mb_id] = bootstrapped_rids
                self._pp_commit_comm_work(send_bootstrapped_work)

                transferred_rids = self._pp_pd_get_prefill_transferred_ids()
                self._pp_commit_comm_work(send_transfer_work)
                tmbs[mb_id] = transferred_rids

                self.process_prefill_chunk(
                    last_batch=self.last_batch, running_batch=self.running_batch
                )
                prefill_plan = self.get_new_batch_prefill(self.running_batch)
                batch = prefill_plan.batch_to_run
                self.running_batch = prefill_plan.running_batch
                batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = (
                        self._pp_recv_vpp_proxy_tensors(first_visit=True)
                        if self._pp_vpp_enabled()
                        else self._pp_recv_proxy_tensors()
                    )

                if get_parallel().pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)
                if cur_batch:
                    if self.enable_staging:
                        self.maybe_prefetch_staging_for_batch(cur_batch)
                    if self._pp_vpp_enabled():
                        result, self.launch_event = self._pp_launch_vpp_batch(
                            mb_id,
                            cur_batch,
                            pp_proxy_tensors,
                            self.mb_metadata,
                            self.last_rank_comm_queue,
                        )
                    else:
                        result, self.launch_event = self._pp_launch_batch(
                            mb_id,
                            cur_batch,
                            pp_proxy_tensors,
                            self.mb_metadata,
                            self.last_rank_comm_queue,
                        )
                if get_parallel().pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                send_consensus_bootstrapped_work, consensus_bootstrapped_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        bmbs,
                        next_first_rank_mb_id,
                        consensus_bootstrapped_rids,
                        bootstrapped_rids,
                    )
                )
                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if bmbs[next_mb_id] is not None:
                    next_consensus_bootstrapped_rids = (
                        self._pp_recv_pyobj_from_prev_stage()
                    )
                    next_consensus_bootstrapped_rids = self.process_bootstrapped_queue(
                        next_consensus_bootstrapped_rids
                    )
                self._pp_commit_comm_work(send_consensus_bootstrapped_work)
                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                self._pp_commit_comm_work(send_release_work)
                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    self._pp_process_batch_result(
                        self.mbs[next_mb_id],
                        next_batch_result,
                    )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if tmbs[next_mb_id] is not None:
                    self.process_disagg_prefill_inflight_queue(next_release_rids)
                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                        bootstrapped_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if cur_batch and not self._pp_vpp_enabled():
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_bootstrapped_rids = next_consensus_bootstrapped_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.on_idle()

    def _pp_vpp_batch_manifest(
        self: Scheduler,
        batch_seq: int,
        slot_id: int,
        batch: ScheduleBatch,
    ) -> Dict[str, object]:
        requests = []
        for req in batch.reqs:
            extend_range = req.extend_range
            requests.append(
                (
                    req.rid,
                    -1 if extend_range is None else int(extend_range.start),
                    -1 if extend_range is None else int(extend_range.end),
                )
            )
        return {
            "protocol_version": _VPP_PROTOCOL_VERSION,
            "batch_seq": batch_seq,
            "generation": batch_seq // len(self.mbs),
            "slot_id": slot_id,
            "requests": tuple(requests),
        }

    def _pp_vpp_activation_group(self: Scheduler, source_rank: int):
        if self.ps.pp_size == 2 and source_rank == 1:
            return get_vpp_pp_reverse_group()
        return self.pp_group

    def _pp_vpp_start_receiver(self: Scheduler) -> None:
        if self._pp_vpp_pending_recv is not None:
            return
        source_rank = (self.ps.pp_rank - 1) % self.ps.pp_size
        activation_group = self._pp_vpp_activation_group(source_rank)
        self._pp_vpp_pending_recv = activation_group.recv_tensor_dict_async(
            all_gather_group=self.attn_tp_group,
            batch_p2p=True,
            tag=_VPP_ACTIVATION_TAG,
        )

    def _pp_vpp_accept_received_tensors(
        self: Scheduler,
        tensors: Dict[str, object],
    ) -> None:
        message_kind = tensors.get("__msg_type__", "default")
        protocol_version = int(tensors.get("vpp_protocol_version", -1))
        batch_seq = int(tensors.get("vpp_batch_seq", -1))
        generation = int(tensors.get("vpp_generation", -1))
        source_stage_id = int(tensors.get("vpp_src_stage_id", -1))
        stage_id = int(tensors.get("vpp_stage_id", -1))
        if message_kind != "vpp_proxy":
            raise RuntimeError(
                f"VPP activation receiver got unexpected message kind {message_kind}"
            )
        if (
            protocol_version != _VPP_PROTOCOL_VERSION
            or batch_seq < 0
            or generation != batch_seq // len(self._pp_vpp_slot_batch_seqs)
            or stage_id <= 0
            or source_stage_id != stage_id - 1
        ):
            raise RuntimeError(
                "VPP activation receiver got invalid identity: "
                f"protocol={protocol_version}, batch={batch_seq}, "
                f"generation={generation}, "
                f"source_stage={source_stage_id}, stage={stage_id}"
            )
        if stage_id % self.ps.pp_size != self.ps.pp_rank:
            raise RuntimeError(
                f"VPP activation for stage {stage_id} arrived on PP rank "
                f"{self.ps.pp_rank}"
            )

        slot_id = batch_seq % len(self._pp_vpp_slot_batch_seqs)
        if self._pp_vpp_slot_batch_seqs[slot_id] != batch_seq:
            raise RuntimeError(
                f"Stale VPP activation for batch {batch_seq} in slot {slot_id}"
            )
        key = (batch_seq, stage_id)
        if key in self._pp_vpp_ready_proxies:
            raise RuntimeError(f"Duplicate VPP activation for batch/stage {key}")
        self._pp_vpp_ready_proxies[key] = PPProxyTensors(tensors)
        if hasattr(self, "_pp_vpp_arrivals"):
            self._pp_vpp_arrivals.append(key)
        self._pp_vpp_pending_recv = None
        self._pp_vpp_start_receiver()

    def _pp_vpp_poll_receiver_tp_consensus(self: Scheduler) -> bool:
        handle: Optional[TensorDictRecvHandle] = self._pp_vpp_pending_recv
        if handle is None:
            self._pp_vpp_start_receiver()
            handle = self._pp_vpp_pending_recv
        payload_ready = handle.poll_payload_ready()
        if not all(self.attn_tp_group.all_gather_object(payload_ready)):
            return False
        handle.start_all_gather()
        receive_complete = handle.poll_all_gather()
        if not all(self.attn_tp_group.all_gather_object(receive_complete)):
            return False
        self._pp_vpp_accept_received_tensors(handle.result())
        return True

    def _pp_vpp_select_tp_action(
        self: Scheduler,
        rank_schedule: PipelineRankSchedule,
        tick: int,
        is_ready,
    ) -> Optional[PipelineWavefrontAction]:
        local_ready = tuple(
            task for task in rank_schedule.ready_tasks if is_ready(task[0], task[1])
        )
        ready_by_lane = self.attn_tp_group.all_gather_object(local_ready)
        common_ready = (
            set.intersection(*(set(tasks) for tasks in ready_by_lane))
            if ready_by_lane
            else set()
        )
        action = (
            rank_schedule.next_action(
                tick,
                is_ready=lambda batch_seq, stage_id: (
                    batch_seq,
                    stage_id,
                )
                in common_ready,
            )
            if self.ps.tp_rank == 0
            else None
        )
        identity = (
            None if action is None else (action.tick, action.batch_seq, action.stage_id)
        )
        identity = self.attn_tp_group.broadcast_object(identity, src=0)
        if identity is None:
            return None
        action_tick, batch_seq, stage_id = map(int, identity)
        if self.ps.tp_rank == 0:
            return action
        if not is_ready(batch_seq, stage_id):
            raise RuntimeError(
                f"TP leader selected unavailable VPP task {(batch_seq, stage_id)}"
            )
        return rank_schedule.dispatch(action_tick, batch_seq, stage_id)

    def _pp_vpp_take_ready_proxy(
        self: Scheduler,
        action: PipelineWavefrontAction,
    ) -> PPProxyTensors:
        key = (action.batch_seq, action.stage_id)
        try:
            return self._pp_vpp_ready_proxies.pop(key)
        except KeyError:
            raise RuntimeError(
                f"VPP activation for batch/stage {key} is not ready"
            ) from None

    def _pp_vpp_reap_send_work(self: Scheduler, pending_work: deque) -> None:
        while pending_work:
            work_group = pending_work[0]
            if isinstance(work_group, list):
                work_group = P2PWorkGroup(work_group)
                pending_work[0] = work_group
            if not work_group.poll():
                return
            pending_work.popleft()

    def _pp_vpp_layout_digest(self: Scheduler) -> str:
        return PipelineLayout.build(
            num_hidden_layers=self.model_config.num_hidden_layers,
            physical_size=self.ps.pp_size,
            virtual_stages=get_parallel().pp_virtual_stages,
        ).digest

    def _pp_vpp_start_control_receiver(self: Scheduler) -> None:
        if self.ps.tp_rank != 0:
            return
        if self._pp_vpp_pending_control_recv is not None:
            return
        self._pp_vpp_pending_control_recv = self.pp_group.recv_tensor_dict_async(
            batch_p2p=True,
            tag=_VPP_CONTROL_TAG,
        )

    def _pp_vpp_poll_control_receiver(
        self: Scheduler,
    ) -> Optional[Tuple[PipelineControlEnvelope, Dict[str, object]]]:
        if self.ps.tp_rank != 0:
            raise RuntimeError("only TP0 may poll the VPP control ring")
        handle: Optional[TensorDictRecvHandle] = self._pp_vpp_pending_control_recv
        if handle is None:
            self._pp_vpp_start_control_receiver()
            handle = self._pp_vpp_pending_control_recv
        wire = handle.poll()
        if wire is None:
            return None
        envelope = PipelineControlEnvelope.from_dict(wire)
        if envelope.protocol_version != _VPP_PROTOCOL_VERSION:
            raise RuntimeError("VPP control protocol version mismatch")
        if envelope.runtime_epoch != self._pp_vpp_runtime_epoch:
            raise RuntimeError("stale VPP control runtime epoch")
        if envelope.layout_digest != self._pp_vpp_control_layout_digest:
            raise RuntimeError("VPP control layout digest mismatch")
        self._pp_vpp_pending_control_recv = None
        self._pp_vpp_start_control_receiver()
        return envelope, wire

    def _pp_vpp_poll_control_receiver_tp_broadcast(
        self: Scheduler,
    ) -> Optional[Tuple[PipelineControlEnvelope, Dict[str, object]]]:
        wire = None
        if self.ps.tp_rank == 0:
            message = self._pp_vpp_poll_control_receiver()
            if message is not None:
                wire = message[1]
        wire = self.attn_tp_group.broadcast_object(wire, src=0)
        if wire is None:
            return None
        envelope = PipelineControlEnvelope.from_dict(wire)
        if envelope.protocol_version != _VPP_PROTOCOL_VERSION:
            raise RuntimeError("VPP control protocol version mismatch")
        if envelope.runtime_epoch != self._pp_vpp_runtime_epoch:
            raise RuntimeError("stale VPP control runtime epoch")
        if envelope.layout_digest != self._pp_vpp_control_layout_digest:
            raise RuntimeError("VPP control layout digest mismatch")
        return envelope, wire

    def _pp_vpp_queue_control(
        self: Scheduler,
        envelope: PipelineControlEnvelope,
        tensors: Optional[Dict[str, object]] = None,
    ) -> None:
        if self.ps.tp_rank != 0:
            return
        wire = envelope.to_dict()
        if tensors:
            cuda_keys = [
                key
                for key, value in tensors.items()
                if isinstance(value, torch.Tensor) and not value.is_cpu
            ]
            if cuda_keys:
                raise RuntimeError(f"VPP control payload must be CPU-only: {cuda_keys}")
            wire.update(tensors)
        self._pp_vpp_control_outbox.append(wire)

    def _pp_vpp_flush_control_outbox(
        self: Scheduler,
        pending_work: deque,
        max_pending: int,
    ) -> None:
        if self.ps.tp_rank != 0:
            return
        self._pp_vpp_reap_send_work(pending_work)
        while self._pp_vpp_control_outbox and len(pending_work) < max_pending:
            wire = self._pp_vpp_control_outbox.popleft()
            work = self.pp_group.send_tensor_dict(
                wire,
                async_send=True,
                batch_p2p=True,
                tag=_VPP_CONTROL_TAG,
            )
            if work:
                pending_work.append(work)

    def _pp_vpp_new_control(
        self: Scheduler,
        kind: PipelineControlKind,
        *,
        batch_seq: int = -1,
        slot_id: int = -1,
        payload: Optional[Dict[str, object]] = None,
    ) -> PipelineControlEnvelope:
        generation = (
            -1 if batch_seq < 0 else batch_seq // len(self._pp_vpp_slot_batch_seqs)
        )
        return PipelineControlEnvelope(
            protocol_version=_VPP_PROTOCOL_VERSION,
            runtime_epoch=self._pp_vpp_runtime_epoch,
            layout_digest=self._pp_vpp_control_layout_digest,
            kind=kind,
            source_rank=self.ps.pp_rank,
            batch_seq=batch_seq,
            generation=generation,
            slot_id=slot_id,
            payload=payload,
        )

    def _pp_vpp_resource_snapshot(
        self: Scheduler,
        pending_send_work,
    ) -> PipelineResourceSnapshot:
        allocator = self.token_to_kv_pool_allocator
        if hasattr(allocator, "full_available_size") and hasattr(
            allocator, "swa_available_size"
        ):
            kv_tokens = min(
                int(allocator.full_available_size())
                + int(self.tree_cache.full_evictable_size()),
                int(allocator.swa_available_size())
                + int(self.tree_cache.swa_evictable_size()),
            )
        else:
            kv_tokens = int(allocator.available_size()) + int(
                self.tree_cache.evictable_size()
            )
        activation_bytes = 0
        for proxy in self._pp_vpp_ready_proxies.values():
            activation_bytes += sum(
                value.numel() * value.element_size()
                for value in proxy.tensors.values()
                if isinstance(value, torch.Tensor)
            )
        pending_recv = getattr(self, "_pp_vpp_pending_recv", None)
        if pending_recv is not None and hasattr(pending_recv, "buffered_tensor_bytes"):
            activation_bytes += pending_recv.buffered_tensor_bytes()
        if isinstance(pending_send_work, int):
            pending_send_count = pending_send_work
        else:
            pending_send_count = len(pending_send_work)
            for work_group in pending_send_work:
                if isinstance(work_group, P2PWorkGroup):
                    activation_bytes += work_group.payload_bytes()
                else:
                    activation_bytes += sum(
                        item.payload.numel() * item.payload.element_size()
                        for item in work_group
                        if isinstance(item.payload, torch.Tensor)
                    )
        return PipelineResourceSnapshot(
            request_slots=int(self.req_to_token_pool.available_size()),
            kv_tokens=kv_tokens,
            activation_bytes=activation_bytes,
            pending_sends=pending_send_count,
            metadata_slots=int(
                self.req_to_metadata_buffer_idx_allocator.available_size()
            ),
        )

    def _pp_vpp_find_req(self: Scheduler, rid: str) -> Optional[Req]:
        for batch in [*self.mbs, *self.last_mbs]:
            if batch is None:
                continue
            for req in batch.reqs:
                if req.rid == rid:
                    return req
        if self.chunked_req is not None and self.chunked_req.rid == rid:
            return self.chunked_req
        return None

    def _pp_vpp_bootstrap_prefix_boundaries(
        self: Scheduler,
        rids: List[str],
    ) -> Dict[str, int]:
        wanted = set(rids)
        return {
            req.rid: len(req.prefix_indices)
            for req in self.disagg_prefill_bootstrap_queue.queue
            if req.rid in wanted
        }

    def _pp_vpp_merge_bootstrap_status(
        self: Scheduler,
        payload: Dict[str, object],
        local_good: List[str],
        local_bad: List[str],
    ) -> None:
        payload["good"] = sorted(set(payload["good"]).intersection(local_good))
        payload["bad"] = sorted(set(payload["bad"]).union(local_bad))
        local_metadata_slots = min(
            self.attn_tp_group.all_gather_object(
                int(self.req_to_metadata_buffer_idx_allocator.available_size())
            )
        )
        payload["metadata_slots"] = min(
            int(payload["metadata_slots"]),
            local_metadata_slots,
        )
        local_boundaries = self._pp_vpp_bootstrap_prefix_boundaries(payload["good"])
        payload["prefix_boundaries"] = {
            rid: min(
                int(payload["prefix_boundaries"][rid]),
                local_boundaries[rid],
            )
            for rid in payload["good"]
        }

    def _pp_vpp_bootstrap_apply_status(
        self: Scheduler,
        payload: Dict[str, object],
    ) -> List[List[str]]:
        good = sorted(set(payload["good"]))
        metadata_slots = max(0, int(payload["metadata_slots"]))
        return [
            good[:metadata_slots],
            sorted(set(payload["bad"])),
        ]

    def _pp_vpp_apply_bootstrap_prefix_boundaries(
        self: Scheduler,
        boundaries: Dict[str, int],
        rids: set[str],
    ) -> None:
        for req in self.disagg_prefill_bootstrap_queue.queue:
            if req.rid not in rids:
                continue
            boundary = int(boundaries[req.rid])
            already_applied = (
                req.vpp_prefix_limit == boundary and len(req.prefix_indices) <= boundary
            )
            req.vpp_prefix_limit = boundary
            if not already_applied and len(req.prefix_indices) > boundary:
                req.init_next_round_input(self.tree_cache)
            if len(req.prefix_indices) > boundary:
                raise RuntimeError(
                    f"VPP prefix cap failed for {req.rid}: "
                    f"expected <= {boundary}, got {len(req.prefix_indices)}"
                )

    def _pp_vpp_try_apply_bootstrap_status(
        self: Scheduler,
        boundaries: Dict[str, int],
        remaining_good: set[str],
        remaining_bad: set[str],
    ) -> bool:
        self._pp_vpp_apply_bootstrap_prefix_boundaries(boundaries, remaining_good)
        applied_good, applied_bad = self.process_bootstrapped_queue(
            [sorted(remaining_good), sorted(remaining_bad)]
        )
        remaining_good.difference_update(applied_good)
        remaining_bad.difference_update(applied_bad)

        # Applying the status is idempotent. A target that disappeared from the
        # bootstrap queue was already moved to its next local state by an earlier
        # attempt and must not prevent the ring ACK from advancing.
        queued_rids = {req.rid for req in self.disagg_prefill_bootstrap_queue.queue}
        remaining_good.intersection_update(queued_rids)
        remaining_bad.intersection_update(queued_rids)
        return not remaining_good and not remaining_bad

    def _pp_vpp_register_prefix_batch(
        self: Scheduler,
        batch: ScheduleBatch,
    ) -> None:
        for req in batch.reqs:
            extend_range = req.extend_range
            if extend_range is None:
                continue
            request_generation = int(getattr(req, "session_generation", None) or 0)
            entry = self._pp_vpp_prefix_registry.get(req.rid, request_generation)
            start = extend_range.start if entry is None else entry.planned_end
            self._pp_vpp_prefix_registry.plan(
                rid=req.rid,
                request_generation=request_generation,
                residency_generation=request_generation,
                start=start,
                end=extend_range.end,
                required_stages=range(
                    self.ps.pp_size * get_parallel().pp_virtual_stages
                ),
            )
            if not hasattr(req, "_vpp_original_skip_radix_cache_insert"):
                req._vpp_original_skip_radix_cache_insert = req.skip_radix_cache_insert
            req.skip_radix_cache_insert = True

    def _pp_vpp_apply_prefix_materialized(
        self: Scheduler,
        envelope: PipelineControlEnvelope,
    ) -> None:
        payload = envelope.payload or {}
        rid = str(payload["rid"])
        request_generation = int(payload["request_generation"])
        entry = self._pp_vpp_prefix_registry.get(rid, request_generation)
        if entry is None:
            return
        end = int(payload["end"])
        materialized_end = self._pp_vpp_prefix_registry.mark_materialized(
            rid,
            request_generation,
            int(payload["stage_id"]),
            end,
        )
        if materialized_end >= end and entry.committed_end < end:
            self._pp_vpp_prefix_registry.commit(
                rid,
                request_generation,
                end,
            )

    def _pp_vpp_advance_prefix_mapping(
        self: Scheduler,
        rid: str,
        request_generation: int,
        end: int,
    ) -> None:
        req = self._pp_vpp_find_req(rid)
        if req is None or not req.kv.holds_kv:
            return
        req.prefix_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :end
        ].to(dtype=torch.int64, copy=True)

    def _pp_vpp_prepare_wavefront_batch(
        self: Scheduler,
        batch_seq: int,
        slot_batch_seqs: List[Optional[int]],
    ) -> Optional[ScheduleBatch]:
        slot_id = batch_seq % len(self.mbs)
        if slot_batch_seqs[slot_id] is not None:
            raise RuntimeError(
                f"VPP wavefront slot {slot_id} is still owned by batch "
                f"{slot_batch_seqs[slot_id]}"
            )

        self.running_batch = self.running_mbs[slot_id]
        self.last_batch = self.last_mbs[slot_id]
        self.process_prefill_chunk(
            last_batch=self.last_batch,
            running_batch=self.running_batch,
        )
        prefill_plan = self.get_new_batch_prefill(self.running_batch)
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
            prefill_plan.batch_to_run
        )
        if batch is not None:
            reqs = getattr(batch, "reqs", None)
            if reqs is None:
                reqs = (
                    []
                    if getattr(batch, "chunked_req", None) is None
                    else [batch.chunked_req]
                )
            batch.disagg_prefill_chunk_end_by_rid = {
                req.rid: min(req.extend_range.end, len(req.origin_input_ids))
                for req in reqs
                if req.extend_range is not None
            }
            for req in reqs:
                req.vpp_prefix_limit = None
        self.running_batch = prefill_plan.running_batch
        self.running_mbs[slot_id] = self.running_batch
        self.mbs[slot_id] = batch
        self.mb_metadata[slot_id] = None
        slot_batch_seqs[slot_id] = batch_seq
        return batch

    def _pp_vpp_execute_wavefront_action(
        self: Scheduler,
        action: Optional[PipelineWavefrontAction],
        slot_batch_seqs: List[Optional[int]],
    ) -> Tuple[bool, List[P2PWork]]:
        if action is None:
            return False, []
        if slot_batch_seqs[action.slot_id] != action.batch_seq:
            raise RuntimeError(
                f"VPP wavefront batch {action.batch_seq} does not own "
                f"slot {action.slot_id}"
            )
        batch = self.mbs[action.slot_id]
        if batch is None:
            raise RuntimeError(
                f"VPP action for batch {action.batch_seq} found an empty "
                f"slot {action.slot_id}"
            )

        self.running_batch = self.running_mbs[action.slot_id]
        self.last_batch = self.last_mbs[action.slot_id]
        self.cur_batch_for_debug = batch
        pp_proxy_tensors = None
        if action.stage_id > 0:
            pp_proxy_tensors = self._pp_recv_vpp_proxy_tensors(
                first_visit=action.stage_id < self.ps.pp_size,
                expected_batch_seq=action.batch_seq,
                expected_stage_id=action.stage_id,
            )
        if self.enable_staging and action.stage_id < self.ps.pp_size:
            self.maybe_prefetch_staging_for_batch(batch)
        _, event, send_work = self._pp_launch_vpp_stage(
            action,
            batch,
            pp_proxy_tensors,
            self.mb_metadata,
            self.last_rank_comm_queue,
        )
        if not hasattr(self, "_pp_vpp_stage_events"):
            self._pp_vpp_stage_events = {}
        self._pp_vpp_stage_events[(action.batch_seq, action.stage_id)] = event
        return True, send_work

    def _pp_vpp_take_local_completion(
        self: Scheduler,
        batch_seq: int,
        last_rank_comm_queue: deque,
    ) -> Dict[str, object]:
        output_event, output_proxy = last_rank_comm_queue.popleft()
        output_event.synchronize()
        output_batch_seq = int(output_proxy.tensors.get("vpp_batch_seq", -1))
        if output_batch_seq != batch_seq:
            raise RuntimeError(
                f"VPP completion mismatch: expected {batch_seq}, got {output_batch_seq}"
            )
        return {
            key: value.to("cpu") if isinstance(value, torch.Tensor) else value
            for key, value in output_proxy.tensors.items()
        }

    def _pp_vpp_finalize_rank_local_batch(
        self: Scheduler,
        batch_seq: int,
        slot_batch_seqs: List[Optional[int]],
        output_tensors: Dict[str, object],
        release_slot: bool = True,
    ) -> bool:
        slot_id = batch_seq % len(self.mbs)
        if slot_batch_seqs[slot_id] != batch_seq:
            raise RuntimeError(
                f"VPP completion for batch {batch_seq} found slot "
                f"{slot_id} owned by {slot_batch_seqs[slot_id]}"
            )
        batch = self.mbs[slot_id]
        if batch is None:
            slot_batch_seqs[slot_id] = None
            return False
        metadata = self.mb_metadata[slot_id]
        if metadata is None:
            raise RuntimeError(
                f"VPP batch {batch_seq} completed without pipeline metadata"
            )
        for req in batch.reqs:
            request_generation = int(getattr(req, "session_generation", None) or 0)
            entry = self._pp_vpp_prefix_registry.get(
                req.rid,
                request_generation,
            )
            if (
                entry is not None
                and entry.committed_end >= entry.planned_end
                and not entry.locked
            ):
                req.skip_radix_cache_insert = getattr(
                    req,
                    "_vpp_original_skip_radix_cache_insert",
                    False,
                )
        with self.copy_stream_ctx:
            self.copy_stream.wait_stream(self.schedule_stream)
            output_tensors = {
                key: (
                    value.to(self.device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in output_tensors.items()
            }
            batch_result = self._pp_prep_batch_result(
                batch,
                metadata,
                PPProxyTensors(output_tensors),
            )
            d2h_event = self.device_module.Event()
            d2h_event.record(self.device_module.current_stream())
        d2h_event.synchronize()
        self._pp_process_batch_result(batch, batch_result)
        if getattr(batch, "contains_last_prefill_chunk", False):
            for req in batch.reqs:
                self._pp_vpp_prefix_registry.release(
                    req.rid,
                    int(getattr(req, "session_generation", None) or 0),
                )
        self.last_mbs[slot_id] = batch
        self.mbs[slot_id] = None
        self.mb_metadata[slot_id] = None
        if release_slot:
            slot_batch_seqs[slot_id] = None
        return True

    def _event_loop_pp_disagg_prefill_vpp(self: Scheduler):
        return self._event_loop_pp_disagg_prefill_vpp_rank_local()

    def _event_loop_pp_disagg_prefill_vpp_rank_local(self: Scheduler):
        self.init_pp_loop_state()
        # #region debug-point H1-H3:prefix-timeline
        self._pp_vpp_timeline_enabled = os.getenv("SGLANG_VPP_TIMELINE") == "1"
        timeline_gate_state = None
        # #endregion
        max_inflight = self._pp_vpp_max_inflight()
        rank_schedule = PipelineRankSchedule(
            physical_rank=self.ps.pp_rank,
            physical_size=self.ps.pp_size,
            virtual_stages=get_parallel().pp_virtual_stages,
            max_inflight=max_inflight,
            prefill_burst_size=get_parallel().pp_vpp_prefill_burst_size,
        )
        slot_batch_seqs: List[Optional[int]] = [None] * max_inflight
        self._pp_vpp_slot_batch_seqs = slot_batch_seqs
        self._pp_vpp_ready_proxies = {}
        self._pp_vpp_arrivals = deque()
        self._pp_vpp_pending_recv = None
        self._pp_vpp_pending_control_recv = None
        self._pp_vpp_control_outbox = deque()
        self._pp_vpp_stage_events = {}
        self._pp_vpp_pending_materialized = deque()
        self._pp_vpp_pending_replica_updates = deque()
        self._pp_vpp_batch_replicas = defaultdict(dict)
        self._pp_vpp_prefix_registry = PipelinePrefixRegistry()
        self._pp_vpp_replica_registry = PipelineReplicaRegistry()
        self._pp_vpp_control_layout_digest = self._pp_vpp_layout_digest()
        is_epoch_source = self.world_group.rank_in_group == 0
        self._pp_vpp_runtime_epoch = self.world_group.broadcast_object(
            time.time_ns() if is_epoch_source else None,
            src=0,
        )
        self._pp_vpp_start_receiver()
        self._pp_vpp_start_control_receiver()

        activation_send_work = deque()
        control_send_work = deque()
        max_control_sends = max_inflight * rank_schedule.logical_size
        token_budget = max(max_prefill_buffer_tokens(), 1)
        hidden_size = int(self.model_config.hidden_size)
        hc_mult = int(getattr(self.model_config.hf_config, "hc_mult", 1))
        activation_high = max_inflight * token_budget * hidden_size * hc_mult * 2
        resource_gate = PipelineResourceGate(
            ranks=range(self.ps.pp_size),
            activation_low_watermark=activation_high // 2,
            activation_high_watermark=activation_high,
            max_pending_sends=max_inflight,
        )
        last_resource_snapshot = None
        next_batch_seq = 0
        pending_admit = None
        pending_remote_admits = deque()
        pending_bootstrap_applies = deque()
        bootstrap_round_active = False
        transfer_round_active = False
        pending_first_pass: Dict[int, set[int]] = defaultdict(set)
        pending_chunk_batches = set()
        tick = 0
        stall_last_progress_at = time.monotonic()
        stall_last_progress_tick = 0
        stall_last_progress_event = "init"
        stall_last_log_at = 0.0

        def mark_progress(event: str) -> None:
            nonlocal stall_last_progress_at
            nonlocal stall_last_progress_tick
            nonlocal stall_last_progress_event
            stall_last_progress_at = time.monotonic()
            stall_last_progress_tick = tick
            stall_last_progress_event = event

        def control_extras(
            envelope: PipelineControlEnvelope,
            wire: Dict[str, object],
        ) -> Dict[str, object]:
            keys = envelope.to_dict().keys()
            return {key: value for key, value in wire.items() if key not in keys}

        def forward_control(
            envelope: PipelineControlEnvelope,
            wire: Dict[str, object],
        ) -> None:
            self._pp_vpp_queue_control(
                envelope.forwarded(),
                control_extras(envelope, wire),
            )

        def try_apply_bootstrap(
            envelope: PipelineControlEnvelope,
            wire: Dict[str, object],
            remaining_good: set[str],
            remaining_bad: set[str],
        ) -> bool:
            applied = self._pp_vpp_try_apply_bootstrap_status(
                envelope.payload["prefix_boundaries"],
                remaining_good,
                remaining_bad,
            )
            if not applied:
                return False
            forward_control(envelope, wire)
            return True

        def replica_identity(payload: Dict[str, object]) -> PipelineReplicaIdentity:
            return PipelineReplicaIdentity(
                content_id=str(payload["content_id"]),
                source_id=int(payload["source_id"]),
                format_version=int(payload["format_version"]),
                consumer_rank=int(payload["consumer_rank"]),
            )

        def apply_cache_update(payload: Dict[str, object]) -> None:
            identity = replica_identity(payload)
            generation = int(payload["residency_generation"])
            start = int(payload["start"])
            end = int(payload["end"])
            state = self._pp_vpp_replica_registry.get(identity)
            if (
                state is not None
                and state.residency_generation == generation
                and state.valid_end >= end
            ):
                return
            self._pp_vpp_replica_registry.install(
                identity,
                generation,
                start,
                end,
            )

        def flush_replica_updates() -> None:
            while self._pp_vpp_pending_replica_updates:
                (
                    batch_seq,
                    stage_id,
                    identity,
                    generation,
                    start,
                    end,
                ) = self._pp_vpp_pending_replica_updates[0]
                event = self._pp_vpp_stage_events[(batch_seq, stage_id)]
                query = getattr(event, "query", None)
                if query is not None and not query():
                    return
                self._pp_vpp_pending_replica_updates.popleft()
                payload = {
                    "content_id": identity.content_id,
                    "source_id": identity.source_id,
                    "format_version": identity.format_version,
                    "consumer_rank": identity.consumer_rank,
                    "residency_generation": generation,
                    "start": start,
                    "end": end,
                }
                apply_cache_update(payload)
                self._pp_vpp_replica_registry.lock(
                    identity,
                    end,
                    owner_id=batch_seq,
                )
                self._pp_vpp_batch_replicas[batch_seq][identity] = (
                    generation,
                    end,
                )
                self._pp_vpp_queue_control(
                    self._pp_vpp_new_control(
                        PipelineControlKind.CACHE_UPDATE,
                        batch_seq=batch_seq,
                        payload=payload,
                    ).forwarded()
                )
                if not any(
                    pending[0] == batch_seq and pending[1] == stage_id
                    for pending in self._pp_vpp_pending_replica_updates
                ) and not any(
                    pending[0] == batch_seq and pending[1] == stage_id
                    for pending in self._pp_vpp_pending_materialized
                ):
                    self._pp_vpp_stage_events.pop((batch_seq, stage_id), None)

        def evict_batch_replicas(batch_seq: int, batch: ScheduleBatch) -> None:
            if not getattr(batch, "contains_last_prefill_chunk", False):
                return
            replicas = self._pp_vpp_batch_replicas.pop(batch_seq, {})
            for identity, (generation, end) in replicas.items():
                state = self._pp_vpp_replica_registry.get(identity)
                if state is None or state.residency_generation != generation:
                    continue
                if state.locked_until:
                    self._pp_vpp_replica_registry.unlock(
                        identity,
                        end,
                        owner_id=batch_seq,
                    )
                if state.locked_until:
                    continue
                self._pp_vpp_replica_registry.evict(identity, generation)
                for tracked in self._pp_vpp_batch_replicas.values():
                    tracked.pop(identity, None)
                self._pp_vpp_queue_control(
                    self._pp_vpp_new_control(
                        PipelineControlKind.EVICT_ACK,
                        batch_seq=batch_seq,
                        payload={
                            "content_id": identity.content_id,
                            "source_id": identity.source_id,
                            "format_version": identity.format_version,
                            "consumer_rank": identity.consumer_rank,
                            "residency_generation": generation,
                        },
                    ).forwarded()
                )

        def flush_materialized() -> None:
            while self._pp_vpp_pending_materialized:
                (
                    batch_seq,
                    stage_id,
                    rid,
                    request_generation,
                    end,
                ) = self._pp_vpp_pending_materialized[0]
                event = self._pp_vpp_stage_events[(batch_seq, stage_id)]
                query = getattr(event, "query", None)
                if query is not None and not query():
                    return
                self._pp_vpp_pending_materialized.popleft()
                materialized = self._pp_vpp_new_control(
                    PipelineControlKind.PREFIX_MATERIALIZED,
                    batch_seq=batch_seq,
                    payload={
                        "rid": rid,
                        "request_generation": request_generation,
                        "stage_id": stage_id,
                        "end": end,
                    },
                )
                self._pp_vpp_apply_prefix_materialized(materialized)
                self._pp_vpp_queue_control(materialized.forwarded())
                if not any(
                    pending[0] == batch_seq and pending[1] == stage_id
                    for pending in self._pp_vpp_pending_materialized
                ) and not any(
                    pending[0] == batch_seq and pending[1] == stage_id
                    for pending in self._pp_vpp_pending_replica_updates
                ):
                    self._pp_vpp_stage_events.pop((batch_seq, stage_id), None)

        def handle_control(
            envelope: PipelineControlEnvelope,
            wire: Dict[str, object],
        ) -> None:
            nonlocal pending_admit, bootstrap_round_active, transfer_round_active
            payload = envelope.payload or {}
            returned_to_source = (
                envelope.source_rank == self.ps.pp_rank
                and envelope.hops >= self.ps.pp_size
            )
            # #region debug-point H1-H2:control-arrival
            if envelope.kind in (
                PipelineControlKind.ADMIT,
                PipelineControlKind.FIRST_PASS_DONE,
                PipelineControlKind.PREFIX_COMMIT,
                PipelineControlKind.COMPLETION,
            ):
                self._pp_vpp_timeline(
                    "control",
                    kind=envelope.kind.value,
                    batch=envelope.batch_seq,
                    source=envelope.source_rank,
                    returned=returned_to_source,
                )
            # #endregion

            if envelope.kind == PipelineControlKind.RESOURCE:
                if self.ps.pp_rank == 0:
                    resource_gate.update(
                        envelope.source_rank,
                        PipelineResourceSnapshot(**payload),
                    )
                else:
                    forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.REQUEST:
                if returned_to_source:
                    return
                requests = payload.get("requests") or ()
                if requests:
                    self.process_input_requests(list(requests))
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.BOOTSTRAP_STATUS:
                phase = str(payload["phase"])
                if phase == "collect":
                    if returned_to_source:
                        final_status = self._pp_vpp_bootstrap_apply_status(payload)
                        if not final_status[0] and not final_status[1]:
                            bootstrap_round_active = False
                            return
                        apply_envelope = self._pp_vpp_new_control(
                            PipelineControlKind.BOOTSTRAP_STATUS,
                            payload={
                                "phase": "apply",
                                "good": final_status[0],
                                "bad": final_status[1],
                                "prefix_boundaries": {
                                    rid: int(payload["prefix_boundaries"][rid])
                                    for rid in final_status[0]
                                },
                            },
                        )
                        apply_wire = apply_envelope.to_dict()
                        remaining_good = set(final_status[0])
                        remaining_bad = set(final_status[1])
                        if not try_apply_bootstrap(
                            apply_envelope,
                            apply_wire,
                            remaining_good,
                            remaining_bad,
                        ):
                            pending_bootstrap_applies.append(
                                (
                                    apply_envelope,
                                    apply_wire,
                                    remaining_good,
                                    remaining_bad,
                                )
                            )
                        return
                    local_good, local_bad = self.get_rids(
                        self.disagg_prefill_bootstrap_queue.queue,
                        True,
                        [KVPoll.WaitingForInput],
                        [KVPoll.Failed],
                    )
                    aborted = {
                        req.rid
                        for req in self.disagg_prefill_bootstrap_queue.queue
                        if isinstance(req.finished_reason, FINISH_ABORT)
                    }
                    local_good, local_bad = self._route_aborts_to_bad(
                        local_good,
                        local_bad,
                        aborted,
                    )
                    self._pp_vpp_merge_bootstrap_status(
                        payload,
                        local_good,
                        local_bad,
                    )
                    forward_control(envelope, wire)
                    return
                if phase != "apply":
                    raise RuntimeError(f"invalid bootstrap phase {phase}")
                if returned_to_source:
                    bootstrap_round_active = False
                    return
                remaining_good = set(payload["good"])
                remaining_bad = set(payload["bad"])
                if not try_apply_bootstrap(
                    envelope,
                    wire,
                    remaining_good,
                    remaining_bad,
                ):
                    pending_bootstrap_applies.append(
                        (
                            envelope,
                            wire,
                            remaining_good,
                            remaining_bad,
                        )
                    )
                return

            if envelope.kind == PipelineControlKind.TRANSFER_STATUS:
                phase = str(payload["phase"])
                if phase == "collect":
                    if returned_to_source:
                        final_rids = sorted(set(payload["rids"]))
                        self.process_disagg_prefill_inflight_queue(final_rids)
                        self._pp_vpp_queue_control(
                            self._pp_vpp_new_control(
                                PipelineControlKind.TRANSFER_STATUS,
                                payload={"phase": "apply", "rids": final_rids},
                            ).forwarded()
                        )
                        return
                    local_rids = self.get_rids(
                        self.disagg_prefill_inflight_queue,
                        True,
                        [KVPoll.Success, KVPoll.Failed],
                    )
                    payload["rids"] = sorted(
                        set(payload["rids"]).intersection(local_rids)
                    )
                    forward_control(envelope, wire)
                    return
                if phase != "apply":
                    raise RuntimeError(f"invalid transfer phase {phase}")
                if returned_to_source:
                    transfer_round_active = False
                    return
                self.process_disagg_prefill_inflight_queue(list(payload["rids"]))
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.ADMIT:
                if returned_to_source:
                    errors = payload.get("errors") or ()
                    if errors:
                        raise RuntimeError(f"VPP admission failed: {errors}")
                    rank_schedule.admit(envelope.batch_seq)
                    pending_admit = None
                    return
                batch = self._pp_vpp_prepare_wavefront_batch(
                    envelope.batch_seq,
                    slot_batch_seqs,
                )
                local_manifest = (
                    None
                    if batch is None
                    else self._pp_vpp_batch_manifest(
                        envelope.batch_seq,
                        envelope.slot_id,
                        batch,
                    )
                )
                expected_manifest = payload.get("manifest")
                if batch is None:
                    self.mbs[envelope.slot_id] = None
                    self.mb_metadata[envelope.slot_id] = None
                    slot_batch_seqs[envelope.slot_id] = None
                    pending_remote_admits.append((envelope, envelope.to_dict()))
                    return
                if local_manifest != expected_manifest:
                    payload.setdefault("errors", []).append(
                        (
                            self.ps.pp_rank,
                            expected_manifest,
                            local_manifest,
                        )
                    )
                self._pp_vpp_register_prefix_batch(batch)
                rank_schedule.admit(envelope.batch_seq)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.FIRST_PASS_DONE:
                if self.ps.pp_rank == 0:
                    pending_first_pass[envelope.batch_seq].add(envelope.source_rank)
                else:
                    forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.PREFIX_COMMIT:
                rid = str(payload["rid"])
                request_generation = int(payload["request_generation"])
                end = int(payload["end"])
                if returned_to_source:
                    pending_chunk_batches.discard(envelope.batch_seq)
                    return
                self._pp_vpp_advance_prefix_mapping(
                    rid,
                    request_generation,
                    end,
                )
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.PREFIX_MATERIALIZED:
                if returned_to_source:
                    return
                self._pp_vpp_apply_prefix_materialized(envelope)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.CACHE_UPDATE:
                if returned_to_source:
                    return
                apply_cache_update(payload)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.EVICT_ACK:
                if returned_to_source:
                    return
                identity = replica_identity(payload)
                state = self._pp_vpp_replica_registry.get(identity)
                generation = int(payload["residency_generation"])
                if (
                    state is not None
                    and state.residency_generation == generation
                    and not state.locked_until
                ):
                    self._pp_vpp_replica_registry.evict(identity, generation)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.CANCEL:
                if returned_to_source:
                    return
                rid = str(payload["rid"])
                req = self._pp_vpp_find_req(rid)
                if req is not None:
                    self._pending_chunked_abort_req = req
                    self.process_pending_chunked_abort()
                rank_schedule.cancel(envelope.batch_seq)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.COMPLETION:
                if returned_to_source:
                    slot_id = envelope.batch_seq % max_inflight
                    slot_batch_seqs[slot_id] = None
                    rank_schedule.retire(envelope.batch_seq)
                    pending_chunk_batches.discard(envelope.batch_seq)
                    return
                output_tensors = control_extras(envelope, wire)
                rank_schedule.mark_completed(envelope.batch_seq)
                batch = self.mbs[envelope.batch_seq % max_inflight]
                if batch is not None:
                    evict_batch_replicas(envelope.batch_seq, batch)
                self._pp_vpp_finalize_rank_local_batch(
                    envelope.batch_seq,
                    slot_batch_seqs,
                    output_tensors,
                )
                rank_schedule.retire(envelope.batch_seq)
                pending_chunk_batches.discard(envelope.batch_seq)
                forward_control(envelope, wire)
                return

            raise RuntimeError(f"unsupported VPP control kind {envelope.kind}")

        while True:
            server_is_idle = True
            while self._pp_vpp_poll_receiver_tp_consensus():
                mark_progress("activation")
            while self._pp_vpp_arrivals:
                batch_seq, stage_id = self._pp_vpp_arrivals.popleft()
                rank_schedule.mark_ready(batch_seq, stage_id)
                # #region debug-point H2:activation-ready
                self._pp_vpp_timeline(
                    "activation_ready", batch=batch_seq, stage=stage_id
                )
                # #endregion
            while True:
                control_message = self._pp_vpp_poll_control_receiver_tp_broadcast()
                if control_message is None:
                    break
                handle_control(*control_message)
                if control_message[0].kind not in (
                    PipelineControlKind.RESOURCE,
                    PipelineControlKind.BOOTSTRAP_STATUS,
                    PipelineControlKind.TRANSFER_STATUS,
                ):
                    mark_progress(f"control:{control_message[0].kind.value}")
            if pending_remote_admits:
                envelope, wire = pending_remote_admits.popleft()
                handle_control(envelope, wire)
            if pending_bootstrap_applies:
                apply_args = pending_bootstrap_applies.popleft()
                if not try_apply_bootstrap(*apply_args):
                    pending_bootstrap_applies.append(apply_args)

            self._pp_vpp_reap_send_work(activation_send_work)
            self._pp_vpp_flush_control_outbox(
                control_send_work,
                max_control_sends,
            )

            snapshot = self._pp_vpp_resource_snapshot(activation_send_work)
            if snapshot != last_resource_snapshot:
                last_resource_snapshot = snapshot
                if self.ps.pp_rank == 0:
                    resource_gate.update(0, snapshot)
                else:
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.RESOURCE,
                            payload={
                                "request_slots": snapshot.request_slots,
                                "kv_tokens": snapshot.kv_tokens,
                                "activation_bytes": snapshot.activation_bytes,
                                "pending_sends": snapshot.pending_sends,
                                "metadata_slots": snapshot.metadata_slots,
                            },
                        )
                    )

            if self.pp_group.is_first_rank:
                recv_reqs = self.ingest_requests()
                if recv_reqs:
                    request_snapshot = pickle.loads(pickle.dumps(tuple(recv_reqs)))
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.REQUEST,
                            payload={"requests": request_snapshot},
                        ).forwarded()
                    )

                start_bootstrap_round = all(
                    self.attn_tp_group.all_gather_object(
                        not bootstrap_round_active
                        and resource_gate.can_bootstrap()
                        and bool(self.disagg_prefill_bootstrap_queue.queue)
                    )
                )
                if start_bootstrap_round:
                    good, bad = self.get_rids(
                        self.disagg_prefill_bootstrap_queue.queue,
                        True,
                        [KVPoll.WaitingForInput],
                        [KVPoll.Failed],
                    )
                    aborted = {
                        req.rid
                        for req in self.disagg_prefill_bootstrap_queue.queue
                        if isinstance(req.finished_reason, FINISH_ABORT)
                    }
                    good, bad = self._route_aborts_to_bad(good, bad, aborted)
                    metadata_slots = min(
                        self.attn_tp_group.all_gather_object(
                            int(
                                self.req_to_metadata_buffer_idx_allocator.available_size()
                            )
                        )
                    )
                    bootstrap_round_active = True
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.BOOTSTRAP_STATUS,
                            payload={
                                "phase": "collect",
                                "good": sorted(good),
                                "bad": sorted(bad),
                                "prefix_boundaries": (
                                    self._pp_vpp_bootstrap_prefix_boundaries(good)
                                ),
                                "metadata_slots": metadata_slots,
                            },
                        ).forwarded()
                    )

                start_transfer_round = all(
                    self.attn_tp_group.all_gather_object(
                        not transfer_round_active
                        and bool(self.disagg_prefill_inflight_queue)
                    )
                )
                if start_transfer_round:
                    terminal_rids = self.get_rids(
                        self.disagg_prefill_inflight_queue,
                        True,
                        [KVPoll.Success, KVPoll.Failed],
                    )
                    transfer_round_active = True
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.TRANSFER_STATUS,
                            payload={
                                "phase": "collect",
                                "rids": sorted(terminal_rids),
                            },
                        ).forwarded()
                    )

                pending_abort = self._pending_chunked_abort_req
                self.process_pending_chunked_abort()
                if pending_abort is not None:
                    cancelled_batches = [
                        slot_batch_seqs[slot_id]
                        for slot_id, batch in enumerate(self.mbs)
                        if batch is not None
                        and any(req.rid == pending_abort.rid for req in batch.reqs)
                    ]
                    if not cancelled_batches:
                        cancelled_batches = [-1]
                    for batch_seq in cancelled_batches:
                        rank_schedule.cancel(batch_seq)
                        self._pp_vpp_queue_control(
                            self._pp_vpp_new_control(
                                PipelineControlKind.CANCEL,
                                batch_seq=batch_seq,
                                payload={"rid": pending_abort.rid},
                            ).forwarded()
                        )

                required_activation = token_budget * hidden_size * hc_mult * 2
                continuation = (
                    self.chunked_req is not None and self.chunked_req.kv.holds_kv
                )
                # #region debug-point H1:admission-gates
                if self._pp_vpp_timeline_active():
                    gate_state = (
                        self.profiler_manager.profile_id,
                        next_batch_seq,
                        pending_admit,
                        bootstrap_round_active,
                        tuple(sorted(pending_chunk_batches)),
                        rank_schedule.can_admit(next_batch_seq),
                        resource_gate.can_admit(
                            required_request_slots=0 if continuation else 1,
                            required_kv_tokens=token_budget,
                            required_activation_bytes=required_activation,
                        ),
                        continuation,
                        len(self.waiting_queue),
                    )
                    if gate_state != timeline_gate_state:
                        self._pp_vpp_timeline(
                            "admission_gate",
                            next_batch=next_batch_seq,
                            pending_admit=pending_admit,
                            bootstrap=bootstrap_round_active,
                            prefix_batches=gate_state[4],
                            slot_ok=gate_state[5],
                            resources_ok=gate_state[6],
                            continuation=continuation,
                            waiting=gate_state[8],
                        )
                        timeline_gate_state = gate_state
                else:
                    timeline_gate_state = None
                # #endregion
                if (
                    pending_admit is None
                    and not bootstrap_round_active
                    and not pending_chunk_batches
                    and rank_schedule.can_admit(next_batch_seq)
                    and resource_gate.can_admit(
                        required_request_slots=0 if continuation else 1,
                        required_kv_tokens=token_budget,
                        required_activation_bytes=required_activation,
                    )
                ):
                    slot_id = next_batch_seq % max_inflight
                    batch = self._pp_vpp_prepare_wavefront_batch(
                        next_batch_seq,
                        slot_batch_seqs,
                    )
                    if batch is None:
                        self.mbs[slot_id] = None
                        self.mb_metadata[slot_id] = None
                        slot_batch_seqs[slot_id] = None
                    else:
                        self._pp_vpp_register_prefix_batch(batch)
                        manifest = self._pp_vpp_batch_manifest(
                            next_batch_seq,
                            slot_id,
                            batch,
                        )
                        envelope = self._pp_vpp_new_control(
                            PipelineControlKind.ADMIT,
                            batch_seq=next_batch_seq,
                            slot_id=slot_id,
                            payload={
                                "manifest": manifest,
                                "errors": [],
                            },
                        )
                        self._pp_vpp_queue_control(envelope.forwarded())
                        pending_admit = next_batch_seq
                        if batch.chunked_req is not None:
                            pending_chunk_batches.add(next_batch_seq)
                        # #region debug-point H1-H2:admit-enqueue
                        self._pp_vpp_timeline(
                            "admit_queued",
                            batch=next_batch_seq,
                            prefix_pending=batch.chunked_req is not None,
                        )
                        # #endregion
                        next_batch_seq += 1
                        mark_progress("admission")
                        server_is_idle = False
            else:
                self.process_pending_chunked_abort()

            for batch_seq, ranks in tuple(pending_first_pass.items()):
                if len(ranks) != self.ps.pp_size:
                    continue
                batch = self.mbs[batch_seq % max_inflight]
                req = None if batch is None else batch.chunked_req
                if req is not None:
                    request_generation = int(
                        getattr(req, "session_generation", None) or 0
                    )
                    end = batch.disagg_prefill_chunk_end_by_rid[req.rid]
                    self._pp_vpp_advance_prefix_mapping(
                        req.rid,
                        request_generation,
                        end,
                    )
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.PREFIX_COMMIT,
                            batch_seq=batch_seq,
                            payload={
                                "rid": req.rid,
                                "request_generation": request_generation,
                                "end": end,
                            },
                        ).forwarded()
                    )
                    # #region debug-point H1:prefix-commit-enqueue
                    self._pp_vpp_timeline(
                        "prefix_commit_queued", batch=batch_seq, end=end
                    )
                    # #endregion
                else:
                    pending_chunk_batches.discard(batch_seq)
                pending_first_pass.pop(batch_seq, None)

            def action_ready(batch_seq: int, stage_id: int) -> bool:
                if (
                    stage_id < rank_schedule.logical_size - 1
                    and len(activation_send_work) >= max_inflight
                ):
                    return False
                if stage_id == 0:
                    return True
                return (batch_seq, stage_id) in self._pp_vpp_ready_proxies

            action = self._pp_vpp_select_tp_action(
                rank_schedule,
                tick,
                action_ready,
            )
            if action is not None:
                replica_restores = []
                if action.stage_id > 0:
                    proxy = self._pp_vpp_take_ready_proxy(action)
                    source_id = proxy.tensors.get("vpp_source_layer_id")
                    batch = self.mbs[action.slot_id]
                    if source_id is not None and batch is not None:
                        for req in batch.reqs:
                            end = batch.disagg_prefill_chunk_end_by_rid.get(req.rid)
                            if end is None:
                                continue
                            content_id = getattr(
                                req,
                                "_vpp_replica_content_id",
                                None,
                            )
                            if content_id is None:
                                content_id = hashlib.sha256(
                                    pickle.dumps(
                                        tuple(req.origin_input_ids),
                                        protocol=pickle.HIGHEST_PROTOCOL,
                                    )
                                ).hexdigest()
                                req._vpp_replica_content_id = content_id
                            identity = PipelineReplicaIdentity(
                                content_id=content_id,
                                source_id=int(source_id),
                                format_version=1,
                                consumer_rank=self.ps.pp_rank,
                            )
                            state = self._pp_vpp_replica_registry.get(identity)
                            generation = (
                                0 if state is None else state.residency_generation
                            )
                            missing = self._pp_vpp_replica_registry.missing_range(
                                identity,
                                generation,
                                end,
                            )
                            if missing is None:
                                self._pp_vpp_replica_registry.lock(
                                    identity,
                                    end,
                                    owner_id=action.batch_seq,
                                )
                                self._pp_vpp_batch_replicas[action.batch_seq][
                                    identity
                                ] = (generation, end)
                            else:
                                replica_restores.append(
                                    (identity, generation, *missing)
                                )
                    self._pp_tensor_dict_inbox["vpp_proxy"].appendleft(proxy.tensors)
                action_executed, send_work = self._pp_vpp_execute_wavefront_action(
                    action,
                    slot_batch_seqs,
                )
                if send_work:
                    activation_send_work.append(send_work)
                if action_executed:
                    mark_progress(f"stage:{action.stage_id}")
                    for identity, generation, start, end in replica_restores:
                        self._pp_vpp_pending_replica_updates.append(
                            (
                                action.batch_seq,
                                action.stage_id,
                                identity,
                                generation,
                                start,
                                end,
                            )
                        )
                    transition = rank_schedule.complete(action)
                    batch = self.mbs[action.slot_id]
                    if batch is not None:
                        for req in batch.reqs:
                            if req.extend_range is None:
                                continue
                            request_generation = int(
                                getattr(req, "session_generation", None) or 0
                            )
                            end = batch.disagg_prefill_chunk_end_by_rid[req.rid]
                            self._pp_vpp_pending_materialized.append(
                                (
                                    action.batch_seq,
                                    action.stage_id,
                                    req.rid,
                                    request_generation,
                                    end,
                                )
                            )
                    if transition.first_pass_done:
                        if self.ps.pp_rank == 0:
                            pending_first_pass[action.batch_seq].add(0)
                        else:
                            self._pp_vpp_queue_control(
                                self._pp_vpp_new_control(
                                    PipelineControlKind.FIRST_PASS_DONE,
                                    batch_seq=action.batch_seq,
                                ).forwarded()
                            )
                    if transition.batch_complete:
                        output_tensors = self._pp_vpp_take_local_completion(
                            action.batch_seq,
                            self.last_rank_comm_queue,
                        )
                        flush_replica_updates()
                        flush_materialized()
                        completion = self._pp_vpp_new_control(
                            PipelineControlKind.COMPLETION,
                            batch_seq=action.batch_seq,
                            slot_id=action.slot_id,
                        )
                        if batch is not None:
                            evict_batch_replicas(action.batch_seq, batch)
                        self._pp_vpp_finalize_rank_local_batch(
                            action.batch_seq,
                            slot_batch_seqs,
                            output_tensors,
                            release_slot=False,
                        )
                        self._pp_vpp_queue_control(
                            completion.forwarded(),
                            output_tensors,
                        )
                    server_is_idle = False

            flush_replica_updates()
            flush_materialized()

            self._pp_vpp_flush_control_outbox(
                control_send_work,
                max_control_sends,
            )
            now = time.monotonic()
            has_pending_work = (
                pending_admit is not None
                or bootstrap_round_active
                or bool(pending_chunk_batches)
                or bool(pending_bootstrap_applies)
                or bool(self.waiting_queue)
                or bool(self.disagg_prefill_bootstrap_queue.queue)
                or bool(self.disagg_prefill_inflight_queue)
                or bool(self._pp_vpp_control_outbox)
                or bool(control_send_work)
                or bool(activation_send_work)
                or rank_schedule.inflight_count > 0
            )
            if (
                self.ps.tp_rank == 0
                and has_pending_work
                and now - stall_last_progress_at >= 10
                and now - stall_last_log_at >= 10
            ):
                stall_last_log_at = now
                logger.warning(
                    "[VPP-STALL] no scheduler progress for %.1fs "
                    "tick=%s last_progress=%s@%s pp=%s "
                    "pending_admit=%s bootstrap_round=%s transfer_round=%s "
                    "pending_chunks=%s pending_first_pass=%s "
                    "slots=%s ready=%s running=%s "
                    "waiting=%s bootstrap=%s inflight=%s "
                    "metadata_slots=%s bootstrap_applies=%s "
                    "materialized=%s replica_updates=%s "
                    "control_outbox=%s control_sends=%s activation_sends=%s "
                    "ready_proxies=%s arrivals=%s resource_ranks=%s blocked_ranks=%s "
                    "local_resource=%s",
                    now - stall_last_progress_at,
                    tick,
                    stall_last_progress_event,
                    stall_last_progress_tick,
                    self.ps.pp_rank,
                    pending_admit,
                    bootstrap_round_active,
                    transfer_round_active,
                    sorted(pending_chunk_batches),
                    {
                        batch_seq: sorted(ranks)
                        for batch_seq, ranks in pending_first_pass.items()
                    },
                    rank_schedule.slot_batch_seqs,
                    rank_schedule.ready_tasks,
                    rank_schedule.running,
                    len(self.waiting_queue),
                    len(self.disagg_prefill_bootstrap_queue.queue),
                    len(self.disagg_prefill_inflight_queue),
                    self.req_to_metadata_buffer_idx_allocator.available_size(),
                    len(pending_bootstrap_applies),
                    len(self._pp_vpp_pending_materialized),
                    len(self._pp_vpp_pending_replica_updates),
                    len(self._pp_vpp_control_outbox),
                    len(control_send_work),
                    len(activation_send_work),
                    len(self._pp_vpp_ready_proxies),
                    len(self._pp_vpp_arrivals),
                    sorted(resource_gate._snapshots),
                    sorted(resource_gate._blocked),
                    snapshot,
                )
            if activation_send_work or control_send_work:
                server_is_idle = False
            if rank_schedule.inflight_count:
                server_is_idle = False
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.on_idle()
            tick += 1

    @DynamicGradMode()
    def event_loop_pp_disagg_decode(self: Scheduler):
        self.init_pp_loop_state()

        # PD additional state initialization
        rmbs = [None] * self.pp_loop_size
        pmbs = [None] * self.pp_loop_size
        tmbs = [None] * self.pp_loop_size
        consensus_retract_rids: Optional[List[str]] = None
        consensus_prealloc_rids: Optional[List[str]] = None
        release_rids: Optional[List[str]] = None  # consensus transferred rids
        send_retract_work = []
        send_prealloc_work = []
        send_transfer_work = []
        send_consensus_retract_work = []
        send_consensus_prealloc_work = []
        send_release_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = self.last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.ps.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_consensus_retract_rids = None
                next_consensus_prealloc_rids = None
                next_release_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.ingest_requests()

                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(self.send_req_work)

                # reaching consensus through PP ranks
                retract_rids = self._pp_pd_get_retract_ids(mb_id)
                rmbs[mb_id] = retract_rids
                self._pp_commit_comm_work(send_retract_work)

                prealloc_rids = self._pp_pd_get_prealloc_ids()
                pmbs[mb_id] = prealloc_rids
                self._pp_commit_comm_work(send_prealloc_work)

                transferred_rids = self._pp_pd_get_decode_transferred_ids()
                tmbs[mb_id] = transferred_rids
                self._pp_commit_comm_work(send_transfer_work)

                # get batch to run and proxy tensors if needed
                plan = self.get_next_disagg_decode_batch_to_run(
                    running_batch=self.running_batch
                )
                self.running_batch = plan.running_batch
                batch = plan.batch_to_run
                self.mbs[mb_id] = batch
                self.running_mbs[mb_id] = self.running_batch

                cur_batch: Optional[ScheduleBatch] = self.mbs[mb_id]
                self.cur_batch_for_debug = cur_batch
                if cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = None
                    if not cur_batch.forward_mode.is_prebuilt():
                        pp_proxy_tensors = self._pp_recv_proxy_tensors()

                # early send output if possible
                if get_parallel().pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self._pp_commit_comm_work(self.send_proxy_work)

                if cur_batch:
                    result, self.launch_event = self._pp_launch_batch(
                        mb_id,
                        cur_batch,
                        pp_proxy_tensors,
                        self.mb_metadata,
                        self.last_rank_comm_queue,
                    )

                if get_parallel().pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )

                # reach consensus on last rank and send to PP=0
                # otherwise, just pass along previous consensus
                send_consensus_retract_work, consensus_retract_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        rmbs,
                        next_first_rank_mb_id,
                        consensus_retract_rids,
                        retract_rids,
                    )
                )

                send_consensus_prealloc_work, consensus_prealloc_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        pmbs,
                        next_first_rank_mb_id,
                        consensus_prealloc_rids,
                        prealloc_rids,
                    )
                )

                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if get_disagg().disaggregation_decode_enable_offload_kvcache:
                    self.decode_offload_manager.check_offload_progress()

                if rmbs[next_mb_id] is not None:
                    next_consensus_retract_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_retract_rids = self.process_retract_queue(
                        next_consensus_retract_rids
                    )
                self._pp_commit_comm_work(send_consensus_retract_work)

                if pmbs[next_mb_id] is not None:
                    next_consensus_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_consensus_prealloc_rids = self.process_prealloc_queue(
                        next_consensus_prealloc_rids
                    )
                self._pp_commit_comm_work(send_consensus_prealloc_work)

                if tmbs[next_mb_id] is not None:
                    next_release_rids = self._pp_recv_pyobj_from_prev_stage()
                    next_release_rids = self.process_decode_transfer_queue(
                        next_release_rids
                    )
                self._pp_commit_comm_work(send_release_work)

                # post-process the coming microbatch
                if self.mbs[next_mb_id] is not None:
                    if not self.mbs[next_mb_id].forward_mode.is_prebuilt():
                        d2h_event.synchronize()
                        self._pp_process_batch_result(
                            self.mbs[next_mb_id],
                            next_batch_result,
                        )
                    self.last_mbs[next_mb_id] = self.mbs[next_mb_id]

                if not self.pp_group.is_last_rank:
                    self.send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_retract_work = self._pp_send_pyobj_to_next_stage(
                        retract_rids, async_send=True
                    )
                    send_prealloc_work = self._pp_send_pyobj_to_next_stage(
                        prealloc_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if cur_batch and not cur_batch.forward_mode.is_prebuilt():
                        self.device_module.current_stream().wait_event(
                            self.launch_event
                        )
                        self.send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                            msg_type="proxy",
                        )

                self.pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_retract_rids = next_consensus_retract_rids
                consensus_prealloc_rids = next_consensus_prealloc_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if get_disagg().disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)

            if server_is_idle and queue_size == 0:
                self.on_idle()

    def init_pp_loop_state(self: Scheduler):
        self.pp_loop_size: int = (
            self._pp_vpp_max_inflight()
            if self._pp_vpp_enabled()
            else self.ps.pp_size + get_parallel().pp_async_batch_depth
        )
        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        self.mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size
        self.pp_outputs: Optional[PPProxyTensors] = None
        self.last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]] = deque()

        self.send_req_work = []
        self.send_proxy_work = []
        self.send_output_work = []
        self.launch_event = None
        self._pp_tensor_dict_inbox: Dict[str, deque[Dict[str, torch.Tensor]]] = (
            defaultdict(deque)
        )

    def process_bootstrapped_queue(
        self: Scheduler, bootstrapped_rids: Optional[List[str]]
    ):
        # finished consensus bootstrapped reqs and prepare the waiting queue
        if bootstrapped_rids is not None:
            (
                good_consensus_bootstrapped_rids,
                bad_consensus_bootstrapped_rids,
            ) = bootstrapped_rids
            good_reqs, failed_reqs = (
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                    return_failed_reqs=True,
                    pp_good_rids=good_consensus_bootstrapped_rids,
                    pp_bad_rids=bad_consensus_bootstrapped_rids,
                )
            )
            self.waiting_queue.extend(good_reqs)
            return [[req.rid for req in good_reqs], [req.rid for req in failed_reqs]]
        return None

    def _pp_pd_get_bootstrapped_ids(self: Scheduler):
        # communicate pre-consensus bootstrapp reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the bootstrap reqs from the bootstrap queue
            good_bootstrapped_rids, bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
            prev_bootstrapped_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_bootstrapped_rids, prev_bad_bootstrapped_rids = (
                prev_bootstrapped_rids
            )
            curr_good_bootstrapped_rids, curr_bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                True,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_bootstrapped_rids = list(
                set(prev_good_bootstrapped_rids) & set(curr_good_bootstrapped_rids)
            )
            bad_bootstrapped_rids = list(
                set(prev_bad_bootstrapped_rids) | set(curr_bad_bootstrapped_rids)
            )
        # Route locally-aborted reqs through the bad-union consensus so every PP
        # rank flushes them in the same consensus round, regardless of when the
        # AbortReq reaches each rank and regardless of whether
        # disagg_kv_sender.abort() drives the poll to Failed (it is optional).
        aborted_rids = {
            req.rid
            for req in self.disagg_prefill_bootstrap_queue.queue
            if isinstance(req.finished_reason, FINISH_ABORT)
        }
        good_bootstrapped_rids, bad_bootstrapped_rids = self._route_aborts_to_bad(
            good_bootstrapped_rids, bad_bootstrapped_rids, aborted_rids
        )
        return [good_bootstrapped_rids, bad_bootstrapped_rids]

    def _pp_pd_get_prefill_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                True,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def _pp_pd_send_consensus_bootstrapped_ids(
        self: Scheduler,
        bmbs: List[List[str]],
        next_first_rank_mb_id: int,
        consensus_bootstrapped_rids: List[str],
        bootstrapped_rids: List[str],
    ):
        # 3 (Release): send the release rids from last stage to the first stage
        send_consensus_bootstrapped_work = []
        if self.pp_group.is_last_rank:
            if bmbs[next_first_rank_mb_id] is not None:
                consensus_bootstrapped_rids = bootstrapped_rids
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if consensus_bootstrapped_rids is not None:
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        return send_consensus_bootstrapped_work, consensus_bootstrapped_rids

    def _pp_pd_send_consensus_release_ids(
        self: Scheduler,
        tmbs: List[List[str]],
        next_first_rank_mb_id: int,
        release_rids: List[str],
        transferred_rids: List[str],
    ):
        send_release_work = []
        if self.pp_group.is_last_rank:
            if tmbs[next_first_rank_mb_id] is not None:
                release_rids = transferred_rids
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if release_rids is not None:
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        return send_release_work, release_rids

    def _pp_commit_comm_work(self: Scheduler, work: List[P2PWork]) -> None:
        for p2p_work in work:
            p2p_work.work.wait()
        work.clear()

    def _pp_commit_send_output_work_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
    ]:
        self._pp_commit_comm_work(work=self.send_output_work)
        (
            next_pp_outputs,
            next_batch_result,
            d2h_event,
            self.send_output_work,
        ) = self._pp_send_recv_and_preprocess_output_tensors(
            next_first_rank_mb_id,
            next_mb_id,
            self.mbs,
            self.mb_metadata,
            self.last_rank_comm_queue,
            self.pp_outputs,
        )
        return next_pp_outputs, next_batch_result, d2h_event

    def _pp_send_pyobj_to_next_stage(self: Scheduler, data, async_send: bool = False):
        p2p_work = []
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = (
                self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
            )
            p2p_work = point_to_point_pyobj(
                data,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                ((self.ps.pp_rank + 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                async_send=async_send,
            )
        return p2p_work

    def _pp_recv_pyobj_from_prev_stage(self: Scheduler):
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            dp_offset = (
                self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
            )
            data = point_to_point_pyobj(
                [],
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
                self.world_group.cpu_group,
                ((self.ps.pp_rank - 1) % self.ps.pp_size) * self.ps.tp_size + dp_offset,
                self.ps.pp_rank * self.ps.tp_size + dp_offset,
            )
        else:
            data = None

        data = attn_cp_tp_broadcast_pyobj(data)
        return data

    def _pp_prepare_tensor_dict(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
        }

        # Draft extend runs only on the last stage, but every rank needs its relayed
        # output to fill PD auxiliary buffers.
        draft_input = result.next_draft_input
        if draft_input is not None and draft_input.topk_p is not None:
            tensor_dict["draft_topk_p"] = draft_input.topk_p.contiguous()
            tensor_dict["draft_topk_index"] = draft_input.topk_index.contiguous()
            tensor_dict["draft_hidden_states"] = draft_input.hidden_states.contiguous()

        if batch.return_logprob:
            logprob_dict = get_logprob_dict_from_result(result)
            tensor_dict = {
                **tensor_dict,
                **logprob_dict,
            }
        auxiliary_output = (
            result.logits_output.auxiliary_device_output
            if result.logits_output is not None
            else None
        )
        add_auxiliary_output_to_pp_tensors(tensor_dict, auxiliary_output)
        return tensor_dict

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: Dict[str, torch.Tensor],
        async_send: bool = True,
        msg_type: str = "default",
        batch_p2p: bool = False,
        tag: int = 0,
    ):
        # Warn once if using default untyped messages
        if msg_type == "default":
            logger.warning_once(
                "PP send: using default untyped message. "
                "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
            )
        tensor_dict["__msg_type__"] = msg_type
        p2p_work = []
        pp_group = (
            self._pp_vpp_activation_group(self.ps.pp_rank)
            if msg_type == "vpp_proxy"
            else self.pp_group
        )
        p2p_work.extend(
            pp_group.send_tensor_dict(
                tensor_dict=tensor_dict,
                all_gather_group=(self.attn_tp_group),
                async_send=async_send,
                batch_p2p=batch_p2p,
                tag=tag,
            )
        )
        return p2p_work

    def _pp_recv_typed_dict(
        self: Scheduler,
        expected_kind: str = "default",
        all_gather_group: Optional = None,
        batch_p2p: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Receive a typed tensor dict, demultiplexing by msg_type.

        If a message of the wrong kind is received, it's stashed in the queue
        and we continue receiving until we get the expected kind.
        """
        if expected_kind in self._pp_tensor_dict_inbox:
            inbox_queue = self._pp_tensor_dict_inbox[expected_kind]
            if inbox_queue:
                return inbox_queue.popleft()

        while True:
            tensor_dict = self.pp_group.recv_tensor_dict(
                all_gather_group=all_gather_group,
                batch_p2p=batch_p2p,
            )
            received_kind = tensor_dict.get("__msg_type__", "default")
            if received_kind == expected_kind:
                if received_kind == "default":
                    logger.warning_once(
                        f"PP recv: got default untyped message. Content keys: {tensor_dict.keys()}"
                        "Consider adding msg_type='proxy' or 'output' to avoid recv conflicts."
                    )
                return tensor_dict
            else:
                logger.debug(
                    f"PP recv: expected {expected_kind}, got {received_kind}, stashing"
                )
                self._pp_tensor_dict_inbox[received_kind].append(tensor_dict)

    def _pp_recv_proxy_tensors(self: Scheduler) -> Optional[PPProxyTensors]:
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self._pp_recv_typed_dict(
                    expected_kind="proxy",
                    all_gather_group=(self.attn_tp_group),
                )
            )
        return pp_proxy_tensors

    def _pp_recv_vpp_proxy_tensors(
        self: Scheduler,
        *,
        first_visit: bool,
        expected_batch_seq: Optional[int] = None,
        expected_stage_id: Optional[int] = None,
    ) -> Optional[PPProxyTensors]:
        if first_visit and self.pp_group.is_first_rank:
            return None
        proxy = PPProxyTensors(
            self._pp_recv_typed_dict(
                expected_kind="vpp_proxy",
                all_gather_group=self.attn_tp_group,
                batch_p2p=True,
            )
        )
        if (
            expected_batch_seq is not None
            and int(proxy.tensors.get("vpp_batch_seq", -1)) != expected_batch_seq
        ):
            raise RuntimeError(
                "VPP activation batch mismatch: expected "
                f"{expected_batch_seq}, got {proxy.tensors.get('vpp_batch_seq')}"
            )
        if (
            expected_stage_id is not None
            and int(proxy.tensors.get("vpp_stage_id", -1)) != expected_stage_id
        ):
            raise RuntimeError(
                "VPP activation stage mismatch: expected "
                f"{expected_stage_id}, got {proxy.tensors.get('vpp_stage_id')}"
            )
        return proxy

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> Dict[str, torch.Tensor]:
        return self._pp_recv_typed_dict(
            expected_kind="output",
            all_gather_group=(self.attn_tp_group),
        )

    def _pp_make_skip_output_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: Optional[PPBatchMetadata],
    ):
        bs = len(batch.reqs)
        placeholder = torch.zeros(bs, dtype=torch.int64, device=self.device)
        # next_pp_outputs = None so non-last ranks skip forwarding
        # (pp_outputs is None gate). Placeholder carried in
        # batch_result.next_token_ids for process_batch_result_prefill.
        batch_result = GenerationBatchResult(
            logits_output=None,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=placeholder,
            can_run_cuda_graph=(
                mb_metadata.can_run_cuda_graph if mb_metadata else False
            ),
            skipped_output_comm=True,
        )
        d2h_event = self.device_module.Event()
        d2h_event.record(self.device_module.current_stream())
        return None, batch_result, d2h_event

    def _pp_prep_batch_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: PPBatchMetadata,
        pp_outputs: PPProxyTensors,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        logits_output = None
        extend_input_len_per_req = None
        extend_logprob_start_len_per_req = None

        if batch.return_logprob:
            (
                logits_output,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = get_logprob_from_pp_outputs(pp_outputs)
        if self.pp_group.is_first_rank:
            observer = self.tp_worker.model_runner.sampling_observer
            auxiliary_output = pop_auxiliary_output_from_pp_tensors(
                pp_outputs.tensors,
                observer,
            )
            if auxiliary_output is not None:
                if logits_output is None:
                    logits_output = LogitsProcessorOutput(next_token_logits=None)
                logits_output.auxiliary_device_output = auxiliary_output
        next_token_ids = pp_outputs["next_token_ids"].to(torch.int64)

        # Rebind the last stage's ring proposal as batch.spec_info so the PD result
        # processor sees the same object on every rank.
        next_draft_input = None
        if "draft_topk_p" in pp_outputs.tensors:
            from sglang.srt.speculative.eagle_info import EagleDraftInput

            next_draft_input = EagleDraftInput(
                topk_p=pp_outputs["draft_topk_p"],
                topk_index=pp_outputs["draft_topk_index"],
                hidden_states=pp_outputs["draft_hidden_states"],
                bonus_tokens=next_token_ids,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )
            batch.spec_info = next_draft_input

        # PP rank 0 also relays into output_tokens_buf so the next iter's
        # resolve_forward_inputs finds these tokens for the decode portion
        # of mixed-chunk batches (which gather via mix_running_indices).
        self.future_map.stash(
            batch.req_pool_indices,
            RelayPayload(
                bonus_tokens=next_token_ids,
                topk_p=None if next_draft_input is None else next_draft_input.topk_p,
                topk_index=(
                    None if next_draft_input is None else next_draft_input.topk_index
                ),
                hidden_states=(
                    None if next_draft_input is None else next_draft_input.hidden_states
                ),
            ),
        )
        batch.input_ids = None
        output_result = GenerationBatchResult(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=pp_outputs["next_token_ids"],
            next_draft_input=next_draft_input,
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            can_run_cuda_graph=mb_metadata.can_run_cuda_graph,
        )
        output_result.copy_auxiliary_output_to_cpu()
        return output_result

    def _pp_process_batch_result(
        self: Scheduler, batch: ScheduleBatch, output_result: GenerationBatchResult
    ):
        self.process_batch_result(batch, output_result)

    def _pp_send_output_to_next_stage(
        self: Scheduler,
        next_first_rank_mb_id: int,
        mbs: List[ScheduleBatch],
        last_rank_comm_queue: deque,
        pp_outputs: PPProxyTensors | None,
    ) -> List[P2PWork]:
        send_output_work = []
        if self.pp_group.is_last_rank:
            # send ready PP output to rank 0
            target = mbs[next_first_rank_mb_id]
            if target is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                if (
                    not target.forward_mode.is_prebuilt()
                    and not _pp_can_skip_output_comm(target)
                ):
                    self.device_module.current_stream().wait_event(q_event)
                    with torch.profiler.record_function("send_res_dict_to_next_stage"):
                        send_output_work = self._pp_send_dict_to_next_stage(
                            pp_outputs_to_send.tensors,
                            async_send=True,
                            msg_type="output",
                        )
        # send the outputs from the last round to let the next stage worker run post processing
        if not self.pp_group.is_last_rank:
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs.tensors,
                        async_send=True,
                        msg_type="output",
                    )
        return send_output_work

    def _pp_send_recv_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        mbs: List[ScheduleBatch],
        mb_metadata: List[PPBatchMetadata],
        last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
        List[P2PWork],
    ]:
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        # On CUDA, isend is async: it enqueues to the stream and returns,
        # so every rank can send first safely. On some backends isend is
        # effectively blocking and does not return until the peer posts a
        # matching recv; if every PP rank sends first, all ranks block
        # waiting for a receiver and the ring deadlocks. Order send/recv
        # by pp_rank parity (even: send->recv, odd: recv->send) so each
        # adjacent pair has one sender and one receiver posted at the
        # same time.

        # CUDA: send first
        # XPU: even ranks send first, odd ranks recv first.
        send_first = (not is_xpu()) or ((self.ps.pp_rank % 2) == 0)

        def _do_send():
            return self._pp_send_output_to_next_stage(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs,
            )

        def _do_recv():
            nonlocal next_pp_outputs, batch_result, d2h_event
            target = mbs[next_mb_id]
            if target is None or target.forward_mode.is_prebuilt():
                return
            if _pp_can_skip_output_comm(target):
                next_pp_outputs, batch_result, d2h_event = (
                    self._pp_make_skip_output_result(target, mb_metadata[next_mb_id])
                )
                return
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.schedule_stream)
                batch_result = self._pp_prep_batch_result(
                    target, mb_metadata[next_mb_id], next_pp_outputs
                )
                d2h_event = self.device_module.Event()
                d2h_event.record(self.device_module.current_stream())

        if send_first:
            send_output_work = _do_send()
            _do_recv()
        else:
            _do_recv()
            send_output_work = _do_send()

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        cur_batch: ScheduleBatch,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque,
    ):
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                set_time_batch(
                    cur_batch.reqs,
                    "set_run_batch_cpu_start_time",
                    trace_only=True,
                )
                result = self.run_batch(cur_batch, pp_proxy_tensors)
                set_time_batch(
                    cur_batch.reqs,
                    "set_run_batch_cpu_end_time",
                    trace_only=True,
                    attrs={"pp_mb_id": mb_id},
                )
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                event = self.device_module.Event()
                event.record(self.device_module.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(result, cur_batch)
                            ),
                        )
                    )
        return result, event

    def _pp_launch_vpp_batch(self: Scheduler, *args, **kwargs):
        raise RuntimeError("VPP2 batches must run through the ready-queue scheduler")

    def _pp_launch_vpp_stage(
        self: Scheduler,
        action: PipelineWavefrontAction,
        cur_batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque,
    ):
        if get_parallel().pp_virtual_stages != 2:
            raise RuntimeError("the VPP scheduler currently supports VPP2 only")
        if action.physical_rank != self.ps.pp_rank:
            raise RuntimeError(
                f"wavefront action for PP rank {action.physical_rank} "
                f"cannot run on PP rank {self.ps.pp_rank}"
            )
        if action.stage_id == 0:
            if pp_proxy_tensors is not None:
                raise RuntimeError("the first VPP stage must use local model inputs")
        elif pp_proxy_tensors is None:
            raise RuntimeError(
                f"logical stage {action.stage_id} requires an activation"
            )

        send_work = []
        # #region debug-point H2-H3:stage-launch
        stage_label = f"run_vpp_stage_{action.stage_id}"
        if self._pp_vpp_timeline_active():
            stage_label += f"/batch_{action.batch_seq}/slot_{action.slot_id}"
            self._pp_vpp_timeline(
                "stage_launch",
                batch=action.batch_seq,
                stage=action.stage_id,
                slot=action.slot_id,
                chunks=[
                    (
                        hashlib.sha256(req.rid.encode()).hexdigest()[:16],
                        req.extend_range.start,
                        cur_batch.disagg_prefill_chunk_end_by_rid.get(req.rid),
                    )
                    for req in cur_batch.reqs
                    if req.extend_range is not None
                ],
            )
        # #endregion
        with torch.profiler.record_function(stage_label):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                if action.stage_id < self.ps.pp_size:
                    set_time_batch(
                        cur_batch.reqs,
                        "set_run_batch_cpu_start_time",
                        trace_only=True,
                    )
                result = self.run_batch(cur_batch, pp_proxy_tensors)
                is_last_stage = action.stage_id == (
                    self.ps.pp_size * get_parallel().pp_virtual_stages - 1
                )
                if is_last_stage:
                    if result.pp_hidden_states_proxy_tensors is not None:
                        raise RuntimeError("the final VPP stage did not produce logits")
                else:
                    proxy = result.pp_hidden_states_proxy_tensors
                    if proxy is None:
                        raise RuntimeError(
                            "a non-final VPP stage produced no activation"
                        )
                    output_stage_id = int(proxy.tensors.get("vpp_stage_id", -1))
                    if output_stage_id != action.stage_id + 1:
                        raise RuntimeError(
                            "VPP stage produced an unexpected successor: "
                            f"stage {action.stage_id} produced {output_stage_id}"
                        )
                    proxy.tensors["vpp_batch_seq"] = action.batch_seq
                    proxy.tensors["vpp_generation"] = action.batch_seq // len(
                        mb_metadata
                    )
                    proxy.tensors["vpp_protocol_version"] = _VPP_PROTOCOL_VERSION
                    proxy.tensors["vpp_src_stage_id"] = action.stage_id
                    send_work = self._pp_send_dict_to_next_stage(
                        proxy.tensors,
                        async_send=True,
                        msg_type="vpp_proxy",
                        batch_p2p=True,
                        tag=_VPP_ACTIVATION_TAG,
                    )
                if action.stage_id >= self.ps.pp_size:
                    set_time_batch(
                        cur_batch.reqs,
                        "set_run_batch_cpu_end_time",
                        trace_only=True,
                        attrs={"pp_mb_id": action.slot_id},
                    )

                mb_metadata[action.slot_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                if is_last_stage:
                    output_tensors = (
                        self._pp_prepare_tensor_dict(result, cur_batch)
                        if self.ps.tp_rank == 0
                        else None
                    )
                    output_tensors = self.attn_tp_group.broadcast_tensor_dict(
                        output_tensors,
                        src=0,
                    )
                    output_tensors["vpp_batch_seq"] = action.batch_seq
                event = self.device_module.Event()
                event.record(self.device_module.current_stream())
                if is_last_stage:
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(output_tensors),
                        )
                    )
        return result, event, send_work

    def get_rids(
        self: Scheduler, req_queue: List[Req], is_send: bool, *poll_statuses_group
    ):
        """
        Used by PP, get the required rids with the given poll statuses.
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender if is_send else req.kv_receiver for req in req_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        rids: List = []
        for poll_statuses in poll_statuses_group:
            rids.append(
                [
                    req.rid if is_send else req.req.rid
                    for req, poll in zip(req_queue, polls)
                    if poll in poll_statuses
                ]
            )
        return tuple(rids) if len(rids) > 1 else rids[0]

    def _pp_pd_get_retract_ids(self: Scheduler, mb_id: int):
        # communicate pre-consensus retracted reqs
        for req in self.disagg_decode_prealloc_queue.retracted_queue:
            # assign retracted reqs to the current microbatch
            if req.retraction_mb_id is None:
                req.retraction_mb_id = mb_id
        curr_retract_rids = [
            req.rid
            for req in self.disagg_decode_prealloc_queue.retracted_queue
            if req.retraction_mb_id == mb_id
        ]
        if self.pp_group.is_first_rank:
            # First rank, get all retracted req ids for the microbatch
            return curr_retract_rids
        else:
            # Other ranks, receive the retracted reqs info from the previous rank and ensure the consensus
            prev_retract_rids = self._pp_recv_pyobj_from_prev_stage()
            return list(set(prev_retract_rids) & set(curr_retract_rids))

    def _pp_pd_get_prealloc_ids(self: Scheduler):
        # communicate pre-consensus prealloc reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the preallocated reqs from the prealloc queue
            good_prealloc_rids, bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the preallocated reqs info from the previous rank and ensure the consensus
            prev_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
            prev_good_prealloc_rids, prev_bad_prealloc_rids = prev_prealloc_rids
            curr_good_prealloc_rids, curr_bad_prealloc_rids = self.get_rids(
                self.disagg_decode_prealloc_queue.queue,
                False,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_prealloc_rids = list(
                set(prev_good_prealloc_rids) & set(curr_good_prealloc_rids)
            )
            bad_prealloc_rids = list(
                set(prev_bad_prealloc_rids) | set(curr_bad_prealloc_rids)
            )
        # Same abort routing as the prefill bootstrap consensus above.
        aborted_rids = {
            decode_req.req.rid
            for decode_req in self.disagg_decode_prealloc_queue.queue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT)
        }
        good_prealloc_rids, bad_prealloc_rids = self._route_aborts_to_bad(
            good_prealloc_rids, bad_prealloc_rids, aborted_rids
        )
        return [good_prealloc_rids, bad_prealloc_rids]

    @staticmethod
    def _route_aborts_to_bad(good_rids, bad_rids, aborted_rids):
        """Move aborted rids out of the good (intersection) set and into the
        bad (union) set, so PP consensus fails them uniformly on every rank.

        This also flushes aborted reqs that never reached good/bad consensus
        (e.g. stuck in Bootstrapping with a sender that has no working abort()).
        """
        if not aborted_rids:
            return good_rids, bad_rids
        good_rids = [rid for rid in good_rids if rid not in aborted_rids]
        bad_rids = list(set(bad_rids) | set(aborted_rids))
        return good_rids, bad_rids

    def _pp_pd_get_decode_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self._pp_recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_decode_transfer_queue.queue,
                False,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def process_retract_queue(self: Scheduler, retract_rids: Optional[List[str]]):
        if retract_rids is not None:
            # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
            resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs(
                retract_rids
            )
            self.waiting_queue.extend(resumed_reqs)
            return [req.rid for req in resumed_reqs]
        return None

    def process_prealloc_queue(self: Scheduler, prealloc_rids: Optional[List[str]]):
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return [[], []]

        if prealloc_rids is not None:
            (
                good_consensus_prealloc_rids,
                bad_consensus_prealloc_rids,
            ) = prealloc_rids
            good_reqs, failed_reqs = self.disagg_decode_prealloc_queue.pop_preallocated(
                pp_good_rids=good_consensus_prealloc_rids,
                pp_bad_rids=bad_consensus_prealloc_rids,
            )
            self.disagg_decode_transfer_queue.extend(good_reqs)
            return [
                [req.req.rid for req in good_reqs],
                [req.req.rid for req in failed_reqs],
            ]
        return None

    def process_decode_transfer_queue(
        self: Scheduler, release_rids: Optional[List[str]]
    ):
        # Resolve held deferred releases every call, independent of release_rids,
        # so ack/timeout-driven releases still fire when no rids are being polled.
        self.disagg_decode_transfer_queue.resolve_deferred_releases()
        if release_rids is not None:
            released_reqs = self.disagg_decode_transfer_queue.pop_transferred(
                release_rids
            )
            if self.enable_hisparse:
                for req in released_reqs:
                    self.hisparse_coordinator.admit_request_direct(req)
            self.waiting_queue.extend(released_reqs)
            return [req.rid for req in released_reqs]
        return None
