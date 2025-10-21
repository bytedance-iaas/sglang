"""
Life cycle of a request in the decode server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.base import BaseKVManager, KVPoll
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.schedule_policy_encode_adder import EncodeAdder
from sglang.srt.mem_cache.multimodal_cache import (
    MultimodalCache,
    PagedMultiModalEmbeddingPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class EncodeBootstrapQueue:
    """
    Store the requests bootstrapping
    """

    def __init__(
        self,
        mm_embedding_pool: PagedMultiModalEmbeddingPool,
        # draft_mm_embedding_pool: Optional[PagedMultiModalCache],
        # req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        # metadata_buffers: EncoderMetadataBuffers,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        encode_dp_size: int,
        scheduler: Scheduler,
        transfer_backend: TransferBackend,
    ):
        self.mm_embedding_pool = mm_embedding_pool
        # self.draft_mm_embedding_pool = draft_mm_embedding_pool
        # self.metadata_buffers = metadata_buffers
        # self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.encode_dp_size = encode_dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        # reqs waiting to be bootstrapped
        self.waiting_reqs: List[Req] = []
        self.gloo_group = gloo_group
        self.max_total_num_tokens = max_total_num_tokens
        self.scheduler = scheduler
        self.transfer_backend = transfer_backend
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> BaseKVManager:
        print(f"initializing kv manager")
        # TODO
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.mm_embedding_pool.get_mm_buffer_info()
        )

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens

        kv_args.aux_data_ptrs = []
        kv_args.aux_data_lens = []
        kv_args.aux_item_lens = None

        kv_args.page_size = 1
        # placeholder
        kv_args.engine_rank = 0

        # kv_args.page_size = self.mm_embedding_pool.page_size
        #
        # kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
        #     self.metadata_buffers.get_buf_infos()
        # )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.ENCODE,
            self.scheduler.server_args,
        )
        return kv_manager

    def add(self, req: Req) -> None:
        logger.debug(f"adding req to EncodeBootstrapQueue, waiting to be bootstrapped")
        if self._check_if_req_exceed_capacity(req):
            return

        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            kv_sender_class = get_kv_class(TransferBackend.FAKE, KVClassType.SENDER)
        else:
            kv_sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)

        # TODO: embedding sender class
        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
        )
        self._process_req(req)
        self.waiting_reqs.append(req)

    def extend(self, reqs: List[Req]) -> None:
        for req in reqs:
            self.add(req)

    def _check_if_req_exceed_capacity(self, req: Req) -> bool:
        # TODO: not accurate check
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message)
            # self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so EncodeAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        rids_to_check: Optional[List[str]] = None,
    ) -> Tuple[List[Req], List[Req]]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.waiting_reqs) == 0:
            if not return_failed_reqs:
                return [], []
            else:
                return [], []

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.waiting_reqs], self.gloo_group
        )
        for i, (req, poll) in enumerate(zip(self.waiting_reqs, polls)):
            if rids_to_check is not None:
                # if req not in reqs_info_to_check, skip
                if req.rid not in rids_to_check:
                    continue
                # Either waiting for input or failed
                assert poll == KVPoll.WaitingForInput or poll == KVPoll.Failed
            # print(f"{req=} {poll=}")
            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                error_message = f"Encode bootstrap failed for request rank={req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                failed_reqs.append(req)
                continue

            logger.debug(f"bootstrapped")
            # KV.WaitingForInput - init here
            num_kv_indices = len(req.origin_input_ids)
            assert req.metadata_buffer_index is not None

            # num_pages = kv_to_page_num(num_kv_indices, self.mm_embedding_pool.page_size)
            num_pages = -1
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.waiting_reqs = [
            entry
            for i, entry in enumerate(self.waiting_reqs)
            if i not in indices_to_remove
        ]

        # print(f"{len(bootstrapped_reqs)=}")
        if not return_failed_reqs:
            return bootstrapped_reqs, []
        else:
            return bootstrapped_reqs, failed_reqs


class SchedulerDisaggregationEncodeMixin:
    """
    Mixin for Scheduler to handle disaggregation encode
    """

    def get_new_batch_encode(self: Scheduler) -> Optional[ScheduleBatch]:
        """Get a new batch for encode mode.

        This method builds a batch of requests for encoding multimodal embeddings,
        ensuring we don't exceed memory limits while maximizing batch size.
        Encode mode doesn't involve tree_cache, chunked prefill, or speculative algorithms.
        """
        assert self.disaggregation_mode == DisaggregationMode.ENCODE

        # Handle the cases where encode is not allowed
        if self.running_batch.batch_is_full or len(self.waiting_queue) == 0:
            return None

        running_bs = len(self.running_batch.reqs)

        # Check if we can allocate more requests
        if self.get_num_allocatable_reqs_encode(running_bs) <= 0:
            self.running_batch.batch_is_full = True
            return None

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue)

        # Encode policy - use mm_embedding_allocator instead of token_to_kv_pool_allocator
        adder = EncodeAdder(
            self.page_size,
            self.mm_embedding_allocator,  # Use multimodal embedding allocator
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            # self.chunked_prefill_size,
            # running_bs if self.is_mixed_chunk else 0,
        )

        # Get requests from the waiting queue to a new encode batch
        for req in self.waiting_queue:
            # Check if we've reached the maximum allocatable requests
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs_encode(
                running_bs
            ):
                self.running_batch.batch_is_full = True
                break

            # Check multimodal embedding pool availability for encode mode
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                # In encode mode, we need to check if the multimodal embedding pool has enough space
                if (
                    len(adder.can_run_list)
                    >= self.mm_embedding_allocator.available_size()
                ):
                    self.running_batch.batch_is_full = True
                    break

            # Add the request to the batch
            res = adder.add_one_req(req)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    # No more memory available for multimodal embeddings
                    self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        if self.enable_metrics:
            # Only record queue time when enable_metrics is True to avoid overhead
            for req in can_run_list:
                req.queue_time_end = time.perf_counter()

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        # Print stats
        if self.current_scheduler_metrics_enabled():
            self.log_encode_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
            chunked_req=None,  # Encode mode doesn't use chunked requests
            device=self.device,
        )

        # Encode mode doesn't use hierarchical cache
        # if self.enable_hierarchical_cache:
        #     new_batch.hicache_consumer_index = (
        #         self.tree_cache.ready_to_load_host_cache()
        #     )

        # Prepare the batch for extend mode (encode is similar to extend)
        new_batch.prepare_for_extend(should_set_req_pool_indices=False)

        # Encode mode doesn't have decoding requests
        new_batch.decoding_reqs = None

        logger.debug(f"get_new_batch_encode: batch_size={len(can_run_list)}")
        return new_batch

    def log_encode_stats(
        self: Scheduler,
        adder: EncodeAdder,
        can_run_list: List[Req],
        running_bs: int,
    ):
        gap_latency = time.perf_counter() - self.last_encode_stats_tic
        self.last_encode_stats_tic = time.perf_counter()
        self.last_input_throughput = self.last_encode_tokens / gap_latency
        self.last_encode_tokens = adder.log_input_tokens

        # num_used, token_usage, _, _ = self._get_token_info()
        # token_msg = f"token usage: {token_usage:.2f}, "

        num_new_seq = len(can_run_list)
        f = (
            f"Encode batch. "
            f"#new-seq: {num_new_seq}, "
            f"#new-token: {adder.log_input_tokens}, "
            # f"{token_msg}"
        )

        f += f"#running-req: {running_bs}, "
        f += f"#queue-req: {len(self.waiting_queue)}, "

        logger.info(f)

        if self.enable_metrics:
            self.stats.num_running_reqs = running_bs
            # self.stats.num_used_tokens = num_used
            # self.stats.token_usage = round(token_usage, 2)
            self.stats.num_queue_reqs = len(self.waiting_queue)

            total_queue_latency = 0
            for req in can_run_list:
                total_queue_latency += req.queue_time_end - req.queue_time_start
            self.stats.avg_request_queue_latency = total_queue_latency / num_new_seq

            self.metrics_collector.log_stats(self.stats)

    @torch.no_grad()
    def event_loop_normal_disagg_encode(self: Scheduler) -> None:
        """A normal scheduler loop for encode worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            bootstrapped, _failed = (
                self.disagg_encode_bootstrap_queue.pop_bootstrapped()
            )
            self.waiting_queue.extend(bootstrapped)

            batch = self.get_new_batch_encode()

            # if require_mlp_sync(self.server_args):
            #     batch = self.prepare_mlp_sync_batch(batch)
            self.cur_batch = batch

            if batch:
                _result = self.run_batch(batch)
                self.process_batch_result_disagg_encode(batch)

            self.process_disagg_encode_inflight_queue()

            if batch is None and len(self.disagg_encode_inflight_queue) == 0:
                # self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    @torch.no_grad()
    def event_loop_overlap_disagg_encode(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            bootstrapped, _failed = (
                self.disagg_encode_bootstrap_queue.pop_bootstrapped()
            )
            self.waiting_queue.extend(bootstrapped)

            batch = self.get_new_batch_encode()

            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    self.set_next_batch_sampling_info_done(tmp_batch)

            if self.last_batch:
                tmp_batch, _tmp_result = self.result_queue.popleft()
                self.process_batch_result_disagg_encode(tmp_batch)

            if len(self.disagg_encode_inflight_queue) > 0:
                self.process_disagg_encode_inflight_queue()

            if batch is None and len(self.disagg_encode_inflight_queue) == 0:
                # self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    def process_batch_result_disagg_encode(
        self: Scheduler,
        batch: ScheduleBatch,
        launch_done: Optional[threading.Event] = None,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        # (
        #     logits_output,
        #     _next_token_ids,
        #     extend_input_len_per_req,
        #     extend_logprob_start_len_per_req,
        # ) = (
        #     result.logits_output,
        #     result.next_token_ids,
        #     result.extend_input_len_per_req,
        #     result.extend_logprob_start_len_per_req,
        # )

        # logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        if self.enable_overlap:
            self.tp_worker.resolve_last_batch_result(launch_done)
        #     raise NotImplementedError()
        # wait
        # else:
        # _next_token_ids = result.next_token_ids.tolist()
        # if batch.return_logprob:
        #     if logits_output.next_token_logprobs is not None:
        #         logits_output.next_token_logprobs = (
        #             logits_output.next_token_logprobs.tolist()
        #         )
        #     if logits_output.input_token_logprobs is not None:
        #         logits_output.input_token_logprobs = tuple(
        #             logits_output.input_token_logprobs.tolist()
        #         )

        # hidden_state_offset = 0
        # for i, (req, next_token_id) in enumerate(
        #     zip(batch.reqs, _next_token_ids, strict=True)
        # ):
        #     req: Req
        #     assert req.is_chunked <= 0
        #     # There is no output_ids for prefill
        #     req.output_ids.append(next_token_id)
        #     self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
        #     self.disagg_encode_inflight_queue.append(req)
        #     if logits_output.hidden_states is not None:
        #         last_hidden_index = (
        #             hidden_state_offset + extend_input_len_per_req[i] - 1
        #         )
        #         req.hidden_states_tensor = (
        #             logits_output.hidden_states[last_hidden_index].cpu().clone()
        #         )
        #         hidden_state_offset += extend_input_len_per_req[i]
        #     else:
        #         req.hidden_states_tensor = None
        #     if req.return_logprob:
        #         assert extend_logprob_start_len_per_req is not None
        #         assert extend_input_len_per_req is not None
        #         extend_logprob_start_len = extend_logprob_start_len_per_req[i]
        #         extend_input_len = extend_input_len_per_req[i]
        #         num_input_logprobs = extend_input_len - extend_logprob_start_len
        #         self.add_logprob_return_values(
        #             i,
        #             req,
        #             logprob_pt,
        #             _next_token_ids,
        #             num_input_logprobs,
        #             logits_output,
        #         )
        #         logprob_pt += num_input_logprobs
        # FIXME: manually set finish reason to let req response

        # print("setting finish reason")
        for i, req in enumerate(batch.reqs):
            req.finished_reason = FINISH_LENGTH(length=0)
            self.disagg_encode_inflight_queue.append(req)
            self.send_embedding_chunk(req)

        # for i, req in enumerate(batch.reqs):
        #     self.mm_embedding_allocator.free(req.cu_mm_embedding_len)
        # We need to remove the sync in the following function for overlap schedule.
        # self.set_next_batch_sampling_info_done(batch)
        # print(f"{batch.reqs=}")
        self.stream_output(batch.reqs, batch.return_logprob)

    def process_disagg_encode_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.disagg_encode_inflight_queue) == 0:
            return []
        # print(f"process_disagg_encode_inflight_queue")

        done_reqs = []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_encode_inflight_queue],
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_encode_inflight_queue, polls):

            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                assert poll == KVPoll.Success or poll == KVPoll.Failed

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                if req.multimodal_inputs is not None:
                    mm_hash = MultimodalCache.combine_hashes(
                        [item.hash for item in req.multimodal_inputs.mm_items]
                    )
                    _loc = self.mm_embedding_pool.free(
                        mm_hash, self.mm_embedding_allocator
                    )
                # self.tree_cache.cache_finished_req(req)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
            elif poll == KVPoll.Failed:
                error_message = f"Encode transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                if req.multimodal_inputs is not None:
                    mm_hash = MultimodalCache.combine_hashes(
                        [item.hash for item in req.multimodal_inputs.mm_items]
                    )
                    _loc = self.mm_embedding_pool.free(
                        mm_hash, self.mm_embedding_allocator
                    )
                # self.tree_cache.cache_finished_req(req)  # unlock the tree
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
            else:
                assert False, f"Unexpected polling state {poll=}"

        # Stream requests which have finished transfer
        # print(f"streaming requests out")
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        # for req in done_reqs:
        #     req: Req
        # self.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
        # req.metadata_buffer_index = -1

        self.disagg_encode_inflight_queue = undone_reqs

        return done_reqs

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP, get the transferred rids but **do not pop**
        """
        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_encode_inflight_queue],
            self.tp_worker.get_tp_group().cpu_group,
        )

        transferred_rids: List[str] = []

        for req, poll in zip(self.disagg_encode_inflight_queue, polls):
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def send_embedding_chunk(self: Scheduler, req: Req) -> None:
        """
        Send a embedding to the prefill server
        """

        mm_hash = MultimodalCache.combine_hashes(
            [item.hash for item in req.multimodal_inputs.mm_items]
        )
        # print(f"{mm_hash=}")
        # print(f"{[item.hash for item in req.multimodal_inputs.mm_items]=}")
        # mm_hash = req.multimodal_inputs.mm_items[0].hash

        mm_indices = (
            self.mm_embedding_pool.get_embedding_locs_from_hash(mm_hash).cpu().numpy()
        )
        req.disagg_kv_sender.send_embedding(mm_indices)
