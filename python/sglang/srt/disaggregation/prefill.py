"""
Life cycle of a request in the prefill server

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
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.base import BaseKVManager, KVPoll
from sglang.srt.disaggregation.decode import EmbeddingRequest
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    kv_to_page_num,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.multimodal_cache import (
    MultimodalCache,
    PagedMultiModalEmbeddingPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import require_mlp_sync

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        decode_tp_size: int,
        decode_dp_size: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
        # bootstrap_host: str,
        # disagg encode
        mm_embedding_pool: Optional[PagedMultiModalEmbeddingPool] = None,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.decode_tp_size = decode_tp_size
        self.decode_dp_size = decode_dp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.max_total_num_tokens = max_total_num_tokens
        self.scheduler = scheduler
        self.transfer_backend = transfer_backend
        self.kv_manager = self._init_kv_manager()

        self.mm_embedding_pool = mm_embedding_pool

    def _init_kv_manager(self) -> BaseKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.decode_tp_size = self.decode_tp_size // self.decode_dp_size
        kv_args.prefill_pp_size = self.pp_size
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )

        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        return kv_manager

    def add(self, req: Req) -> None:
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
            kv_sender_class = get_kv_class(TransferBackend.FAKE, KVClassType.SENDER)
        else:
            kv_sender_class = get_kv_class(self.transfer_backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
        )
        self._process_req(req)
        self.queue.append(req)

    def extend(self, reqs: List[Req]) -> None:
        for req in reqs:
            self.add(req)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
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
        if len(self.queue) == 0:
            if not return_failed_reqs:
                return [], []
            else:
                return [], []

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.queue], self.gloo_group
        )
        for i, (req, poll) in enumerate(zip(self.queue, polls)):

            if rids_to_check is not None:
                # if req not in reqs_info_to_check, skip
                if req.rid not in rids_to_check:
                    continue
                # Either waiting for input or failed
                assert poll == KVPoll.WaitingForInput or poll == KVPoll.Failed

            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                error_message = f"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
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

            # KV.WaitingForInput - init here
            num_kv_indices = len(req.origin_input_ids)
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None

            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)

            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if not return_failed_reqs:
            return bootstrapped_reqs, []
        else:
            return bootstrapped_reqs, failed_reqs


class MMEmbeddingTransferQueue:
    """
    Store the requests that is polling mm embedding from encoders
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        # req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        # metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
    ):
        self.queue: List[EmbeddingRequest] = []
        self.gloo_group = gloo_group
        # self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        # self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        # self.spec_algorithm = scheduler.spec_algorithm

    def add(self, prefill_req: EmbeddingRequest) -> None:
        self.queue.append(prefill_req)

    def extend(self, prefill_reqs: List[EmbeddingRequest]) -> None:
        self.queue.extend(prefill_reqs)

    def pop_transferred(self) -> List[Req]:
        if not self.queue:
            return []
        polls = poll_and_all_reduce(
            [embedding_req.embedding_receiver for embedding_req in self.queue],
            self.gloo_group,
        )

        transferred_reqs = []
        indices_to_remove = set()
        for i, (embedding_req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Failed:
                error_message = f"Encode transfer failed for request rank={self.tp_rank} {embedding_req.req.rid=} {embedding_req.req.bootstrap_room=}"
                try:
                    embedding_req.embedding_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)

                prepare_abort(
                    embedding_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.stream_output(
                    [embedding_req.req], embedding_req.req.return_logprob
                )
                # unlock the kv cache or it will have memory leak
                indices_to_remove.add(i)
                continue
            elif poll == KVPoll.Success:

                # idx = prefill_req.metadata_buffer_index
                # (
                #     output_id,
                #     output_token_logprobs_val,
                #     output_token_logprobs_idx,
                #     output_top_logprobs_val,
                #     output_top_logprobs_idx,
                #     output_hidden_states,
                # ) = self.metadata_buffers.get_buf(idx)

                # prefill_req.req.output_ids.append(output_id[0].item())
                # if prefill_req.req.return_logprob:
                #     prefill_req.req.output_token_logprobs_val.append(
                #         output_token_logprobs_val[0].item()
                #     )
                #     prefill_req.req.output_token_logprobs_idx.append(
                #         output_token_logprobs_idx[0].item()
                #     )
                #     prefill_req.req.output_top_logprobs_val.append(
                #         output_top_logprobs_val[
                #         : prefill_req.req.top_logprobs_num
                #         ].tolist()
                #     )
                #     prefill_req.req.output_top_logprobs_idx.append(
                #         output_top_logprobs_idx[
                #         : prefill_req.req.top_logprobs_num
                #         ].tolist()
                #     )

                if hasattr(embedding_req.embedding_receiver, "clear"):
                    embedding_req.embedding_receiver.clear()

                # special handling for sampling_params.max_new_tokens == 1
                # if embedding_req.sampling_params.max_new_tokens == 1:
                #     # finish immediately
                #     embedding_req.check_finished()
                #     self.scheduler.stream_output(
                #         [embedding_req], embedding_req.return_logprob
                #     )
                # else:
                #     transferred_reqs.append(embedding_req)
                transferred_reqs.append(embedding_req.req)

                indices_to_remove.add(i)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        # for i in indices_to_remove:
        #     idx = self.queue[i].metadata_buffer_index
        #     assert idx != -1
        #     self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs


class MMEmbeddingPreallocQueue:
    """
    Store the requests that are preallocating, to prealloc mm embeddings from encoder
    """

    def __init__(
        self,
        mm_embedding_pool: PagedMultiModalEmbeddingPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        # req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        # metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: MMEmbeddingTransferQueue,
        gloo_group: ProcessGroup,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        # prefill_pp_size: int,
        # num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
        tp_rank: Optional[int] = 0,
    ):
        self.mm_embedding_pool = mm_embedding_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        # self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        # self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        # self.metadata_buffers = metadata_buffers
        # self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        # self.prefill_pp_size = prefill_pp_size
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[EmbeddingRequest] = []
        self.retracted_queue: List[Req] = []
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> BaseKVManager:
        # TODO
        ...
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        #
        # attn_tp_size = self.tp_size // self.dp_size
        kv_args.engine_rank = self.tp_rank
        # kv_args.decode_tp_size = attn_tp_size
        # kv_args.prefill_pp_size = self.prefill_pp_size
        # kv_data_ptrs, kv_data_lens, kv_item_lens = (
        #     self.token_to_kv_pool.get_contiguous_buf_infos()
        # )
        #
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.mm_embedding_pool.get_mm_buffer_info()
        )
        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens

        kv_args.aux_data_ptrs = []
        kv_args.aux_data_lens = []
        kv_args.aux_item_lens = None

        #
        # kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
        #     self.metadata_buffers.get_buf_infos()
        # )
        #
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.TEXT,
            self.scheduler.server_args,
        )
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        logger.debug(
            f"adding request {req.rid=} to MMEmbeddingPreallocQueue, waiting to be bootstrapped..."
        )
        if self._check_if_req_exceed_mm_pool_capacity(req):
            return
        if is_retracted:
            self.retracted_queue.append(req)
        else:
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                kv_receiver_class = get_kv_class(
                    TransferBackend.FAKE, KVClassType.RECEIVER
                )
            else:
                kv_receiver_class = get_kv_class(
                    self.transfer_backend, KVClassType.RECEIVER
                )
            kv_receiver = kv_receiver_class(
                mgr=self.kv_manager,
                bootstrap_addr=f"{req.bootstrap_host_encode}:{req.bootstrap_port_encode}",
                bootstrap_room=req.bootstrap_room,
                data_parallel_rank=req.data_parallel_rank,
                disaggregation_mode=DisaggregationMode.PREFILL,
            )
            logger.debug(f"EmbeddingRequest added")
            self.queue.append(
                EmbeddingRequest(
                    req=req, embedding_receiver=kv_receiver, waiting_for_input=False
                )
            )

    def _check_if_req_exceed_mm_pool_capacity(self, req: Req) -> bool:
        # TODO:
        # if len(req.origin_input_ids) > self.max_total_num_tokens:
        #     message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
        #     logger.error(message)
        #     prepare_abort(req, message)
        #     self.scheduler.stream_output([req], req.return_logprob)
        #     return True
        return False

    def extend(self, reqs: List[Req]) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req)

    def resume_retracted_reqs(self) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible
        # print(f"resume_retracted_reqs...")
        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        allocatable_tokens = self._allocatable_tokens()

        for i, req in enumerate(self.retracted_queue):
            if self.mm_embedding_pool.available_size() <= 0:
                break

            # print(f"{req=}")

            required_tokens_for_request = sum(req.mm_embedding_lens)
            if required_tokens_for_request > allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            allocatable_tokens -= required_tokens_for_request

            # load from cpu, release the cpu copy
            # req.load_kv_cache(self.mm_embedding_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(self) -> None:
        if not self.queue:
            return

        if all(embedding_req.waiting_for_input for embedding_req in self.queue):
            return

        polls = poll_and_all_reduce(
            [embedding_req.embedding_receiver for embedding_req in self.queue],
            self.gloo_group,
        )

        for i, (encode_req, poll) in enumerate(zip(self.queue, polls)):
            # print(f"{poll=}")
            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                encode_req.waiting_for_input = True
            elif poll == KVPoll.Failed:
                error_message = f"Language handshake failed for request rank={self.tp_rank} {encode_req.req.rid=} {encode_req.req.bootstrap_room=}"
                try:
                    encode_req.embedding_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    encode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def pop_preallocated(self) -> List[EmbeddingRequest]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        self._update_handshake_waiters()

        preallocated_reqs = []
        indices_to_remove = set()

        allocatable_tokens = self._allocatable_tokens()
        # First, remove all failed requests from the queue
        for i, encode_req in enumerate(self.queue):
            if isinstance(encode_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [encode_req.req], encode_req.req.return_logprob
                )
                indices_to_remove.add(i)

        # Then, preallocate the remaining requests if possible
        for i, encode_req in enumerate(self.queue):
            if i in indices_to_remove:
                continue

            if not encode_req.waiting_for_input:
                continue

            if self.mm_embedding_pool.available_size() <= 0:
                break

            # if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
            #     break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            required_tokens_for_request = sum(encode_req.req.mm_embedding_lens)

            if required_tokens_for_request > allocatable_tokens:
                break

            allocatable_tokens -= required_tokens_for_request
            mm_embedding_indices = (
                self._pre_alloc(encode_req.req).to(torch.int32).cpu().numpy()
            )  # it's type should be int32

            logger.debug(f"{encode_req.req.mm_hashes=}")

            # mm_embedding_indices = [self.mm_embedding_pool.get_embedding_locs_from_hash(mm_hash) for mm_hash in
            #                         encode_req.req.mm_hashes]

            # print(f"{mm_embedding_indices=}")

            # prefill_req.metadata_buffer_index = (
            #     self.req_to_metadata_buffer_idx_allocator.alloc()
            # )
            # assert prefill_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(mm_embedding_indices, page_size=1)
            encode_req.embedding_receiver.init(page_indices, -1)
            preallocated_reqs.append(encode_req)
            indices_to_remove.add(i)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            len(decode_req.req.fill_ids) for decode_req in self.transfer_queue.queue
        )

    def _allocatable_tokens(self) -> int:
        return self.mm_embedding_pool.available_size()

    def _pre_alloc(self, req: Req) -> List[torch.Tensor]:
        """Pre-allocate the memory for req_to_token and token_kv_pool"""
        logger.debug(f"pre_allocating...")
        # req_pool_indices = self.mm_embedding_pool.alloc(1)

        # assert (
        #     req_pool_indices is not None
        # ), "req_pool_indices is full! There is a bug in memory estimation."
        #
        # req.req_pool_idx = req_pool_indices[0]

        # if self.token_to_kv_pool_allocator.page_size == 1:
        #     raise NotImplementedError()
        #     # kv_loc = self.token_to_kv_pool_allocator.alloc(
        #     #     len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        #     # )
        # else:
        mm_hash = MultimodalCache.combine_hashes(
            [item.hash for item in req.multimodal_inputs.mm_items]
        )
        mm_embedding_lens = sum(
            mm_embedding_len for mm_embedding_len in req.mm_embedding_lens
        )

        # print(f"prefill 699 | {mm_hash=}")
        # print(f"prefill 701 | {mm_embedding_lens=}")

        embedding_locs = self.mm_embedding_pool.reserve_mm_embedding(
            mm_hash, mm_embedding_lens, self.token_to_kv_pool_allocator
        )

        # assert (
        #     kv_loc is not None
        # ), "KV cache is full! There is a bug in memory estimation."

        # self.mm_embedding_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)

        # populate metadata
        # req.fill_ids = req.origin_input_ids + req.output_ids
        # req.extend_input_len = len(req.origin_input_ids)

        return embedding_locs


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def process_prefill_bootstrapping_queue(self: Scheduler):
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        # Poll for bootstrapped requests and add to the bootstrapped_queue
        bootstrapped, _failed = self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
        if bootstrapped:
            # print(f"{bootstrapped}")
            # Find requests that are already in the embedding_received_queue
            embedding_received_rooms = {
                req.bootstrap_room for req in self.embedding_received_queue
            }
            ready_reqs = [
                req
                for req in bootstrapped
                if req.bootstrap_room in embedding_received_rooms
            ]

            # if ready_reqs:
            #     print(f"{ready_reqs=}")
            # Add ready requests to the waiting_queue
            self.waiting_queue.extend(ready_reqs)

            # Remove ready requests from embedding_received_queue
            ready_rooms = {req.bootstrap_room for req in ready_reqs}
            self.embedding_received_queue = [
                req
                for req in self.embedding_received_queue
                if req.bootstrap_room not in ready_rooms
            ]

            # Add the remaining bootstrapped requests to the bootstrapped_queue
            self.bootstrapped_queue.extend(
                req for req in bootstrapped if req.bootstrap_room not in ready_rooms
            )
            # if self.bootstrapped_queue:
            #     print(f"{self.bootstrapped_queue=}")

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self.server_args.encoder_disaggregated:

                self.process_prefill_mm_embedding_transfer_queue()
                self.process_prefill_bootstrapping_queue()
            else:
                bootstrapped, _failed = (
                    self.disagg_encode_bootstrap_queue.pop_bootstrapped()
                )
                self.waiting_queue.extend(bootstrapped)
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            if require_mlp_sync(self.server_args):
                batch = self.prepare_mlp_sync_batch(batch)
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result_disagg_prefill(batch, result)

            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self.server_args.encoder_disaggregated:
                self.process_prefill_mm_embedding_transfer_queue()
                self.process_prefill_bootstrapping_queue()
            else:
                bootstrapped, _failed = (
                    self.disagg_encode_bootstrap_queue.pop_bootstrapped()
                )
                self.waiting_queue.extend(bootstrapped)
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            if require_mlp_sync(self.server_args):
                batch = self.prepare_mlp_sync_batch(batch)
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
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                self.process_batch_result_disagg_prefill(tmp_batch, tmp_result)

            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
        )

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        if self.enable_overlap:
            # wait
            logits_output, next_token_ids, _ = self.tp_worker.resolve_last_batch_result(
                launch_done
            )
        else:
            next_token_ids = result.next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )

        hidden_state_offset = 0
        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            req: Req
            if req.is_chunked <= 0:
                # There is no output_ids for prefill
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                self.disagg_prefill_inflight_queue.append(req)

                if req.multimodal_inputs:
                    mm_hash = MultimodalCache.combine_hashes(
                        [item.hash for item in req.multimodal_inputs.mm_items]
                    )
                    _loc = self.mm_embedding_pool.free(
                        mm_hash, self.mm_embedding_allocator
                    )

                if logits_output.hidden_states is not None:
                    last_hidden_index = (
                        hidden_state_offset + extend_input_len_per_req[i] - 1
                    )
                    req.hidden_states_tensor = (
                        logits_output.hidden_states[last_hidden_index].cpu().clone()
                    )
                    hidden_state_offset += extend_input_len_per_req[i]
                else:
                    req.hidden_states_tensor = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                self.send_kv_chunk(req, last_chunk=True)

                if req.grammar is not None:
                    # FIXME: this try-except block is for handling unexpected xgrammar issue.
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        self.tree_cache.cache_finished_req(req)
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                if self.enable_overlap:
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)

        # We need to remove the sync in the following function for overlap schedule.
        self.set_next_batch_sampling_info_done(batch)

    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []

        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):

            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                assert poll == KVPoll.Success or poll == KVPoll.Failed

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
            elif poll == KVPoll.Failed:
                error_message = f"Prefill transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                self.tree_cache.cache_finished_req(req)  # unlock the tree
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
            else:
                assert False, f"Unexpected polling state {poll=}"

        # Stream requests which have finished transfer
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req
            self.req_to_metadata_buffer_idx_allocator.free(req.metadata_buffer_index)
            req.metadata_buffer_index = -1

        self.disagg_prefill_inflight_queue = undone_reqs

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

    def process_prefill_chunk(self: Scheduler) -> None:
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.chunked_req:
                # Move the chunked request out of the batch so that we can merge
                # only finished requests to running_batch.
                self.last_batch.filter_batch(chunked_req_to_exclude=self.chunked_req)
                self.tree_cache.cache_unfinished_req(self.chunked_req)
                if self.enable_overlap:
                    # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                    self.chunked_req.tmp_end_idx = min(
                        len(self.chunked_req.fill_ids),
                        len(self.chunked_req.origin_input_ids),
                    )
                else:
                    self.send_kv_chunk(self.chunked_req)
                # chunked request keeps its rid but will get a new req_pool_idx
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
                self.running_batch.batch_is_full = False

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        end_idx = (
            end_idx
            if end_idx is not None
            else min(len(req.fill_ids), len(req.origin_input_ids))
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)
        page_indices = kv_to_page_indices(kv_indices, page_size)
        if len(page_indices) == 0:
            logger.info(
                f"Skip sending kv chunk for request {req.rid=} {req.bootstrap_room=} because page_indices is empty"
            )
            return
        req.disagg_kv_sender.send(page_indices)

    def process_prefill_mm_embedding_transfer_queue(self: Scheduler):
        """
        process text-model prefill queue when encoder if disaggregated, dumping mm_received req to disagg_prefill_bootstrap_queue, waiting to be bootstrapped
        """
        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        # resumed_reqs = self.disagg_prefill_prealloc_queue.resume_retracted_reqs()
        # self.waiting_queue.extend(resumed_reqs)
        # if len(self.disagg_prefill_prealloc_queue.retracted_queue) > 0:
        #     if there are still retracted requests, we do not allocate new requests
        #     return

        # the req whose embedding has been pre-allocated
        # TODO: this is time-consuming and foreground
        allocated_reqs = self.disagg_prefill_prealloc_queue.pop_preallocated()
        self.disagg_prefill_receiving_queue.extend(allocated_reqs)

        mm_received_reqs = self.disagg_prefill_receiving_queue.pop_transferred()
        if mm_received_reqs:
            if self.server_args.disaggregation_mode == "prefill":
                # Find requests that are already in the bootstrapped_queue
                received_rooms = {req.bootstrap_room for req in mm_received_reqs}
                ready_reqs = [
                    req
                    for req in self.bootstrapped_queue
                    if req.bootstrap_room in received_rooms
                ]

                # Add ready requests to the waiting_queue
                self.waiting_queue.extend(ready_reqs)

                # Remove ready requests from bootstrapped_queue
                ready_rooms = {req.bootstrap_room for req in ready_reqs}
                self.bootstrapped_queue = [
                    req
                    for req in self.bootstrapped_queue
                    if req.bootstrap_room not in ready_rooms
                ]

                # Add the remaining received requests to the embedding_received_queue
                self.embedding_received_queue.extend(
                    req
                    for req in mm_received_reqs
                    if req.bootstrap_room not in ready_rooms
                )
            else:
                self.waiting_queue.extend(mm_received_reqs)
