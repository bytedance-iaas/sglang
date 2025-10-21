from contextlib import contextmanager

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    CLIP_MAX_NEW_TOKENS_ESTIMATION,
    IGNORE_EOS_RESERVE_TOKENS,
    AddReqResult,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


class EncodeAdder:
    """
    Different from PrefillAdder, there are no mem pools in encode.

    The returned hidden_states will be transferred to downstream once returned, immediately, and encoder requires no kv-cache
    """

    def __init__(
        self,
        page_size: int,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        mixed_with_decode_tokens: int = 0,
    ):
        self.page_size = page_size
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        # print(f"{self.token_to_kv_pool_allocator=}")
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens

        self.rem_total_token_offset = mixed_with_decode_tokens
        self.cur_rem_token_offset = mixed_with_decode_tokens

        self.req_states = None
        self.can_run_list = []
        self.new_chunked_req = None
        self.log_hit_tokens = 0
        # TODO(lsyin): report the real input tokens excluding page alignment
        self.log_input_tokens = 0
        # print(f"EncodeAdder, {self.rem_total_tokens=}")

        if running_batch is not None:
            self.rem_total_token_offset += sum(
                [
                    min(
                        (r.sampling_params.max_new_tokens - len(r.output_ids)),
                        CLIP_MAX_NEW_TOKENS_ESTIMATION,
                    )
                    * self.new_token_ratio
                    for r in running_batch.reqs
                ]
            )

    @property
    def rem_total_tokens(self):
        return (
            self.token_to_kv_pool_allocator.available_size()
            # + self.tree_cache.evictable_size()
            # - self.rem_total_token_offset
        )

    @property
    def cur_rem_tokens(self):
        return (
            self.token_to_kv_pool_allocator.available_size()
            # + self.tree_cache.evictable_size()
            # - self.cur_rem_token_offset
        )

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # if self.rem_input_tokens <= 0 or (
        #     self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0
        # ):
        #     return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_encode_budget(self, embedding_length: int):
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        embedding_length = self.ceil_paged_tokens(embedding_length)

        self.rem_total_token_offset += embedding_length
        self.cur_rem_token_offset += embedding_length
        self.rem_input_tokens -= embedding_length
        # if self.rem_chunk_tokens is not None:
        #     self.rem_chunk_tokens -= extend_input_len

        # self.log_hit_tokens += prefix_len
        self.log_input_tokens += embedding_length

    @contextmanager
    def _lock_node(self):
        try:
            # self.tree_cache.inc_lock_ref(last_node)
            yield None
        finally:
            pass
            # self.tree_cache.dec_lock_ref(last_node)

    # def add_one_req_ignore_eos(self, req: Req, has_chunked_req: bool):
    #     # Early exit if no enough tokens for the input tokens
    #     if self.ceil_paged_tokens(req.extend_input_len) > min(
    #         self.cur_rem_tokens, self.rem_total_tokens
    #     ):
    #         return AddReqResult.NO_TOKEN
    #
    #     def add_req_state(r, insert_sort=False):
    #         new_token_ratio = (
    #             1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
    #         )
    #         tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
    #             r.output_ids
    #         )
    #         tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)
    #
    #         if tokens_left <= 0:
    #             return
    #
    #         if not insert_sort:
    #             self.req_states.append((tokens_left, tokens_occupied))
    #         else:
    #             i = 0
    #             for i in range(len(self.req_states)):
    #                 if tokens_left <= self.req_states[i][0]:
    #                     break
    #             self.req_states.insert(i, (tokens_left, tokens_occupied))
    #
    #     if self.req_states is None:
    #         self.req_states = []
    #         add_req_state(req)
    #         if self.running_batch is not None:
    #             for r in self.running_batch.reqs:
    #                 add_req_state(r)
    #         for r in self.can_run_list:
    #             add_req_state(r)
    #         self.req_states.sort(key=lambda x: x[0])
    #     else:
    #         add_req_state(req, insert_sort=True)
    #
    #     cur_rem_tokens = self.cur_rem_tokens - len(req.origin_input_ids)
    #     tokens_freed = 0
    #     for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
    #         # tokens_left gives a reservative calculation as the last token is not stored
    #         bs = len(self.req_states) - i
    #         min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
    #         # reserve tokens for corner cases
    #         if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
    #             return AddReqResult.NO_TOKEN
    #         tokens_freed += tokens_occupied
    #
    #     # if (
    #     #     self.rem_chunk_tokens is None  # chunked prefill is disabled
    #     #     or req.extend_input_len <= self.rem_chunk_tokens  # it is the last chunk
    #     # ):
    #         # Non-chunked prefill
    #     self.can_run_list.append(req)
    #     self._update_encode_budget(
    #         req.cu_mm_embedding_len
    #     )
    #     # else:
    #     #     if self.rem_chunk_tokens == 0:
    #     #         return AddReqResult.OTHER
    #     #
    #     #     # Chunked prefill
    #     #     trunc_len = self.rem_chunk_tokens
    #     #
    #     #     req.extend_input_len = trunc_len
    #     #     req.fill_ids = req.fill_ids[:trunc_len]
    #     #     self.can_run_list.append(req)
    #     #     self.new_chunked_req = req
    #     #     self._update_encode_budget(0, trunc_len, 0)
    #
    #     return self.budget_state()

    def add_one_req(self, req: Req):
        total_tokens = req.cu_mm_embedding_len

        # adjusting the input_tokens based on host_hit_length and page_size
        real_input_tokens = req.cu_mm_embedding_len

        # if total_tokens >= self.rem_total_tokens:
        #     return AddReqResult.NO_TOKEN

        if real_input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        # with self._lock_node(req.last_node):
        # self.rem_total_tokens may decrease after the lock acquisition
        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        # if req.host_hit_length > 0:
        #     new_indices, req.last_node = self.tree_cache.init_load_back(
        #         req.last_host_node, req.host_hit_length
        #     )
        #     req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
        #     req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        #     prefix_len = len(req.prefix_indices)

        input_tokens = self.ceil_paged_tokens(req.cu_mm_embedding_len)

        if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        # if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
        # Non-chunked prefill
        self.can_run_list.append(req)
        # self.tree_cache.inc_lock_ref(req.last_node)
        self._update_encode_budget(
            # prefix_len,
            input_tokens,
            # min(
            #     req.sampling_params.max_new_tokens,
            #     CLIP_MAX_NEW_TOKENS_ESTIMATION,
            # ),
        )
        # else:
        #     # Make sure at least one page is available
        #     trunc_len = self.rem_chunk_tokens - self.page_size + 1
        #     if trunc_len <= 0:
        #         return AddReqResult.OTHER
        #
        #     # Chunked prefill
        #     req.extend_input_len = trunc_len
        #     req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
        #
        #     self.can_run_list.append(req)
        #     self.new_chunked_req = req
        #     self._update_encode_budget(trunc_len, 0)

        # FIXME: rem_tokens
        return self.budget_state()
        # return AddReqResult.CONTINUE
