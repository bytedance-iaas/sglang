from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Deque, List, Optional

from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerFlowMatchingMixin:
    """Mixin for the Scheduler that handles Alpamayo-R1 flow matching batches.

    Flow matching is treated as a prefill-like operation: the scheduler
    dispatches a single FLOW_MATCHING batch, and the model runs all Euler
    steps synchronously inside forward_flow_matching().

    Scheduling priority (inserted in get_next_batch_to_run):
        1. Merge last extend batch (existing)
        2. Flow matching batch  ← this mixin (high priority)
        3. New prefill batch    (existing)
        4. Decode               (existing)
    """

    def init_flow_matching(self: Scheduler):
        """Initialize flow matching state. Called from Scheduler.__init__."""
        self.flow_matching_queue: Deque[Req] = deque()

    def get_new_batch_flow_matching(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a ScheduleBatch for all pending flow matching requests.

        Returns None if the queue is empty.
        """
        if not self.flow_matching_queue:
            return None

        reqs: List[Req] = [
            r for r in self.flow_matching_queue if r.req_pool_idx is not None
        ]
        self.flow_matching_queue.clear()
        if not reqs:
            return None

        new_batch = ScheduleBatch(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            forward_mode=ForwardMode.FLOW_MATCHING,
            enable_overlap=False,
        )
        new_batch.prepare_for_flow_matching()
        return new_batch

    def process_batch_result_flow_matching(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        """Handle results after forward_flow_matching() completes.

        Collects trajectory data from logits_output.customized_info,
        releases KV cache, and marks requests as finished.
        """
        logits_output = result.logits_output

        for i, req in enumerate(batch.reqs):
            req.needs_flow_matching = False

            # Collect trajectory into req.customized_info using the standard helper.
            # maybe_collect_customized_info does req.customized_info["pred_traj"].append(v[i]),
            # matching the format expected by stream_output_generation.
            self.maybe_collect_customized_info(i, req, logits_output)

            # Flow matching is a one-shot operation; mark request finished
            req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
            req.time_stats.set_completion_time()
            if req.req_pool_idx is not None:
                release_kv_cache(req, self.tree_cache)

        self.stream_output(batch.reqs, return_logprob=False)
