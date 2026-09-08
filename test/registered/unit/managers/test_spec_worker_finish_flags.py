"""Finish-state contracts for request-scoped speculative observers."""

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    _spec_worker_finish_flags,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_spec_worker_finish_flags_distinguish_stop_length_and_abort():
    assert _spec_worker_finish_flags(FINISH_MATCHED_TOKEN(matched=1)) == (True, True)
    assert _spec_worker_finish_flags(FINISH_LENGTH(length=512)) == (False, True)
    assert _spec_worker_finish_flags(FINISH_ABORT(message="cancelled")) == (
        False,
        False,
    )
