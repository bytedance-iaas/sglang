"""Contract for the numerical probe's independent draft-graph gate."""

from types import SimpleNamespace

from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_dp_draft_graph_gate_is_independent_from_target_graph_gate():
    runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
    runner.captured_req_width = 1
    runner.require_mlp_tp_gather = False
    runner.require_mlp_sync = True
    runner.disable_padding = True
    runner.backend = SimpleNamespace(can_run=lambda *_args: True)
    runner._make_graph_key = lambda batch_size: batch_size

    forward_batch = SimpleNamespace(
        batch_size=4,
        spec_info=SimpleNamespace(num_tokens_per_req=1),
        can_run_dp_cuda_graph=True,
        can_run_dp_draft_cuda_graph=False,
    )
    assert runner.can_run_graph(forward_batch) is False

    forward_batch.can_run_dp_draft_cuda_graph = True
    assert runner.can_run_graph(forward_batch) is True
