import pytest
import torch

from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=5, stage="base-b-test-small-1-gpu")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA graph capture"
)


class _Leaf:
    def __init__(self):
        self.forward_metadata = None


class _Backend:
    attn_backend_list = None

    def __init__(self, source, output):
        self.source = source
        self.output = output
        self.leaf = _Leaf()
        self.calls = 0

    def init_forward_metadata_out_graph(self, forward_batch):
        self.calls += 1
        self.output.copy_(self.source * forward_batch.multiplier)
        self.leaf.forward_metadata = {"call": self.calls, "output": self.output}


def test_metadata_glue_graph_capture_replay_reset():
    source = torch.tensor([1.0, 2.0], device="cuda")
    output = torch.zeros_like(source)
    backend = _Backend(source, output)
    glue = MetadataGlueGraph("cuda", leaves=[backend.leaf])
    view = type("View", (), {"multiplier": 3.0})()

    # Two warmups, then capture+replay.
    glue.run(backend, view, key=4)
    glue.run(backend, view, key=4)
    glue.run(backend, view, key=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.tensor([3.0, 6.0], device="cuda"))
    assert backend.calls == 3
    captured_metadata = backend.leaf.forward_metadata

    source.copy_(torch.tensor([4.0, 5.0], device="cuda"))
    backend.leaf.forward_metadata = None
    glue.run(backend, view, key=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.tensor([12.0, 15.0], device="cuda"))
    assert backend.calls == 3
    assert backend.leaf.forward_metadata is captured_metadata

    glue.reset()
    glue.run(backend, view, key=4)
    assert backend.calls == 4
