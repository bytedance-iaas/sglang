import unittest

from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    BatchTokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import SenderWrapper
from sglang.srt.managers.scheduler import Scheduler, SenderWrapper as SchedulerSender
from sglang.srt.sampling.sampling_params import SamplingParams


class DummySocket:
    def __init__(self):
        self.obj = None

    def send_pyobj(self, obj):
        self.obj = obj


def make_generate_req(http_worker_ipc=None):
    return TokenizedGenerateReqInput(
        input_text="",
        input_ids=[1, 2],
        mm_inputs={},
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=[],
        stream=False,
        http_worker_ipc=http_worker_ipc,
    )


def make_embedding_req(http_worker_ipc=None):
    return TokenizedEmbeddingReqInput(
        input_text="",
        input_ids=[1, 2],
        image_inputs={},
        token_type_ids=None,
        sampling_params=SamplingParams(),
        http_worker_ipc=http_worker_ipc,
    )


class TestMultiTokenizerIpcRouting(unittest.TestCase):
    def test_sender_wrapper_attaches_ipc_to_batch_and_items(self):
        socket = DummySocket()
        wrapper = SenderWrapper(
            port_args=type("PortArgs", (), {"tokenizer_ipc_name": "ipc://worker-0"}),
            send_to_scheduler=socket,
        )
        batch = BatchTokenizedGenerateReqInput(batch=[make_generate_req()])

        wrapper.send_pyobj(batch)

        self.assertEqual(socket.obj.http_worker_ipcs, ["ipc://worker-0"])
        self.assertEqual(socket.obj.batch[0].http_worker_ipc, "ipc://worker-0")

    def test_scheduler_batch_generate_uses_batch_ipc_fallback(self):
        scheduler = Scheduler.__new__(Scheduler)
        seen = []

        def handle_generate_request(req):
            seen.append(req)

        scheduler.handle_generate_request = handle_generate_request
        batch = BatchTokenizedGenerateReqInput(
            batch=[make_generate_req()],
            http_worker_ipcs=["ipc://worker-1"],
        )

        Scheduler.handle_batch_generate_request(scheduler, batch)

        self.assertEqual(seen[0].http_worker_ipc, "ipc://worker-1")

    def test_scheduler_batch_embedding_uses_batch_ipc_fallback(self):
        scheduler = Scheduler.__new__(Scheduler)
        seen = []

        def handle_embedding_request(req):
            seen.append(req)

        scheduler.handle_embedding_request = handle_embedding_request
        batch = BatchTokenizedEmbeddingReqInput(
            batch=[make_embedding_req()],
            http_worker_ipcs=["ipc://worker-2"],
        )

        Scheduler.handle_batch_embedding_request(scheduler, batch)

        self.assertEqual(seen[0].http_worker_ipc, "ipc://worker-2")

    def test_scheduler_sender_copies_batch_ipc_to_immediate_output(self):
        socket = DummySocket()
        wrapper = SchedulerSender(socket)
        recv_obj = BatchTokenizedGenerateReqInput(
            batch=[make_generate_req()],
            http_worker_ipcs=["ipc://worker-3"],
        )
        output = BatchTokenizedGenerateReqInput(batch=[make_generate_req()])

        wrapper.send_output(output, recv_obj)

        self.assertEqual(socket.obj.http_worker_ipcs, ["ipc://worker-3"])


if __name__ == "__main__":
    unittest.main()
