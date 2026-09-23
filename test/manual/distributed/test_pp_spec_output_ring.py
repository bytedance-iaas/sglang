"""Exercise the PP speculative output ring with real CUDA P2P work.

Run with two GPUs and no model weights:
    python -m torch.distributed.run --standalone --nproc-per-node=2 \
        test/manual/distributed/test_pp_spec_output_ring.py

The five tensor shapes and dtypes reproduce a GLM PD-prefill output relay.
Both ranks have an output to send in the same slot. Sending every tensor
before posting any receive deadlocks even when host-side isend is asynchronous.
"""

import datetime
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    scheduler = SchedulerPPMixin()
    scheduler.ps = SimpleNamespace(pp_rank=rank)
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    scheduler.schedule_stream = torch.cuda.current_stream()
    scheduler.copy_stream = torch.cuda.Stream()
    scheduler.copy_stream_ctx = torch.cuda.stream(scheduler.copy_stream)
    scheduler.device_module = torch.cuda
    shapes = [(1,), (1, 1), (1, 1), (768,), (256,)]
    dtypes = [torch.int64, torch.float32, torch.int64, torch.bfloat16, torch.int32]
    values = {
        str(i): torch.full(shape, rank + i + 1, dtype=dtype, device="cuda")
        for i, (shape, dtype) in enumerate(zip(shapes, dtypes))
    }
    received = {}

    def send(*args):
        return [dist.isend(t, dst=1 - rank) for t in values.values()]

    def recv():
        nonlocal received
        received = {}
        for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
            t = torch.empty(shape, dtype=dtype, device="cuda")
            dist.irecv(t, src=1 - rank).wait()
            received[str(i)] = t
        return received

    scheduler._pp_send_output_to_next_stage = send
    scheduler._pp_recv_dict_from_prev_stage = recv
    scheduler._pp_prep_batch_result = lambda *args: None
    target = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reqs=[object()],
        contains_last_prefill_chunk=True,
        return_logprob=False,
    )
    torch.cuda.synchronize()
    dist.barrier()
    print(f"RING_START rank={rank}", flush=True)
    for iteration in range(64):
        _, _, event, works = scheduler._pp_send_recv_and_preprocess_output_tensors(
            0, 1, [target, target], [None, None], [], None
        )
        event.synchronize()
        for work in works:
            work.wait()
        torch.cuda.synchronize()
        for i, tensor in enumerate(received.values()):
            assert bool(torch.all(tensor == (1 - rank) + i + 1)), (rank, iteration, i)
    print(f"RING_PASS rank={rank} rounds=64 tensors_per_round=5", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
