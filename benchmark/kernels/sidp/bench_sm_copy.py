#!/usr/bin/env python3
"""Benchmark the production SiDP SM-copy JIT kernel across launch shapes.

Each GPU process owns one source allocation and pulls one peer's allocation
into a local destination through raw CUDA IPC. The benchmark compares:

* CUDA DMA (``cudaMemcpyAsync``) over the same IPC mapping; and
* SiDP's production ``copy_selected_kernel`` while scanning CTA/block counts.

The default ring pattern avoids owner incast so this benchmark isolates the
copy backend and launch geometry. Use ``--peer-pattern incast`` only when the
intended experiment is bandwidth contention rather than raw copy efficiency.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import socket
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable


SGLANG_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SGLANG_ROOT / "python"))

import torch  # noqa: E402

from sglang.kernels.ops.sidp import (  # noqa: E402
    copy_selected,
    load_sidp_sm_copy_module,
)
from sglang.srt.layers.sidp.cuda_memcpy import SidpCudaMemcpy  # noqa: E402


SUPPORTED_BLOCK_SIZES = (128, 256, 512)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_positive_csv(raw: str, option: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{option} must be a comma-separated integer list"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{option} values must all be positive")
    return list(dict.fromkeys(values))


def _default_ctas(sm_count: int) -> list[int]:
    fractions = [
        math.ceil(sm_count / 16),
        math.ceil(sm_count / 8),
        math.ceil(sm_count / 4),
        math.ceil(sm_count / 2),
    ]
    return sorted(set(fractions + [sm_count, 2 * sm_count, 4 * sm_count]))


def _peer_for(rank: int, world_size: int, pattern: str) -> int:
    if pattern == "ring":
        return (rank + 1) % world_size
    # No process can IPC-open its own allocation. Rank 0 uses rank 1 while all
    # other requesters read rank 0, producing a deliberate owner-0 incast.
    return 1 if rank == 0 else 0


def _timed_launch(
    launch: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    world_size: int,
) -> list[float]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    torch.distributed.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        launch()
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))

    elapsed_by_rank: list[float | None] = [None] * world_size
    torch.distributed.all_gather_object(elapsed_by_rank, elapsed_ms)
    return [float(value) for value in elapsed_by_rank if value is not None]


def _bandwidth_result(
    elapsed_ms: list[float], nbytes: int, iterations: int, world_size: int
) -> dict[str, object]:
    rank_gbps = [
        iterations * nbytes / (rank_ms / 1000.0) / 1e9
        for rank_ms in elapsed_ms
    ]
    # Max rank duration is the completion time of the concurrent all-rank copy.
    aggregate_gbps = (
        world_size * iterations * nbytes / (max(elapsed_ms) / 1000.0) / 1e9
    )
    return {
        "elapsed_ms_by_rank": elapsed_ms,
        "elapsed_ms_max": max(elapsed_ms),
        "rank_gbps": rank_gbps,
        "rank_gbps_mean": statistics.mean(rank_gbps),
        "rank_gbps_min": min(rank_gbps),
        "rank_gbps_max": max(rank_gbps),
        "aggregate_gbps": aggregate_gbps,
    }


def _worker(
    rank: int,
    world_size: int,
    port: int,
    buffer_bytes: int,
    iterations: int,
    warmup: int,
    cta_values: list[int],
    block_sizes: list[int],
    peer_pattern: str,
    output_path: str,
    command: str,
) -> None:
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=300),
    )
    copier = SidpCudaMemcpy()
    source = torch.full(
        (buffer_bytes // 4,), rank + 1, dtype=torch.int32, device=rank
    )
    destination = torch.zeros_like(source)
    local_descriptor = copier.export_ipc_pointer(source.data_ptr(), source.nbytes)
    descriptors: list[dict | None] = [None] * world_size
    torch.distributed.all_gather_object(descriptors, local_descriptor)

    peer = _peer_for(rank, world_size, peer_pattern)
    copier.enable_peer_access(peer)
    peer_descriptor = descriptors[peer]
    if peer_descriptor is None:
        raise RuntimeError(f"rank {rank} did not receive rank {peer}'s IPC handle")
    peer_ptr = copier.open_ipc_pointer(peer_descriptor)

    source_ptrs = torch.tensor([peer_ptr], dtype=torch.uint64, device=rank)
    destination_ptrs = torch.tensor(
        [destination.data_ptr()], dtype=torch.uint64, device=rank
    )
    sizes = torch.tensor([buffer_bytes], dtype=torch.int64, device=rank)
    selected = torch.zeros(1, dtype=torch.int32, device=rank)
    error_state = torch.zeros(1, dtype=torch.int32, device=rank)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Compile/load before any measurement, then validate the same indirect
    # descriptor path used by SiDP runtime.
    load_sidp_sm_copy_module()
    copy_selected(
        source_ptrs,
        destination_ptrs,
        sizes,
        selected,
        cta_values[0],
        block_sizes[0],
        error_state,
    )
    torch.cuda.synchronize()
    expected = peer + 1
    if not bool(torch.all(destination == expected).item()):
        raise RuntimeError(
            f"SM-copy validation failed at rank {rank}: expected value {expected}"
        )
    if error_state.item() != 0:
        raise RuntimeError(
            f"SM-copy error state is non-zero at rank {rank}: "
            f"{error_state.item()}"
        )
    torch.distributed.barrier()

    dma_elapsed = _timed_launch(
        lambda: copier.async_copy(
            destination.data_ptr(), peer_ptr, buffer_bytes, stream_ptr
        ),
        warmup=warmup,
        iterations=iterations,
        world_size=world_size,
    )
    dma = _bandwidth_result(dma_elapsed, buffer_bytes, iterations, world_size)

    sm_results: list[dict[str, object]] = []
    for block_size in block_sizes:
        for ctas in cta_values:
            elapsed = _timed_launch(
                lambda ctas=ctas, block_size=block_size: copy_selected(
                    source_ptrs,
                    destination_ptrs,
                    sizes,
                    selected,
                    ctas,
                    block_size,
                    error_state,
                ),
                warmup=warmup,
                iterations=iterations,
                world_size=world_size,
            )
            result = _bandwidth_result(
                elapsed, buffer_bytes, iterations, world_size
            )
            result.update(
                {
                    "ctas": ctas,
                    "block_size": block_size,
                    "sm_to_dma": (
                        float(result["aggregate_gbps"])
                        / float(dma["aggregate_gbps"])
                    ),
                }
            )
            sm_results.append(result)

    if rank == 0:
        properties = torch.cuda.get_device_properties(0)
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "command": command,
            "device": properties.name,
            "device_sm_count": properties.multi_processor_count,
            "world_size": world_size,
            "peer_pattern": peer_pattern,
            "buffer_bytes_per_rank": buffer_bytes,
            "iterations": iterations,
            "warmup": warmup,
            "cta_values": cta_values,
            "block_sizes": block_sizes,
            "dma": dma,
            "sm": sm_results,
        }
        resolved_output = Path(output_path)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print("\nSiDP raw IPC copy bandwidth")
        print(
            f"device={properties.name}, GPUs={world_size}, "
            f"SMs/GPU={properties.multi_processor_count}, "
            f"buffer={buffer_bytes / 2**20:.1f} MiB, pattern={peer_pattern}"
        )
        print(
            f"DMA baseline: aggregate={float(dma['aggregate_gbps']):.2f} GB/s, "
            f"rank_mean={float(dma['rank_gbps_mean']):.2f} GB/s"
        )
        print()
        print(" block   CTAs   aggregate GB/s   rank mean GB/s   SM/DMA")
        print(" -----  -----   --------------   --------------   ------")
        for result in sm_results:
            print(
                f" {int(result['block_size']):>5}  "
                f"{int(result['ctas']):>5}   "
                f"{float(result['aggregate_gbps']):>14.2f}   "
                f"{float(result['rank_gbps_mean']):>14.2f}   "
                f"{float(result['sm_to_dma']):>6.3f}"
            )
        print(f"\nJSON report: {resolved_output}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of local GPU processes (default: 8)",
    )
    parser.add_argument(
        "--buffer-mb",
        type=int,
        default=32,
        help="Bytes copied by each GPU per iteration, in MiB (default: 32)",
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Timed launches per setting"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Untimed launches per setting"
    )
    parser.add_argument(
        "--cta-values",
        default="",
        help=(
            "Comma-separated absolute CTA counts. The default scans "
            "ceil(SM/{16,8,4,2}), SM, 2xSM and 4xSM."
        ),
    )
    parser.add_argument(
        "--block-sizes",
        default="128,256,512",
        help="Comma-separated block sizes from {128,256,512}",
    )
    parser.add_argument(
        "--peer-pattern",
        choices=("ring", "incast"),
        default="ring",
        help="ring isolates raw bandwidth; incast deliberately contends on owner 0",
    )
    parser.add_argument(
        "--output",
        help="JSON output path (default: check_logs/sidp_sm_copy_benchmark_<time>.json)",
    )
    args = parser.parse_args()

    if args.num_gpus < 2:
        parser.error("--num-gpus must be at least 2 for peer copy")
    if args.num_gpus > torch.cuda.device_count():
        parser.error(
            f"--num-gpus={args.num_gpus} exceeds visible GPU count "
            f"{torch.cuda.device_count()}"
        )
    if args.buffer_mb <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("buffer/iterations must be positive and warmup non-negative")

    args.block_sizes = _parse_positive_csv(args.block_sizes, "--block-sizes")
    unsupported = sorted(set(args.block_sizes) - set(SUPPORTED_BLOCK_SIZES))
    if unsupported:
        parser.error(
            f"unsupported --block-sizes {unsupported}; choose from "
            f"{SUPPORTED_BLOCK_SIZES}"
        )
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    args.cta_values = (
        _parse_positive_csv(args.cta_values, "--cta-values")
        if args.cta_values
        else _default_ctas(sm_count)
    )
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"check_logs/sidp_sm_copy_benchmark_{timestamp}.json"
    return args


def main() -> int:
    args = parse_args()
    buffer_bytes = args.buffer_mb * 2**20
    if buffer_bytes % 16:
        raise ValueError("buffer size must be 16-byte aligned")
    command = " ".join(shlex.quote(value) for value in sys.argv)
    torch.multiprocessing.spawn(
        _worker,
        args=(
            args.num_gpus,
            _free_tcp_port(),
            buffer_bytes,
            args.iterations,
            args.warmup,
            args.cta_values,
            args.block_sizes,
            args.peer_pattern,
            args.output,
            command,
        ),
        nprocs=args.num_gpus,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
