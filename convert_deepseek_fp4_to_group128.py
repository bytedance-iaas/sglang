#!/usr/bin/env python3
"""把 DeepSeek 风格的 packed FP4 权重重新量化为 group_size=128。

这个脚本对应的是：
1. 源权重仍然是 FP4（E2M1-like codebook，两个 4-bit 打包到一个 int8 字节）
2. 源 scale 是按较小 group（默认 32）提供
3. 目标仍然输出 FP4，只是把 scale 粒度改成更大的 group（默认 128）

脚本会处理：
- `layers.*.ffn.experts.*.weight`
- `layers.*.ffn.shared_experts.*.weight`

注意：
- 这不是 AWQ/GPTQ/Marlin 那类 `bits=4, group_size=128` 的线性 INT4 格式转换。
- 如果目标后端期待的是 `qweight/qzeros/scales/g_idx` 一类张量布局，本脚本不能直接满足，
  那种场景需要“反量化到浮点 -> 走目标量化器重新量化”。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import traceback
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file


FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

FP4_ABS_THRESHOLDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    dtype=torch.float32,
)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolve_devices(
    device_arg: str,
    devices_arg: str | None,
    num_gpu_workers: int,
) -> list[str]:
    if devices_arg:
        devices = [item.strip() for item in devices_arg.split(",") if item.strip()]
        if not devices:
            raise ValueError("--devices was set but no valid device was provided")
        return [str(resolve_device(device)) for device in devices]

    if num_gpu_workers < 1:
        raise ValueError(f"--num-gpu-workers must be >= 1, got {num_gpu_workers}")
    if num_gpu_workers == 1:
        return [str(resolve_device(device_arg))]

    if not torch.cuda.is_available():
        raise RuntimeError("--num-gpu-workers > 1 requires CUDA")
    device_count = torch.cuda.device_count()
    if device_count < num_gpu_workers:
        raise RuntimeError(
            f"Requested {num_gpu_workers} GPU workers, but only {device_count} "
            "CUDA device(s) are visible"
        )
    return [f"cuda:{idx}" for idx in range(num_gpu_workers)]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def looks_like_fp8_config(src_dir: Path) -> bool:
    config_path = src_dir / "config.json"
    if not config_path.exists():
        return False
    config = load_json(config_path)
    quant_config = config.get("quantization_config", {})
    return (
        quant_config.get("quant_method") == "fp8"
        or quant_config.get("fmt") == "e4m3"
        or quant_config.get("weight_block_size") == [128, 128]
    )


def is_fp4_expert_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        (".ffn.experts." in name or ".ffn.shared_experts." in name)
        and name.endswith(".weight")
        and tensor.dtype == torch.int8
        and tensor.ndim == 2
    )


def unpack_fp4_values(packed: torch.Tensor, fp4_table: torch.Tensor) -> torch.Tensor:
    """把 packed int8 中的两个 nibble 解成 FP4 码本值。"""
    if packed.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError(f"expected int8 2D packed tensor, got {packed.dtype}, {packed.shape}")

    out_dim, packed_in_dim = packed.shape
    packed_u8 = packed.view(torch.uint8)
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    values = torch.stack(
        (fp4_table[low.long()], fp4_table[high.long()]),
        dim=-1,
    ).reshape(out_dim, packed_in_dim * 2)
    return values


def dequant_fp4_grouped(
    packed: torch.Tensor,
    scale: torch.Tensor,
    src_group_size: int,
    fp4_table: torch.Tensor,
) -> torch.Tensor:
    """按源 group_size 反量化得到实值张量。"""
    fp4 = unpack_fp4_values(packed, fp4_table)
    out_dim, in_dim = fp4.shape
    expected_scale_shape = (out_dim, in_dim // src_group_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape mismatch: got {tuple(scale.shape)}, expected {expected_scale_shape}"
        )

    grouped = fp4.view(out_dim, in_dim // src_group_size, src_group_size)
    real = grouped * scale.float().unsqueeze(-1)
    return real.reshape(out_dim, in_dim)


def nearest_fp4_nibble(
    normalized: torch.Tensor,
    abs_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Map normalized values to nearest E2M1-like FP4 code without 16-way distance."""
    abs_value = normalized.abs()
    level = torch.bucketize(abs_value.contiguous(), abs_thresholds).to(torch.uint8)
    negative = normalized < 0
    signed_level = level + negative.to(torch.uint8) * 8
    # FP4_TABLE has +0 at code 0 and -0 at code 8. The old argmin path chose
    # code 0 on zero ties, so keep all near-zero values canonicalized to 0.
    return torch.where(level == 0, level, signed_level)


def quantize_real_to_fp4(
    real: torch.Tensor,
    dst_group_size: int,
    fp4_abs_thresholds: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把实值重新量化到 FP4，并生成新的 group scale。"""
    if real.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(real.shape)}")

    out_dim, in_dim = real.shape
    if in_dim % dst_group_size != 0:
        raise ValueError(f"in_dim={in_dim} must be divisible by dst_group_size={dst_group_size}")

    num_groups = in_dim // dst_group_size
    grouped = real.view(out_dim, num_groups, dst_group_size)

    # FP4_TABLE 的最大幅值是 6.0，所以新的 scale 用 max_abs / 6。
    max_abs = grouped.abs().amax(dim=-1)
    new_scale = torch.clamp(max_abs / 6.0, min=eps)

    normalized = grouped / new_scale.unsqueeze(-1)
    nibble = nearest_fp4_nibble(normalized, fp4_abs_thresholds)

    # 打包回 int8：偶数列放低 4 bit，奇数列放高 4 bit。
    nibble = nibble.view(out_dim, in_dim)
    if in_dim % 2 != 0:
        raise ValueError(f"in_dim={in_dim} must be even for nibble packing")
    low = nibble[:, 0::2]
    high = nibble[:, 1::2]
    packed_u8 = low | (high << 4)
    packed_i8 = packed_u8.view(torch.int8)
    return packed_i8, new_scale.float()


def regroup_fp4_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    src_group_size: int,
    dst_group_size: int,
    fp4_table: torch.Tensor,
    fp4_abs_thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    real = dequant_fp4_grouped(
        packed,
        scale,
        src_group_size=src_group_size,
        fp4_table=fp4_table,
    )
    return quantize_real_to_fp4(
        real,
        dst_group_size=dst_group_size,
        fp4_abs_thresholds=fp4_abs_thresholds,
    )


def convert_config(src_dir: Path, out_dir: Path, dst_group_size: int) -> None:
    config_path = src_dir / "config.json"
    if not config_path.exists():
        return

    config = load_json(config_path)
    quant_config = config.setdefault("quantization_config", {})
    quant_config["group_size"] = dst_group_size
    quant_config["fp4_group_size"] = dst_group_size
    quant_config["quant_method"] = "fp8"
    quant_config["activation_scheme"] = "dynamic"
    quant_config.pop("weight_block_size", None)
    quant_config.pop("fmt", None)
    quant_config.pop("scale_fmt", None)
    save_json(out_dir / "config.json", config)


def should_skip_copy(rel_path: Path) -> bool:
    name = rel_path.name
    if rel_path == Path("config.json"):
        return True
    if rel_path == Path("model.safetensors.index.json"):
        return True
    if name.startswith("model-") and name.endswith(".safetensors"):
        return True
    return False


def copy_non_weight_artifacts(src_dir: Path, out_dir: Path) -> None:
    for path in sorted(src_dir.rglob("*")):
        rel = path.relative_to(src_dir)
        if rel == Path("."):
            continue
        if should_skip_copy(rel):
            continue

        dst = out_dir / rel
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def read_existing_shard_keys_and_size(path: Path) -> tuple[set[str], int]:
    with safe_open(path, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        total_size = 0
        for key in keys:
            tensor = reader.get_tensor(key)
            total_size += tensor.numel() * tensor.element_size()
    return keys, total_size


def convert_shard(
    *,
    file_idx: int,
    total_files: int,
    file_name: str,
    names: list[str],
    src_dir: Path,
    out_dir: Path,
    src_group_size: int,
    dst_group_size: int,
    device: torch.device,
    fp4_table: torch.Tensor,
    fp4_abs_thresholds: torch.Tensor,
    resume: bool,
    progress_every: int,
) -> tuple[dict[str, str], int]:
    src_file = src_dir / file_name
    out_file = out_dir / file_name
    expected_names = set(names)

    if resume and out_file.exists():
        existing_keys, existing_size = read_existing_shard_keys_and_size(out_file)
        if expected_names.issubset(existing_keys):
            if file_idx == 1 or file_idx % progress_every == 0 or file_idx == total_files:
                print(
                    f"[{file_idx}/{total_files}] skipping existing completed shard: "
                    f"{file_name}",
                    flush=True,
                )
            return {name: file_name for name in expected_names}, existing_size
        print(
            f"[{file_idx}/{total_files}] existing shard incomplete, regenerating: "
            f"{file_name}",
            flush=True,
        )
    elif file_idx == 1 or file_idx % progress_every == 0 or file_idx == total_files:
        print(
            f"[{file_idx}/{total_files}] converting shard on {device}: {file_name}",
            flush=True,
        )

    state_dict: dict[str, torch.Tensor] = {}
    emitted_map: dict[str, str] = {}
    with safe_open(src_file, framework="pt", device="cpu") as reader:
        key_set = set(reader.keys())
        consumed: set[str] = set()
        for name in names:
            if name in consumed:
                continue
            if name not in key_set:
                continue
            tensor = reader.get_tensor(name)

            if is_fp4_expert_weight(name, tensor):
                scale_name = name.removesuffix(".weight") + ".scale"
                if scale_name not in key_set:
                    raise KeyError(f"missing scale tensor for {name}: {scale_name}")
                tensor_on_device = tensor.to(device, non_blocking=True)
                scale_on_device = reader.get_tensor(scale_name).to(
                    device, non_blocking=True
                )
                new_weight, new_scale = regroup_fp4_tensor(
                    tensor_on_device,
                    scale_on_device,
                    src_group_size=src_group_size,
                    dst_group_size=dst_group_size,
                    fp4_table=fp4_table,
                    fp4_abs_thresholds=fp4_abs_thresholds,
                )
                state_dict[name] = new_weight.cpu()
                state_dict[scale_name] = new_scale.cpu()
                emitted_map[name] = file_name
                emitted_map[scale_name] = file_name
                consumed.add(scale_name)
                del tensor_on_device, scale_on_device, new_weight, new_scale
                continue

            if name.endswith(".scale"):
                # 非 expert 的 scale 先原样保留；如果目标后端也要求改布局，需要再单独处理。
                state_dict[name] = tensor.float()
            else:
                state_dict[name] = tensor
            emitted_map[name] = file_name

    shard_size = sum(
        tensor.numel() * tensor.element_size() for tensor in state_dict.values()
    )
    save_file(state_dict, out_file, metadata={"format": "pt"})
    return emitted_map, shard_size


def convert_shards_on_device(
    *,
    worker_id: int,
    device_arg: str,
    tasks: list[tuple[int, int, str, list[str]]],
    src_dir: str,
    out_dir: str,
    src_group_size: int,
    dst_group_size: int,
    resume: bool,
    progress_every: int,
    empty_cache_every: int,
    result_queue: mp.Queue,
) -> None:
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        device = resolve_device(device_arg)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        fp4_table = FP4_TABLE.to(device)
        fp4_abs_thresholds = FP4_ABS_THRESHOLDS.to(device)

        for local_idx, (file_idx, total_files, file_name, names) in enumerate(
            tasks, start=1
        ):
            emitted_map, shard_size = convert_shard(
                file_idx=file_idx,
                total_files=total_files,
                file_name=file_name,
                names=names,
                src_dir=Path(src_dir),
                out_dir=Path(out_dir),
                src_group_size=src_group_size,
                dst_group_size=dst_group_size,
                device=device,
                fp4_table=fp4_table,
                fp4_abs_thresholds=fp4_abs_thresholds,
                resume=resume,
                progress_every=progress_every,
            )
            result_queue.put(("result", file_name, emitted_map, shard_size))
            if (
                device.type == "cuda"
                and empty_cache_every > 0
                and local_idx % empty_cache_every == 0
            ):
                torch.cuda.empty_cache()

        result_queue.put(("done", worker_id))
    except BaseException:
        result_queue.put(("error", worker_id, traceback.format_exc()))


def run_parallel_conversion(
    *,
    devices: list[str],
    tasks: list[tuple[int, int, str, list[str]]],
    src_dir: Path,
    out_dir: Path,
    src_group_size: int,
    dst_group_size: int,
    resume: bool,
    progress_every: int,
    empty_cache_every: int,
) -> tuple[dict[str, str], dict[str, int]]:
    if len(devices) == 1:
        device = resolve_device(devices[0])
        if device.type == "cuda":
            torch.cuda.set_device(device)
        fp4_table = FP4_TABLE.to(device)
        fp4_abs_thresholds = FP4_ABS_THRESHOLDS.to(device)
        emitted_map: dict[str, str] = {}
        shard_sizes: dict[str, int] = {}
        for local_idx, (file_idx, total_files, file_name, names) in enumerate(
            tasks, start=1
        ):
            shard_emitted_map, shard_size = convert_shard(
                file_idx=file_idx,
                total_files=total_files,
                file_name=file_name,
                names=names,
                src_dir=src_dir,
                out_dir=out_dir,
                src_group_size=src_group_size,
                dst_group_size=dst_group_size,
                device=device,
                fp4_table=fp4_table,
                fp4_abs_thresholds=fp4_abs_thresholds,
                resume=resume,
                progress_every=progress_every,
            )
            emitted_map.update(shard_emitted_map)
            shard_sizes[file_name] = shard_size
            if (
                device.type == "cuda"
                and empty_cache_every > 0
                and local_idx % empty_cache_every == 0
            ):
                torch.cuda.empty_cache()
        return emitted_map, shard_sizes

    task_groups: list[list[tuple[int, int, str, list[str]]]] = [
        [] for _ in devices
    ]
    for idx, task in enumerate(tasks):
        task_groups[idx % len(devices)].append(task)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes: list[mp.Process] = []
    for worker_id, (device_arg, worker_tasks) in enumerate(zip(devices, task_groups)):
        if not worker_tasks:
            continue
        process = ctx.Process(
            target=convert_shards_on_device,
            kwargs={
                "worker_id": worker_id,
                "device_arg": device_arg,
                "tasks": worker_tasks,
                "src_dir": str(src_dir),
                "out_dir": str(out_dir),
                "src_group_size": src_group_size,
                "dst_group_size": dst_group_size,
                "resume": resume,
                "progress_every": progress_every,
                "empty_cache_every": empty_cache_every,
                "result_queue": result_queue,
            },
        )
        process.start()
        processes.append(process)

    emitted_map: dict[str, str] = {}
    shard_sizes: dict[str, int] = {}
    done_workers = 0
    try:
        while done_workers < len(processes):
            message = result_queue.get()
            kind = message[0]
            if kind == "result":
                _, file_name, shard_emitted_map, shard_size = message
                emitted_map.update(shard_emitted_map)
                shard_sizes[file_name] = shard_size
            elif kind == "done":
                done_workers += 1
            elif kind == "error":
                _, worker_id, error_text = message
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                raise RuntimeError(
                    f"conversion worker {worker_id} failed:\n{error_text}"
                )
            else:
                raise RuntimeError(f"unknown worker message: {message!r}")
    finally:
        for process in processes:
            process.join()

    failed = [process.exitcode for process in processes if process.exitcode not in (0,)]
    if failed:
        raise RuntimeError(f"conversion worker exit code(s): {failed}")
    return emitted_map, shard_sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert packed FP4 tensors to group_size=128 FP4.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--src-group-size", type=int, default=32)
    parser.add_argument("--dst-group-size", type=int, default=128)
    parser.add_argument(
        "--device",
        default="auto",
        help="转换计算设备：auto/cpu/cuda/cuda:0 等。默认 auto，有 CUDA 就用 CUDA。",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "逗号分隔的转换设备列表，例如 cuda:0,cuda:1。设置后会覆盖 --device "
            "和 --num-gpu-workers。"
        ),
    )
    parser.add_argument(
        "--num-gpu-workers",
        type=int,
        default=1,
        help="使用前 N 张 CUDA GPU 并行转换。默认 1，设置为 8 会使用 cuda:0 到 cuda:7。",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help="CUDA 模式下每处理多少个 shard 调用一次 torch.cuda.empty_cache()，0 表示不主动清理。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑。若输出 shard 已存在且包含该 shard 所需全部 tensor，则直接跳过。",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="每处理多少个 shard 打印一次进度，默认每个 shard 都打印。",
    )
    parser.add_argument(
        "--skip-input-format-check",
        action="store_true",
        help="跳过 config.json 的输入格式检查。仅当你确认目录虽然写着 fp8，但实际 expert 权重仍是 packed FP4 时使用。",
    )
    args = parser.parse_args()

    src_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    devices = resolve_devices(args.device, args.devices, args.num_gpu_workers)

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    if looks_like_fp8_config(src_dir) and not args.skip_input_format_check:
        raise ValueError(
            "input config.json looks like an FP8 checkpoint already. "
            "If your actual expert weights are still packed FP4 and only config.json is misleading, "
            "rerun with --skip-input-format-check."
        )

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to continue")
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_json(index_path)
    weight_map = index["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, file_name in weight_map.items():
        by_file.setdefault(file_name, []).append(name)

    ordered_files = sorted(by_file)
    total_files = len(ordered_files)
    tasks = [
        (file_idx, total_files, file_name, by_file[file_name])
        for file_idx, file_name in enumerate(ordered_files, start=1)
    ]
    print(
        f"starting conversion: {total_files} shard(s), devices={','.join(devices)}",
        flush=True,
    )
    emitted_map, shard_sizes = run_parallel_conversion(
        devices=devices,
        tasks=tasks,
        src_dir=src_dir,
        out_dir=out_dir,
        src_group_size=args.src_group_size,
        dst_group_size=args.dst_group_size,
        resume=args.resume,
        progress_every=args.progress_every,
        empty_cache_every=args.empty_cache_every,
    )

    total_size = sum(shard_sizes.values())

    save_json(
        out_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(emitted_map.items())),
        },
    )
    convert_config(src_dir, out_dir, args.dst_group_size)
    copy_non_weight_artifacts(src_dir, out_dir)

    print(f"converted checkpoint written to {out_dir}")


if __name__ == "__main__":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    main()
