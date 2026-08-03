#!/usr/bin/env python3
"""Prune routed experts from a DeepSeek-V4 Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_EXPERT_RE = re.compile(r"^(?P<owner>.+\.ffn)\.experts\.(?P<id>\d+)\.")
_ROUTER_RE = re.compile(
    r"^(?:model\.)?(?:layers\.\d+|mtp\.\d+)\.ffn\.gate\."
    r"(?P<kind>weight|bias|tid2eid)$"
)
_INDEX_NAME = "model.safetensors.index.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep a prefix of DeepSeek-V4 routed experts while preserving "
            "num_experts_per_tok."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-num-experts", type=int, default=192)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already completed and validated output shards.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def _write_json_atomic(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, path)


def _copy_auxiliary_files(input_dir: Path, output_dir: Path) -> None:
    skipped_files = {"config.json", _INDEX_NAME}
    for source in input_dir.iterdir():
        if source.name in skipped_files or source.suffix == ".safetensors":
            continue
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)
        else:
            shutil.copy2(source, destination)


def _expert_id(name: str) -> int | None:
    match = _EXPERT_RE.match(name)
    return int(match.group("id")) if match is not None else None


def _keep_tensor(name: str, keep_num_experts: int) -> bool:
    expert_id = _expert_id(name)
    return expert_id is None or expert_id < keep_num_experts


def _remap_hash_router(
    tensor: torch.Tensor, *, keep_num_experts: int, num_experts_per_tok: int
) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[1] != num_experts_per_tok:
        raise ValueError(
            "Expected tid2eid shape [vocab_size, num_experts_per_tok], got "
            f"{tuple(tensor.shape)}."
        )
    if keep_num_experts < num_experts_per_tok:
        raise ValueError(
            f"Cannot route {num_experts_per_tok} distinct experts from "
            f"{keep_num_experts} retained experts."
        )

    remapped = tensor.remainder(keep_num_experts)
    for column in range(1, num_experts_per_tok):
        previous = remapped[:, :column]
        collision = (remapped[:, column : column + 1] == previous).any(dim=1)
        while collision.any().item():
            remapped[collision, column] = (remapped[collision, column] + 1).remainder(
                keep_num_experts
            )
            collision = (remapped[:, column : column + 1] == previous).any(dim=1)
    return remapped


def _transform_router_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    source_num_experts: int,
    keep_num_experts: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    match = _ROUTER_RE.match(name)
    if match is None:
        return tensor

    kind = match.group("kind")
    if kind == "tid2eid":
        return _remap_hash_router(
            tensor,
            keep_num_experts=keep_num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

    if tensor.ndim == 0 or tensor.shape[0] != source_num_experts:
        raise ValueError(
            f"Expected {name} dimension 0 to equal {source_num_experts}, "
            f"got shape {tuple(tensor.shape)}."
        )
    return tensor[:keep_num_experts].contiguous()


def _safetensors_payload_size(path: Path) -> int:
    with path.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
    return sum(
        value["data_offsets"][1] - value["data_offsets"][0]
        for name, value in header.items()
        if name != "__metadata__"
    )


def _expected_keys_by_shard(weight_map: dict[str, str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        result.setdefault(shard, set()).add(name)
    return result


def _reconcile_weight_map(
    input_dir: Path, indexed_weight_map: dict[str, str]
) -> dict[str, str]:
    actual_weight_map = {}
    for shard in sorted(set(indexed_weight_map.values())):
        shard_path = input_dir / shard
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            for name in file.keys():  # noqa: SIM118 - safe_open is not iterable.
                previous_shard = actual_weight_map.setdefault(name, shard)
                if previous_shard != shard:
                    raise ValueError(
                        f"Tensor {name} exists in both {previous_shard} and {shard}."
                    )

    indexed_names = set(indexed_weight_map)
    actual_names = set(actual_weight_map)
    stale_names = sorted(indexed_names - actual_names)
    unindexed_names = sorted(actual_names - indexed_names)
    relocated_names = sorted(
        name
        for name in indexed_names & actual_names
        if indexed_weight_map[name] != actual_weight_map[name]
    )
    if stale_names:
        print(
            f"Ignore {len(stale_names)} stale index entries absent from shards: "
            f"{stale_names[:10]}",
            flush=True,
        )
    if unindexed_names:
        print(
            f"Add {len(unindexed_names)} shard tensors missing from the index: "
            f"{unindexed_names[:10]}",
            flush=True,
        )
    if relocated_names:
        print(
            f"Correct {len(relocated_names)} tensors mapped to the wrong shard: "
            f"{relocated_names[:10]}",
            flush=True,
        )
    return actual_weight_map


def _validate_shard_keys(path: Path, expected_keys: set[str]) -> None:
    with safe_open(path, framework="pt", device="cpu") as file:
        actual_keys = set(file.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:10]
        unexpected = sorted(actual_keys - expected_keys)[:10]
        raise ValueError(
            f"{path} key mismatch: missing={missing}, unexpected={unexpected}."
        )


def _process_shard(
    source_path: Path,
    output_path: Path,
    *,
    source_keys: set[str],
    output_keys: set[str],
    source_num_experts: int,
    keep_num_experts: int,
    num_experts_per_tok: int,
) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)

    with safe_open(source_path, framework="pt", device="cpu") as file:
        actual_source_keys = set(file.keys())
        if actual_source_keys != source_keys:
            raise ValueError(
                f"{source_path} keys do not match {_INDEX_NAME}: "
                f"missing={sorted(source_keys - actual_source_keys)[:10]}, "
                f"unexpected={sorted(actual_source_keys - source_keys)[:10]}."
            )
        metadata = file.metadata()
        tensors = {}
        for name in sorted(output_keys):
            tensor = file.get_tensor(name)
            tensors[name] = _transform_router_tensor(
                name,
                tensor,
                source_num_experts=source_num_experts,
                keep_num_experts=keep_num_experts,
                num_experts_per_tok=num_experts_per_tok,
            )

    save_file(tensors, temporary_path, metadata=metadata)
    os.replace(temporary_path, output_path)
    del tensors
    gc.collect()


def _assert_distinct_hash_routes(
    name: str, tensor: torch.Tensor, *, keep_num_experts: int
) -> None:
    if tensor.numel() == 0:
        return
    if tensor.min().item() < 0 or tensor.max().item() >= keep_num_experts:
        raise ValueError(f"{name} contains an out-of-range expert ID.")
    sorted_routes = tensor.sort(dim=1).values
    if (sorted_routes[:, 1:] == sorted_routes[:, :-1]).any().item():
        raise ValueError(f"{name} contains duplicate active experts.")


def _validate_output(
    output_dir: Path,
    *,
    config: dict,
    index: dict,
    keep_num_experts: int,
    num_experts_per_tok: int,
) -> None:
    if config["n_routed_experts"] != keep_num_experts:
        raise ValueError("Output config has the wrong n_routed_experts.")
    if config["num_experts_per_tok"] != num_experts_per_tok:
        raise ValueError("Output config changed num_experts_per_tok.")

    weight_map = index["weight_map"]
    expert_ids_by_owner: dict[str, set[int]] = {}
    for name in weight_map:
        match = _EXPERT_RE.match(name)
        if match is None:
            continue
        expert_id = int(match.group("id"))
        if expert_id >= keep_num_experts:
            raise ValueError(f"Index still references removed expert: {name}.")
        expert_ids_by_owner.setdefault(match.group("owner"), set()).add(expert_id)

    expected_expert_ids = set(range(keep_num_experts))
    for owner, expert_ids in expert_ids_by_owner.items():
        if expert_ids != expected_expert_ids:
            raise ValueError(
                f"{owner} has incomplete expert IDs: "
                f"expected 0..{keep_num_experts - 1}, got "
                f"{min(expert_ids)}..{max(expert_ids)}."
            )

    expected_keys = _expected_keys_by_shard(weight_map)
    payload_size = 0
    for shard, shard_keys in sorted(expected_keys.items()):
        shard_path = output_dir / shard
        _validate_shard_keys(shard_path, shard_keys)
        payload_size += _safetensors_payload_size(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            for name in shard_keys:
                match = _ROUTER_RE.match(name)
                if match is None:
                    continue
                tensor = file.get_tensor(name)
                if match.group("kind") == "tid2eid":
                    if tensor.shape[1] != num_experts_per_tok:
                        raise ValueError(f"{name} changed active expert count.")
                    _assert_distinct_hash_routes(
                        name, tensor, keep_num_experts=keep_num_experts
                    )
                elif tensor.shape[0] != keep_num_experts:
                    raise ValueError(
                        f"{name} has shape {tuple(tensor.shape)}; expected "
                        f"dimension 0 to equal {keep_num_experts}."
                    )

    if index["metadata"]["total_size"] != payload_size:
        raise ValueError(
            f"Index total_size={index['metadata']['total_size']} does not match "
            f"tensor payload size={payload_size}."
        )


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the input directory.")
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{output_dir} is not empty; use --resume or a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_config = _read_json(input_dir / "config.json")
    if source_config["model_type"] != "deepseek_v4":
        raise ValueError(
            f"Expected model_type='deepseek_v4', got {source_config['model_type']!r}."
        )
    source_num_experts = int(source_config["n_routed_experts"])
    keep_num_experts = args.keep_num_experts
    num_experts_per_tok = int(source_config["num_experts_per_tok"])
    if not num_experts_per_tok <= keep_num_experts < source_num_experts:
        raise ValueError(
            f"Expected num_experts_per_tok <= keep_num_experts < "
            f"n_routed_experts, got {num_experts_per_tok} <= "
            f"{keep_num_experts} < {source_num_experts}."
        )

    source_index = _read_json(input_dir / _INDEX_NAME)
    source_weight_map = _reconcile_weight_map(input_dir, source_index["weight_map"])
    output_weight_map = {
        name: shard
        for name, shard in source_weight_map.items()
        if _keep_tensor(name, keep_num_experts)
    }
    source_keys_by_shard = _expected_keys_by_shard(source_weight_map)
    output_keys_by_shard = _expected_keys_by_shard(output_weight_map)

    _copy_auxiliary_files(input_dir, output_dir)
    output_config = dict(source_config)
    output_config["n_routed_experts"] = keep_num_experts
    _write_json_atomic(output_dir / "config.json", output_config)

    for position, shard in enumerate(sorted(source_keys_by_shard), start=1):
        source_path = input_dir / shard
        output_path = output_dir / shard
        output_keys = output_keys_by_shard[shard]
        if args.resume and output_path.exists():
            _validate_shard_keys(output_path, output_keys)
            print(
                f"[{position}/{len(source_keys_by_shard)}] Reuse validated {shard}",
                flush=True,
            )
            continue
        print(
            f"[{position}/{len(source_keys_by_shard)}] Process {shard}: "
            f"{len(source_keys_by_shard[shard])} -> {len(output_keys)} tensors",
            flush=True,
        )
        _process_shard(
            source_path,
            output_path,
            source_keys=source_keys_by_shard[shard],
            output_keys=output_keys,
            source_num_experts=source_num_experts,
            keep_num_experts=keep_num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

    payload_size = sum(
        _safetensors_payload_size(output_dir / shard) for shard in output_keys_by_shard
    )
    output_index = {
        "metadata": {"total_size": payload_size},
        "weight_map": output_weight_map,
    }
    _write_json_atomic(output_dir / _INDEX_NAME, output_index)
    _validate_output(
        output_dir,
        config=output_config,
        index=output_index,
        keep_num_experts=keep_num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    print(
        f"Done: {source_num_experts} -> {keep_num_experts} routed experts; "
        f"num_experts_per_tok remains {num_experts_per_tok}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
