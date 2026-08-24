"""Capture-safe, opt-in timing instrumentation for the SiDP CUDA Graph.

All timing events are inserted while the model graph is captured, so the same
markers execute on every decode replay.  A sampled replay is synchronized only
after launch and emitted as one JSONL record.  This is intentionally a
diagnostic path: the event nodes and sampled synchronization must not be used
for final throughput numbers.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

import torch

from sglang.srt.layers.sidp.scheduler import owner_of


def _timing_event() -> torch.cuda.Event:
    # external=True materializes record/wait nodes in a captured graph. Without
    # it, PyTorch may internalize the dependency and the event has no replay
    # timestamp available to elapsed_time().
    return torch.cuda.Event(enable_timing=True, external=True)


class SidpGraphProfiler:
    """Own the timing events shared by all captured decode graph shapes."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        dp_rank: int,
        dp_size: int,
        num_cycles: int,
        cycle_layers: dict[int, list[int]],
        sample_interval: int,
        warmup_replays: int,
        output_dir: str,
        peak_shifting: bool,
    ) -> None:
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.num_cycles = num_cycles
        self.cycle_layers = {
            cycle: list(layers) for cycle, layers in cycle_layers.items()
        }
        self.sample_interval = sample_interval
        self.warmup_replays = warmup_replays
        self.order = "peak-shifting" if peak_shifting else "compute"
        self.replay_index = 0

        self.forward_start = _timing_event()
        self.forward_compute_end = _timing_event()
        self.forward_end = _timing_event()
        self.anchor = _timing_event()
        self.compute_start = {cycle: _timing_event() for cycle in range(num_cycles)}
        self.compute_end = {cycle: _timing_event() for cycle in range(num_cycles)}
        self.comm_start = {cycle: _timing_event() for cycle in range(num_cycles)}
        self.comm_end = {cycle: _timing_event() for cycle in range(num_cycles)}

        non_local_layers = [
            layer_id
            for cycle in range(num_cycles)
            for layer_id in self.cycle_layers.get(cycle, [])
        ]
        self.copy_start = {layer_id: _timing_event() for layer_id in non_local_layers}
        self.copy_end = {layer_id: _timing_event() for layer_id in non_local_layers}
        # Cycle 0 is resident at graph entry and therefore has no in-forward RAW
        # wait.  Its copy events describe the tail refill for the next forward.
        waited_layers = [
            layer_id for layer_id in non_local_layers if layer_id // dp_size > 0
        ]
        self.wait_start = {layer_id: _timing_event() for layer_id in waited_layers}
        self.wait_end = {layer_id: _timing_event() for layer_id in waited_layers}
        self.copy_nbytes: dict[int, int] = {}

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        self.path = output_path / (
            f"sidp_graph_profile_{self.order}_rank{dp_rank}_pid{os.getpid()}.jsonl"
        )
        self._output = self.path.open("a", encoding="utf-8")
        self._write(
            {
                "record_type": "metadata",
                "schema_version": self.SCHEMA_VERSION,
                "rank": dp_rank,
                "dp_size": dp_size,
                "order": self.order,
                "pid": os.getpid(),
                "num_cycles": num_cycles,
                "sample_interval": sample_interval,
                "warmup_replays": warmup_replays,
                "cycle_layers": self.cycle_layers,
            }
        )

    def _write(self, record: dict) -> None:
        self._output.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._output.flush()

    def record_forward_start(self, stream: torch.cuda.Stream) -> None:
        self.forward_start.record(stream)

    def record_forward_compute_end(self, stream: torch.cuda.Stream) -> None:
        self.forward_compute_end.record(stream)

    def record_forward_end(self, stream: torch.cuda.Stream) -> None:
        self.forward_end.record(stream)

    def record_cycle_compute_start(self, cycle: int, stream: torch.cuda.Stream) -> None:
        self.compute_start[cycle].record(stream)

    def record_cycle_compute_end(self, cycle: int, stream: torch.cuda.Stream) -> None:
        self.compute_end[cycle].record(stream)

    def record_cycle_comm_start(self, cycle: int, stream: torch.cuda.Stream) -> None:
        self.comm_start[cycle].record(stream)

    def record_cycle_comm_end(self, cycle: int, stream: torch.cuda.Stream) -> None:
        self.comm_end[cycle].record(stream)

    def record_copy_start(
        self, layer_id: int, nbytes: int, stream: torch.cuda.Stream
    ) -> None:
        self.copy_nbytes[layer_id] = nbytes
        self.copy_start[layer_id].record(stream)

    def record_copy_end(self, layer_id: int, stream: torch.cuda.Stream) -> None:
        self.copy_end[layer_id].record(stream)

    def record_wait_start(self, layer_id: int, stream: torch.cuda.Stream) -> None:
        self.wait_start[layer_id].record(stream)

    def record_wait_end(self, layer_id: int, stream: torch.cuda.Stream) -> None:
        self.wait_end[layer_id].record(stream)

    @staticmethod
    def _round_ms(value: float) -> float:
        return round(value, 6)

    def _offset_ms(self, event: torch.cuda.Event) -> float:
        return self._round_ms(self.forward_start.elapsed_time(event))

    def _duration_ms(self, start: torch.cuda.Event, end: torch.cuda.Event) -> float:
        return self._round_ms(start.elapsed_time(end))

    def _layer_records(self, layer_ids: Iterable[int]) -> list[dict]:
        records = []
        for schedule_index, layer_id in enumerate(layer_ids):
            cycle = layer_id // self.dp_size
            records.append(
                {
                    "layer": layer_id,
                    "cycle": cycle,
                    "owner": owner_of(layer_id, self.dp_size),
                    "schedule_index": schedule_index,
                    "nbytes": self.copy_nbytes[layer_id],
                    "start_ms": self._offset_ms(self.copy_start[layer_id]),
                    "end_ms": self._offset_ms(self.copy_end[layer_id]),
                    "duration_ms": self._duration_ms(
                        self.copy_start[layer_id], self.copy_end[layer_id]
                    ),
                }
            )
        return records

    def collect_after_graph_replay(
        self, *, raw_batch_size: int, graph_batch_size: int
    ) -> None:
        """Synchronize and emit one sampled replay.

        ``anchor_host_ns`` is a same-host monotonic clock approximation of the
        anchor event completion.  It permits cross-rank alignment at millisecond
        scale, while Nsight/Kineto remains the source of truth for sub-millisecond
        absolute alignment.
        """
        self.replay_index += 1
        if self.replay_index <= self.warmup_replays:
            return
        if (self.replay_index - self.warmup_replays - 1) % self.sample_interval:
            return

        self.anchor.record(torch.cuda.current_stream())
        sync_started_ns = time.monotonic_ns()
        self.anchor.synchronize()
        anchor_host_ns = time.monotonic_ns()

        anchor_offset_ms = self._offset_ms(self.anchor)
        cycles = []
        copies = []
        waits = []
        for cycle in range(self.num_cycles):
            compute_start_ms = self._offset_ms(self.compute_start[cycle])
            compute_end_ms = self._offset_ms(self.compute_end[cycle])
            comm_start_ms = self._offset_ms(self.comm_start[cycle])
            comm_end_ms = self._offset_ms(self.comm_end[cycle])
            cycles.append(
                {
                    "cycle": cycle,
                    "comm_target": "next_forward_cycle0" if cycle == 0 else cycle,
                    "compute_start_ms": compute_start_ms,
                    "compute_end_ms": compute_end_ms,
                    "compute_duration_ms": self._duration_ms(
                        self.compute_start[cycle], self.compute_end[cycle]
                    ),
                    "comm_start_ms": comm_start_ms,
                    "comm_end_ms": comm_end_ms,
                    "comm_duration_ms": self._duration_ms(
                        self.comm_start[cycle], self.comm_end[cycle]
                    ),
                }
            )
            cycle_copies = self._layer_records(self.cycle_layers.get(cycle, []))
            if cycle == 0:
                for copy in cycle_copies:
                    copy["target_forward_offset"] = 1
            else:
                for copy in cycle_copies:
                    copy["target_forward_offset"] = 0
            copies.extend(cycle_copies)

        for layer_id in sorted(self.wait_start):
            copy_end_ms = self._offset_ms(self.copy_end[layer_id])
            wait_start_ms = self._offset_ms(self.wait_start[layer_id])
            waits.append(
                {
                    "layer": layer_id,
                    "cycle": layer_id // self.dp_size,
                    "copy_end_ms": copy_end_ms,
                    "demand_ms": wait_start_ms,
                    "ready_margin_ms": self._round_ms(wait_start_ms - copy_end_ms),
                    "wait_start_ms": wait_start_ms,
                    "wait_end_ms": self._offset_ms(self.wait_end[layer_id]),
                    "exposed_wait_ms": self._duration_ms(
                        self.wait_start[layer_id], self.wait_end[layer_id]
                    ),
                }
            )

        compute_end_ms = self._offset_ms(self.forward_compute_end)
        forward_end_ms = self._offset_ms(self.forward_end)
        record = {
            "record_type": "sample",
            "schema_version": self.SCHEMA_VERSION,
            "rank": self.dp_rank,
            "order": self.order,
            "pid": os.getpid(),
            "replay_index": self.replay_index,
            "sample_interval": self.sample_interval,
            "raw_batch_size": raw_batch_size,
            "graph_batch_size": graph_batch_size,
            "anchor_host_ns": anchor_host_ns,
            "anchor_offset_ms": anchor_offset_ms,
            "collection_sync_wait_ms": self._round_ms(
                (anchor_host_ns - sync_started_ns) / 1_000_000
            ),
            "forward_compute_end_ms": compute_end_ms,
            "forward_end_ms": forward_end_ms,
            "forward_duration_ms": self._duration_ms(
                self.forward_start, self.forward_end
            ),
            "tail_join_ms": self._round_ms(forward_end_ms - compute_end_ms),
            "cycles": cycles,
            "copies": copies,
            "waits": waits,
        }
        self._write(record)
