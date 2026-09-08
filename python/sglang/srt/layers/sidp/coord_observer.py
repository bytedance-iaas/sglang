"""Schedule-consistency observer for SiDP Direction A (read-only diagnostics).

Purpose: the coordinated device barrier only stays efficient while all members
make the SAME prefill/decode decision each scheduling round. Once one member
diverges (e.g. it finished a few requests -> running freed -> it inserts a
prefill while the others decode), the barrier idles. Before designing a fix we
need to MEASURE when and why divergence happens.

This observer records, once per rendezvous round on each member, that member's
scheduling decision plus the state that explains it, into a per-rank jsonl file.
An offline tool (verify_sidp_sched_consistency.py) aligns the rows by the global
round id and reports per-round agreement / divergence and its likely cause.

Design constraints:
  * READ-ONLY: it never touches scheduling or barrier behavior. It only reads
    already-computed state at the rendezvous point and appends a log row.
  * Cheap + non-perturbing: rows are buffered in memory and flushed in batches
    (and at close), so the hot loop does a dict build + list append, no I/O.
  * Global alignment: rows are keyed by the rendezvous round id, which every
    member increments in lockstep (unconditional all-vote), so round R on member
    A is the same scheduling wave as round R on member B.

The "eos" field is necessarily the PREVIOUS round's completions: a request's EOS
is detected in process_batch_result (AFTER this round's rendezvous), so at
rendezvous time we can only see EOS that the last forward produced -- which is
exactly the factor that explains why THIS round might want to prefill.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from typing import Any, Dict, List, Optional


class CoordScheduleObserver:
    """Per-rank buffered jsonl recorder of per-round scheduling decisions."""

    def __init__(
        self,
        *,
        member_rank: int,
        world_size: int,
        out_dir: str,
        flush_every: int = 64,
    ) -> None:
        self.member_rank = member_rank
        self.world_size = world_size
        self.flush_every = max(flush_every, 1)
        os.makedirs(out_dir, exist_ok=True)
        # One file per rank; the offline tool globs the directory and joins by
        # round id. Truncate on start so a re-run does not append to stale data.
        self.path = os.path.join(out_dir, f"sched_trace_rank{member_rank}.jsonl")
        self._buf: List[str] = []
        self._closed = False
        self._fh = open(self.path, "w")
        # Flush the tail on interpreter exit / graceful shutdown so we don't lose
        # the last buffered rows when the server is SIGTERM'd after a run.
        atexit.register(self.close)
        # Header row records the run's identity so the offline tool can sanity
        # check that all rank files belong to the same run.
        self._write_row({
            "kind": "header",
            "member_rank": member_rank,
            "world_size": world_size,
            "start_ts": time.time(),
        })

    def record(
        self,
        *,
        round_id: int,
        mode: str,
        running: int,
        waiting: int,
        batch_size: int,
        kv_used_frac: float,
        retracted: int,
        eos_prev: int,
        participate: bool,
        live_nptr: int,
    ) -> None:
        """Append one round's decision. Called once per rendezvous on this rank."""
        self._write_row({
            "kind": "round",
            "round": round_id,
            "rank": self.member_rank,
            "mode": mode,               # "prefill" | "decode" | "idle"
            "running": running,
            "waiting": waiting,
            "batch_size": batch_size,
            "kv_used_frac": round(kv_used_frac, 4),
            "retracted": retracted,     # this round (seen before rendezvous)
            "eos_prev": eos_prev,       # completions from the PREVIOUS forward
            "participate": participate, # did this member run the barrier this round
            "live_nptr": live_nptr,     # cohort-wide barrier participant count
            "ts": time.time(),
        })

    def _write_row(self, row: Dict[str, Any]) -> None:
        self._buf.append(json.dumps(row, ensure_ascii=False))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        self._fh.write("\n".join(self._buf) + "\n")
        self._fh.flush()
        self._buf.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            try:
                self._fh.close()
            except Exception:
                pass
