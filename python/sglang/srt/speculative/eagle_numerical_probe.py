"""Exact-request numerical fingerprints for EAGLE verify/NextN localization.

The probe is diagnostic-only and default-off.  It records the first matching
decode target-verify and draft-extend sequence.  Every selected stage
synchronizes its current CUDA stream and emits a fingerprint immediately, so a
later device fault does not erase the completed stage prefix.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from typing import Optional

import torch

from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_REQUIRED_STAGES = (
    "target_verify_input",
    "target_verify_output",
    "target_verify_sample",
    "target_verify_accept",
    "target_verify_handoff",
    "draft_extend_input",
    "nextn_embed",
    "nextn_decoder",
    "nextn_norm",
    "nextn_logits",
    "proposed_token",
)
_REQUIRED_STAGE_SET = frozenset(_REQUIRED_STAGES)
_REQUIRED_TENSORS = {
    "target_verify_input": frozenset(
        {
            "draft_token",
            "positions",
            "retrieve_index",
            "retrieve_next_token",
            "retrieve_next_sibling",
        }
    ),
    "target_verify_output": frozenset({"logits", "hidden_states"}),
    "target_verify_sample": frozenset({"predict"}),
    "target_verify_accept": frozenset({"accept_lens", "accept_index"}),
    "target_verify_handoff": frozenset({"predict", "hidden_states"}),
    "draft_extend_input": frozenset({"input_ids", "positions", "target_hidden_states"}),
    "nextn_embed": frozenset({"hidden_states"}),
    "nextn_decoder": frozenset({"hidden_states"}),
    "nextn_norm": frozenset({"hidden_states"}),
    "nextn_logits": frozenset({"logits"}),
    "proposed_token": frozenset({"topk_index", "topk_probability"}),
}
_DENSE_ROW_DOMAIN = "dense_request_major_prefix"
_PROPOSAL_ROW_DOMAIN = "request_terminal"
_TARGET_TREE_ROW_DOMAIN = "target_verify_tree_node"
_REQUEST_ROW_DOMAIN = "request"


def _rank_payload() -> Optional[dict[str, int]]:
    try:
        parallel = get_parallel()
        return {
            "world": parallel.world_rank,
            "pp": parallel.pp_rank,
            "attn_dp": parallel.attn_dp_rank,
        }
    except (AssertionError, AttributeError, RuntimeError):
        return None


def _synchronize_cuda_tensors(
    tensors: dict[str, Optional[torch.Tensor]],
) -> None:
    """Expose asynchronous device faults at the owning stage boundary."""

    devices = {
        tensor.device
        for tensor in tensors.values()
        if tensor is not None and tensor.is_cuda
    }
    for device in sorted(devices, key=str):
        torch.cuda.current_stream(device=device).synchronize()


def _tensor_fingerprint(tensor: torch.Tensor, logical_rows: int) -> dict:
    value = tensor.detach()
    if logical_rows <= 0:
        raise ValueError(f"logical_rows must be positive, got {logical_rows}")
    if value.ndim == 0:
        raise ValueError("row-domain tensor must have at least one dimension")
    if value.shape[0] < logical_rows:
        raise ValueError(
            f"tensor has {value.shape[0]} rows, expected at least {logical_rows}"
        )
    value = value[:logical_rows]
    cpu = value.contiguous().cpu()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    result = {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if cpu.numel() == 0:
        result.update({"finite": True, "sum": 0.0, "abs_max": 0.0})
        return result

    numeric = cpu.to(torch.float64)
    finite = torch.isfinite(numeric)
    result["finite"] = bool(finite.all())
    if bool(finite.any()):
        finite_values = numeric[finite]
        result["sum"] = float(finite_values.sum())
        result["abs_max"] = float(finite_values.abs().max())
    else:
        result["sum"] = None
        result["abs_max"] = None
    if cpu.numel() <= 16:
        result["values"] = cpu.reshape(-1).tolist()
    return result


class EagleNumericalProbe:
    """Capture one complete target-verify through draft-extend fingerprint."""

    def __init__(
        self,
        expected_rid: Optional[str],
        *,
        capture_id: Optional[str] = None,
        pod_name: Optional[str] = None,
        pod_uid: Optional[str] = None,
    ) -> None:
        self.expected_rid = expected_rid or None
        capture_values = (capture_id, pod_name, pod_uid)
        if self.expected_rid is None and any(capture_values):
            raise ValueError(
                "numerical probe capture identity requires an exact request id"
            )
        if self.expected_rid is not None and not all(capture_values):
            raise ValueError(
                "numerical probe requires capture id and Downward API Pod name/UID"
            )
        self.capture = (
            {"id": capture_id, "pod_name": pod_name, "pod_uid": pod_uid}
            if self.expected_rid is not None
            else None
        )
        self._rejection: Optional[str] = None
        self._sealed = False
        self._seen = False
        self._active_phase: Optional[str] = None
        self._active_rows = 0
        self._records: dict[str, dict[str, dict]] = {}
        self._proposal_rows: dict[str, int] = {}
        self._next_ordinal = 1

    @property
    def can_probe(self) -> bool:
        return (
            self.expected_rid is not None
            and self._rejection is None
            and not self._sealed
        )

    def matches_rids(self, rids: Optional[list[str]]) -> bool:
        # This diagnostic must not perturb unrelated requests.  A co-batched
        # occurrence is deliberately deferred instead of forcing the whole DP
        # batch eager and then permanently rejecting the one-shot capture.
        return bool(self.can_probe and rids == [self.expected_rid])

    def matches_schedule_batch(self, batch) -> bool:
        return self.matches_rids([req.rid for req in batch.reqs])

    def needs_eager_for_schedule_batch(self, batch) -> bool:
        return self.matches_schedule_batch(batch) and "draft_extend_input" not in (
            self._records.get("decode", {})
        )

    def _reject(self, reason: str) -> None:
        if self._rejection is None:
            self._rejection = reason
            logger.error(
                "EAGLE numerical diagnostic failed closed for rid=%s: %s",
                self.expected_rid,
                reason,
            )

    def _emit_stage_event(
        self,
        *,
        prefix: str,
        phase: str,
        stage: str,
        row_domain: str,
        fingerprints: dict[str, dict],
        error: Optional[BaseException] = None,
    ) -> None:
        payload = {
            "rid": self.expected_rid,
            "capture": self.capture,
            "phase": phase,
            "stage": stage,
            "ordinal": self._next_ordinal,
            "row_domain": row_domain,
            "logical_rows": self._active_rows,
            "fingerprints": fingerprints,
            "rank": _rank_payload(),
        }
        if error is not None:
            payload.update(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        logger.warning(
            "%s %s",
            prefix,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def _handle_stage_error(
        self,
        *,
        phase: str,
        stage: str,
        row_domain: str,
        error: BaseException,
    ) -> None:
        self._emit_stage_event(
            prefix="EAGLE_NUMERICAL_PROBE_STAGE_ERROR",
            phase=phase,
            stage=stage,
            row_domain=row_domain,
            fingerprints={},
            error=error,
        )
        self._reject(f"failed to synchronize/fingerprint {phase}.{stage}: {error}")

    def _record(
        self,
        stage: str,
        tensors: dict[str, Optional[torch.Tensor]],
        *,
        row_domain: str = _DENSE_ROW_DOMAIN,
    ) -> None:
        phase = self._active_phase
        if not self.can_probe or phase is None:
            return
        bucket = self._records.setdefault(phase, {})
        if stage in bucket:
            self._reject(f"duplicate stage {phase}.{stage}")
            return
        required = _REQUIRED_TENSORS.get(stage)
        if required is None:
            self._reject(f"unexpected stage {phase}.{stage}")
            return
        expected_stage = _REQUIRED_STAGES[len(bucket)]
        if stage != expected_stage:
            self._reject(
                f"out-of-order stage {phase}.{stage}, expected {expected_stage}"
            )
            return
        missing = sorted(name for name in required if tensors.get(name) is None)
        if missing:
            self._reject(f"stage {phase}.{stage} missing tensors {missing}")
            return
        try:
            _synchronize_cuda_tensors(tensors)
        except RuntimeError as exc:
            self._handle_stage_error(
                phase=phase, stage=stage, row_domain=row_domain, error=exc
            )
            raise

        try:
            fingerprints = {
                name: _tensor_fingerprint(tensor, self._active_rows)
                for name, tensor in sorted(tensors.items())
                if tensor is not None
            }
            bucket[stage] = {
                "row_domain": row_domain,
                "logical_rows": self._active_rows,
                "tensors": fingerprints,
            }
            self._emit_stage_event(
                prefix="EAGLE_NUMERICAL_PROBE_STAGE",
                phase=phase,
                stage=stage,
                row_domain=row_domain,
                fingerprints=fingerprints,
            )
            self._next_ordinal += 1
        except (RuntimeError, TypeError, ValueError) as exc:
            self._handle_stage_error(
                phase=phase, stage=stage, row_domain=row_domain, error=exc
            )
            if isinstance(exc, RuntimeError) and any(
                tensor is not None and tensor.is_cuda for tensor in tensors.values()
            ):
                raise

    def _record_explicit(
        self,
        stage: str,
        tensors: dict[str, Optional[torch.Tensor]],
        *,
        logical_rows: int,
        row_domain: str,
    ) -> None:
        previous_phase = self._active_phase
        previous_rows = self._active_rows
        self._active_phase = "decode"
        self._active_rows = logical_rows
        try:
            self._record(stage, tensors, row_domain=row_domain)
        finally:
            self._active_phase = previous_phase
            self._active_rows = previous_rows

    def record_target_verify_input(
        self,
        *,
        rids: Optional[list[str]],
        draft_token: torch.Tensor,
        positions: torch.Tensor,
        retrieve_index: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        batch_size: int,
        draft_token_num: int,
    ) -> bool:
        """Start the one-shot capture at the target-verify input tree."""
        if not self.matches_rids(rids):
            return False
        self._seen = True
        if "decode" in self._records:
            return False
        logical_rows = batch_size * draft_token_num
        self._proposal_rows["decode"] = batch_size
        self._record_explicit(
            "target_verify_input",
            {
                "draft_token": draft_token,
                "positions": positions,
                "retrieve_index": retrieve_index.reshape(-1),
                "retrieve_next_token": retrieve_next_token.reshape(-1),
                "retrieve_next_sibling": retrieve_next_sibling.reshape(-1),
            },
            logical_rows=logical_rows,
            row_domain=_TARGET_TREE_ROW_DOMAIN,
        )
        return self.can_probe

    def record_target_verify_output(
        self,
        *,
        logits: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        logical_rows: int,
    ) -> None:
        self._record_explicit(
            "target_verify_output",
            {"logits": logits, "hidden_states": hidden_states},
            logical_rows=logical_rows,
            row_domain=_TARGET_TREE_ROW_DOMAIN,
        )

    def record_target_verify_sample(
        self, *, predict: torch.Tensor, logical_rows: int
    ) -> None:
        self._record_explicit(
            "target_verify_sample",
            {"predict": predict},
            logical_rows=logical_rows,
            row_domain=_TARGET_TREE_ROW_DOMAIN,
        )

    def record_target_verify_accept(
        self, *, accept_lens: torch.Tensor, accept_index: torch.Tensor, batch_size: int
    ) -> None:
        self._record_explicit(
            "target_verify_accept",
            {"accept_lens": accept_lens, "accept_index": accept_index},
            logical_rows=batch_size,
            row_domain=_REQUEST_ROW_DOMAIN,
        )

    def record_target_verify_handoff(
        self,
        *,
        predict: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        logical_rows: int,
    ) -> None:
        self._record_explicit(
            "target_verify_handoff",
            {"predict": predict, "hidden_states": hidden_states},
            logical_rows=logical_rows,
            row_domain=_TARGET_TREE_ROW_DOMAIN,
        )

    @contextlib.contextmanager
    def forward_scope(
        self,
        forward_batch,
        *,
        phase: str,
        logical_rows: int,
        using_cuda_graph: bool,
        input_ids: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor],
        positions: torch.Tensor,
    ):
        if not self.matches_rids(forward_batch.rids):
            yield False
            return

        self._seen = True
        if not self.can_probe or "draft_extend_input" in self._records.get(phase, {}):
            yield False
            return
        if phase != "decode":
            self._reject(f"unexpected phase {phase}")
            yield False
            return
        if using_cuda_graph:
            self._reject("CUDA graph capture/replay is unsupported")
            yield False
            return

        previous_callback = forward_batch._eagle_numerical_probe_callback
        previous_phase = forward_batch._eagle_numerical_probe_phase
        self._active_phase = phase
        self._active_rows = logical_rows
        self._proposal_rows[phase] = len(forward_batch.rids)
        forward_batch._eagle_numerical_probe_callback = self._record
        forward_batch._eagle_numerical_probe_phase = phase
        self._record(
            "draft_extend_input",
            {
                "input_ids": input_ids,
                "positions": positions,
                "target_hidden_states": target_hidden_states,
            },
        )
        try:
            yield True
        finally:
            forward_batch._eagle_numerical_probe_callback = previous_callback
            forward_batch._eagle_numerical_probe_phase = previous_phase
            self._active_phase = None
            self._active_rows = 0

    def record_proposal(
        self,
        *,
        phase: str,
        logical_rows: int,
        topk_index: torch.Tensor,
        topk_probability: torch.Tensor,
    ) -> None:
        if (
            not self.can_probe
            or phase not in self._records
            or "proposed_token" in self._records[phase]
        ):
            return
        expected_rows = self._proposal_rows.get(phase)
        if logical_rows != expected_rows:
            self._reject(
                f"phase {phase} proposal has {logical_rows} rows, "
                f"expected {expected_rows}"
            )
            return
        previous_phase = self._active_phase
        previous_rows = self._active_rows
        self._active_phase = phase
        self._active_rows = logical_rows
        try:
            self._record(
                "proposed_token",
                {
                    "topk_index": topk_index,
                    "topk_probability": topk_probability,
                },
                row_domain=_PROPOSAL_ROW_DOMAIN,
            )
        finally:
            self._active_phase = previous_phase
            self._active_rows = previous_rows

    def finish(self, *, rid: str, natural_stop: bool, normal_completion: bool) -> None:
        if rid != self.expected_rid or self._sealed:
            return
        self._sealed = True
        if self._rejection is None and not normal_completion:
            self._rejection = "request did not complete normally"
        if self._rejection is None:
            for phase in ("decode",):
                missing = _REQUIRED_STAGE_SET - self._records.get(phase, {}).keys()
                if missing:
                    self._rejection = f"phase {phase} missing stages {sorted(missing)}"
                    break
        payload = {
            "rid": rid,
            "capture": self.capture,
            "natural_stop": natural_stop,
            "normal_completion": normal_completion,
            "status": "rejected" if self._rejection else "complete",
            "rejection": self._rejection,
            "seen": self._seen,
            "phases": self._records,
        }
        rank = _rank_payload()
        if rank is not None:
            payload["rank"] = rank
        logger.warning(
            "EAGLE_NUMERICAL_PROBE_RESULT %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        self._records.clear()
        self._proposal_rows.clear()


def maybe_record_eagle_numerical_stage(
    forward_batch, stage: str, **tensors: Optional[torch.Tensor]
) -> None:
    callback = getattr(forward_batch, "_eagle_numerical_probe_callback", None)
    if callback is not None:
        callback(stage, tensors)
