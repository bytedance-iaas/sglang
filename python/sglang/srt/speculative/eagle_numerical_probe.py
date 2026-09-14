"""Exact-request numerical fingerprints for EAGLE and PD handoff localization.

The probes are diagnostic-only and default-off. ``EaglePDHandoffProbe`` records
the physical Prefill-to-Decode token boundary without depending on a draft
worker; ``EagleNumericalProbe`` records the first Decode target-verify and
draft-extend sequence. Those probes synchronize and emit each selected stage
immediately. ``EaglePPSenderProbe`` instead snapshots asynchronously after the
proxy send is enqueued and finalizes at the scheduler's next safe boundary.
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

_BASE_REQUIRED_STAGES = (
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
    "target_verify_pp_input": frozenset({"hidden_states", "residual"}),
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
_PP_RANK_LOCAL_TARGET_TREE_ROW_DOMAIN = "pp_rank_local_target_verify_tree_node"
_REQUEST_ROW_DOMAIN = "request"
_PD_HANDOFF_STAGES = {
    "prefill": ("pp_output", "metadata_write"),
    "decode": ("metadata_read", "prebuilt_bonus"),
}
_PD_HANDOFF_REQUIRED_TENSORS = {
    "pp_output": frozenset({"next_token_ids", "serialized_next_token_ids"}),
    "metadata_write": frozenset({"sampled_token", "wire_output_id"}),
    "metadata_read": frozenset({"wire_output_id", "committed_output_id"}),
    "prebuilt_bonus": frozenset({"committed_output_id", "bonus_tokens"}),
}
_PP_SENDER_STAGE = "target_verify_pp_output"
_PP_SENDER_REQUIRED_TENSORS = frozenset({"hidden_states", "residual"})


class EaglePPSenderProbe:
    """Capture one PP-non-last target output before the async proxy send.

    The sender cannot complete the PP-last EAGLE numerical state machine, so
    this observer owns an independent one-stage record. The worker only attaches
    the observer to an exact-RID result. The scheduler first enqueues the proxy
    send, then starts a non-blocking host snapshot, and finalizes the hash after
    the existing next-slot send commit.
    """

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
                "PP sender probe capture identity requires an exact request id"
            )
        if self.expected_rid is not None and not all(capture_values):
            raise ValueError(
                "PP sender probe requires capture id and Downward API Pod name/UID"
            )
        self.capture = (
            {"id": capture_id, "pod_name": pod_name, "pod_uid": pod_uid}
            if self.expected_rid is not None
            else None
        )
        self._sealed = False
        self._pending: Optional[dict] = None

    @property
    def can_probe(self) -> bool:
        return self.expected_rid is not None and not self._sealed

    def matches_schedule_batch(self, batch) -> bool:
        reqs = getattr(batch, "reqs", ())
        return bool(
            self.can_probe
            and [getattr(req, "rid", None) for req in reqs] == [self.expected_rid]
        )

    def begin_target_verify_pp_output(
        self,
        *,
        pp_proxy_tensors: Optional[dict[str, object]],
        target_world_rank: int,
        require_attn_tp_allgather: bool,
    ) -> bool:
        """Queue a snapshot after the proxy send without waiting for it."""
        if not self.can_probe:
            return False
        self._sealed = True
        proxy = pp_proxy_tensors or {}
        tensors = {
            name: proxy.get(name)
            for name in _PP_SENDER_REQUIRED_TENSORS
            if isinstance(proxy.get(name), torch.Tensor)
        }
        missing = sorted(_PP_SENDER_REQUIRED_TENSORS - tensors.keys())
        if missing:
            self._emit(
                rejection=f"stage decode.{_PP_SENDER_STAGE} missing tensors {missing}"
            )
            return False
        hidden_rows = tensors["hidden_states"].shape[0]
        residual_rows = tensors["residual"].shape[0]
        if hidden_rows != residual_rows:
            self._emit(
                rejection=(
                    f"stage decode.{_PP_SENDER_STAGE} row mismatch "
                    f"hidden_states={hidden_rows}, residual={residual_rows}"
                )
            )
            return False

        try:
            snapshots = {}
            cuda_devices = set()
            for name, tensor in tensors.items():
                value = tensor.detach()
                if value.is_cuda:
                    snapshot = torch.empty(
                        value.shape, dtype=value.dtype, device="cpu", pin_memory=True
                    )
                    snapshot.copy_(value, non_blocking=True)
                    value.record_stream(torch.cuda.current_stream(value.device))
                    cuda_devices.add(value.device)
                else:
                    snapshot = value.clone()
                snapshots[name] = snapshot
            if len(cuda_devices) > 1:
                raise RuntimeError(
                    "PP sender tensors span multiple CUDA devices: "
                    f"{sorted(map(str, cuda_devices))}"
                )
            completion_event = None
            if cuda_devices:
                device = next(iter(cuda_devices))
                completion_event = torch.cuda.Event()
                completion_event.record(torch.cuda.current_stream(device=device))
        except (RuntimeError, TypeError, ValueError) as exc:
            self._emit(
                rejection=(
                    f"failed to queue snapshot for decode.{_PP_SENDER_STAGE}: {exc}"
                )
            )
            raise

        parallel = get_parallel()
        rank = _rank_payload() or {}
        for name in (
            "pp_size",
            "tp_rank",
            "tp_size",
            "attn_tp_rank",
            "attn_tp_size",
            "attn_dp_size",
        ):
            value = getattr(parallel, name, None)
            if value is not None:
                rank[name] = value
        self._pending = {
            "snapshots": snapshots,
            "logical_rows": hidden_rows,
            "completion_event": completion_event,
            "rank": rank,
            "transport": {
                "target_world_rank": target_world_rank,
                "mode": (
                    "send_slice_recv_allgather"
                    if require_attn_tp_allgather
                    else "direct_rank_local"
                ),
                "require_attn_tp_allgather": require_attn_tp_allgather,
            },
        }
        return True

    def finalize_target_verify_pp_output(self) -> None:
        """Finalize only after the existing proxy-send commit."""
        if self._pending is None:
            return
        pending, self._pending = self._pending, None
        try:
            if pending["completion_event"] is not None:
                pending["completion_event"].synchronize()
            fingerprints = {
                name: _tensor_fingerprint(tensor, pending["logical_rows"])
                for name, tensor in sorted(pending["snapshots"].items())
            }
            stage = {
                "row_domain": _PP_RANK_LOCAL_TARGET_TREE_ROW_DOMAIN,
                "logical_rows": pending["logical_rows"],
                "tensors": fingerprints,
            }
        except (RuntimeError, TypeError, ValueError) as exc:
            self._emit(
                rejection=(
                    f"failed to finalize snapshot for decode.{_PP_SENDER_STAGE}: {exc}"
                ),
                rank=pending["rank"],
                transport=pending["transport"],
            )
            raise
        self._emit(
            stage=stage,
            rank=pending["rank"],
            transport=pending["transport"],
        )

    def _emit(
        self,
        *,
        stage: Optional[dict] = None,
        rejection: Optional[str] = None,
        rank: Optional[dict] = None,
        transport: Optional[dict] = None,
    ) -> None:
        payload = {
            "rid": self.expected_rid,
            "capture": self.capture,
            "phase": "decode",
            "role": "decode",
            "status": "rejected" if rejection else "complete",
            "rejection": rejection,
            "stages": ({_PP_SENDER_STAGE: stage} if stage is not None else {}),
        }
        rank = _rank_payload() if rank is None else rank
        if rank is not None:
            payload["rank"] = rank
        if transport is not None:
            payload["transport"] = transport
        logger.warning(
            "EAGLE_PP_SENDER_PROBE_RESULT %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )


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


class EaglePDHandoffProbe:
    """Capture the real two-process PD token handoff for one request."""

    def __init__(
        self,
        expected_rid: Optional[str],
        *,
        role: str,
        capture_id: Optional[str] = None,
        pod_name: Optional[str] = None,
        pod_uid: Optional[str] = None,
    ) -> None:
        if role not in _PD_HANDOFF_STAGES:
            raise ValueError(f"unsupported PD handoff role {role!r}")
        self.expected_rid = expected_rid or None
        self.role = role
        capture_values = (capture_id, pod_name, pod_uid)
        if self.expected_rid is None and any(capture_values):
            raise ValueError(
                "PD handoff probe capture identity requires an exact request id"
            )
        if self.expected_rid is not None and not all(capture_values):
            raise ValueError(
                "PD handoff probe requires capture id and Downward API Pod name/UID"
            )
        self.capture = (
            {"id": capture_id, "pod_name": pod_name, "pod_uid": pod_uid}
            if self.expected_rid is not None
            else None
        )
        self._records: dict[str, dict] = {}
        self._rejection: Optional[str] = None
        self._sealed = False
        self._seen = False

    @property
    def can_probe(self) -> bool:
        return self.expected_rid is not None and not self._sealed

    def _matches(self, rids: Optional[list[str]]) -> bool:
        return bool(self.can_probe and rids == [self.expected_rid])

    def _emit(
        self,
        marker: str,
        stage: str,
        logical_rows: int,
        fingerprints: dict[str, dict],
        error: Optional[BaseException] = None,
    ) -> None:
        payload = {
            "rid": self.expected_rid,
            "capture": self.capture,
            "phase": "pd_handoff",
            "role": self.role,
            "stage": stage,
            # Successful stages are inserted before emission, while a stage
            # error is emitted before insertion. Keep both paths one-based.
            "ordinal": len(self._records) + (stage not in self._records),
            "row_domain": _REQUEST_ROW_DOMAIN,
            "logical_rows": logical_rows,
            "fingerprints": fingerprints,
            "rank": _rank_payload(),
        }
        if error is not None:
            payload.update({"error_type": type(error).__name__, "error": str(error)})
        logger.warning(
            "%s %s", marker, json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )

    def _reject(self, reason: str) -> None:
        if self._rejection is None:
            self._rejection = reason
            logger.error(
                "EAGLE PD handoff diagnostic failed closed for rid=%s role=%s: %s",
                self.expected_rid,
                self.role,
                reason,
            )

    def _record(
        self,
        stage: str,
        tensors: dict[str, Optional[torch.Tensor]],
        *,
        logical_rows: int,
    ) -> None:
        if not self.can_probe:
            return
        self._seen = True
        expected_stages = _PD_HANDOFF_STAGES[self.role]
        if self._rejection is not None:
            if stage == expected_stages[-1]:
                self._seal()
            return
        expected_stage = expected_stages[len(self._records)]
        if stage != expected_stage:
            self._reject(f"out-of-order stage {stage}, expected {expected_stage}")
            self._seal()
            return
        required = _PD_HANDOFF_REQUIRED_TENSORS[stage]
        missing = sorted(name for name in required if tensors.get(name) is None)
        if logical_rows <= 0 or missing:
            self._reject(
                f"stage {stage} invalid logical_rows={logical_rows} missing={missing}"
            )
            self._seal()
            return
        try:
            _synchronize_cuda_tensors(tensors)
            fingerprints = {
                name: _tensor_fingerprint(tensor, logical_rows)
                for name, tensor in sorted(tensors.items())
                if tensor is not None
            }
        except (RuntimeError, TypeError, ValueError) as exc:
            self._emit(
                "EAGLE_PD_HANDOFF_PROBE_STAGE_ERROR",
                stage,
                logical_rows,
                {},
                error=exc,
            )
            self._reject(f"failed to fingerprint {stage}: {exc}")
            self._seal()
            if isinstance(exc, RuntimeError) and any(
                tensor is not None and tensor.is_cuda for tensor in tensors.values()
            ):
                raise
            return
        self._records[stage] = {
            "row_domain": _REQUEST_ROW_DOMAIN,
            "logical_rows": logical_rows,
            "tensors": fingerprints,
        }
        self._emit("EAGLE_PD_HANDOFF_PROBE_STAGE", stage, logical_rows, fingerprints)
        if stage == expected_stages[-1]:
            self._seal()

    def _seal(self) -> None:
        if self._sealed:
            return
        self._sealed = True
        expected_stages = _PD_HANDOFF_STAGES[self.role]
        missing = [stage for stage in expected_stages if stage not in self._records]
        if self._rejection is None and missing:
            self._rejection = f"missing stages {missing}"
        payload = {
            "rid": self.expected_rid,
            "capture": self.capture,
            "phase": "pd_handoff",
            "role": self.role,
            "status": "rejected" if self._rejection else "complete",
            "rejection": self._rejection,
            "seen": self._seen,
            "stages": self._records,
            "rank": _rank_payload(),
        }
        logger.warning(
            "EAGLE_PD_HANDOFF_PROBE_RESULT %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def record_prefill_pp_output(
        self,
        *,
        rids: Optional[list[str]],
        next_token_ids: torch.Tensor,
        serialized_next_token_ids: torch.Tensor,
    ) -> None:
        if self.role == "prefill" and self._matches(rids):
            self._record(
                "pp_output",
                {
                    "next_token_ids": next_token_ids,
                    "serialized_next_token_ids": serialized_next_token_ids,
                },
                logical_rows=len(rids),
            )

    def record_prefill_metadata_write(
        self, *, rid: str, sampled_token: int, wire_output_id: torch.Tensor
    ) -> None:
        if self.role == "prefill" and self._matches([rid]):
            sampled_token_tensor = torch.as_tensor(
                [sampled_token],
                dtype=wire_output_id.dtype,
                device=wire_output_id.device,
            )
            self._record(
                "metadata_write",
                {
                    "sampled_token": sampled_token_tensor,
                    "wire_output_id": wire_output_id,
                },
                logical_rows=1,
            )

    def record_decode_metadata_read(
        self, *, rid: str, wire_output_id: torch.Tensor, committed_output_id: int
    ) -> None:
        if self.role == "decode" and self._matches([rid]):
            committed_output_id_tensor = torch.as_tensor(
                [committed_output_id],
                dtype=wire_output_id.dtype,
                device=wire_output_id.device,
            )
            self._record(
                "metadata_read",
                {
                    "wire_output_id": wire_output_id,
                    "committed_output_id": committed_output_id_tensor,
                },
                logical_rows=1,
            )

    def record_decode_prebuilt_bonus(
        self,
        *,
        rids: Optional[list[str]],
        committed_output_id: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> None:
        if self.role == "decode" and self._matches(rids):
            self._record(
                "prebuilt_bonus",
                {
                    "committed_output_id": committed_output_id,
                    "bonus_tokens": bonus_tokens,
                },
                logical_rows=len(rids),
            )


class EagleNumericalProbe:
    """Capture one Decode numerical fingerprint."""

    def __init__(
        self,
        expected_rid: Optional[str],
        *,
        capture_id: Optional[str] = None,
        pod_name: Optional[str] = None,
        pod_uid: Optional[str] = None,
        require_target_verify_pp_input: bool = False,
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
        self._required_stages = list(_BASE_REQUIRED_STAGES)
        if require_target_verify_pp_input:
            self._required_stages.insert(1, "target_verify_pp_input")
        self._required_stage_set = frozenset(self._required_stages)

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
        expected_stage = self._required_stages[len(bucket)]
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

    def record_target_verify_pp_input(
        self,
        *,
        pp_proxy_tensors: Optional[dict[str, object]],
    ) -> None:
        """Fingerprint the PP-last target input before graph-buffer staging.

        This runs outside CUDA Graph replay.  It observes the real PP proxy
        tensors before ``DecodeCudaGraphRunner.load_batch`` copies them into
        pointer-stable graph buffers, so it does not add request-conditioned
        Python work to the captured graph or alter graph admission.
        """
        proxy = pp_proxy_tensors or {}
        # The transport dictionary also carries non-tensor metadata such as
        # ``__msg_type__``.  Keep that protocol envelope out of the numerical
        # observer and admit only model inputs understood by this stage.
        tensors = {
            name: proxy.get(name)
            for name in ("hidden_states", "residual")
            if isinstance(proxy.get(name), torch.Tensor)
        }
        missing = sorted(
            name
            for name in _REQUIRED_TENSORS["target_verify_pp_input"]
            if name not in tensors
        )
        if missing:
            self._reject(
                f"stage decode.target_verify_pp_input missing tensors {missing}"
            )
            return
        hidden_rows = tensors["hidden_states"].shape[0]
        residual_rows = tensors["residual"].shape[0]
        if hidden_rows != residual_rows:
            self._reject(
                "stage decode.target_verify_pp_input row mismatch "
                f"hidden_states={hidden_rows}, residual={residual_rows}"
            )
            return
        self._record_explicit(
            "target_verify_pp_input",
            tensors,
            logical_rows=hidden_rows,
            row_domain=_PP_RANK_LOCAL_TARGET_TREE_ROW_DOMAIN,
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
                missing = self._required_stage_set - self._records.get(phase, {}).keys()
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
