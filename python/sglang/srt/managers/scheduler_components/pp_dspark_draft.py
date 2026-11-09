from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.dspark_components.dspark_pp import (
    PPDSparkIdentity,
    draft_owner,
    pack_proposal,
    pack_proxy_proposal,
    pop_proxy_proposal,
    unpack_proposal,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult


@dataclass(frozen=True, slots=True)
class PPDSparkDraftWork:
    batch: ScheduleBatch
    draft_input: object
    pp_outputs: Optional[PPProxyTensors] = None
    generation: Optional[int] = None
    slot_id: Optional[int] = None
    is_idle: bool = False


@dataclass(frozen=True, slots=True)
class PPDSparkProxyProposal:
    generation: int
    slot_id: int
    payload: dict


def snapshot_pp_dspark_batch(batch: ScheduleBatch) -> ScheduleBatch:
    snapshot = copy.copy(batch)
    snapshot.reqs = batch.reqs[:]
    snapshot.req_pool_indices = batch.req_pool_indices.clone()
    return snapshot


class PPDSparkDraftCoordinator:
    """Owns request-scoped PP DSpark draft scheduling state."""

    def __init__(self, scheduler: Scheduler):
        self._scheduler = scheduler
        self._pending: deque[PPDSparkDraftWork] = deque()
        self._send_work = []
        self._proxy_outbox: dict[int, PPDSparkProxyProposal] = {}
        self._pipelined_ready_event: Optional[torch.Event] = None

    def on_batch_launched(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        scheduler = self._scheduler
        if self._is_pipelined_proxy_protocol():
            # PP0 derives the next proposal from the relayed final-stage result.
            # No other PP stage may enter the replicated Draft collectives.
            return
        if result.pp_dspark_draft_idle:
            if scheduler.pp_group.is_last_rank:
                self._run_idle_draft(batch)
            return

        if (
            result.pp_dspark_projected_context is None
            or result.accept_lens is None
            or result.pp_dspark_next_proposal is not None
        ):
            return

        if self._is_bubble_policy() and not self._is_serial_generation_protocol():
            self.enqueue(batch, result.next_draft_input)
        else:
            result.pp_dspark_next_proposal = self._run_draft(
                batch, result.next_draft_input
            )

    def on_relayed_idle(self, batch: ScheduleBatch, pp_outputs: PPProxyTensors) -> None:
        scheduler = self._scheduler
        if (
            "dspark_draft_idle" not in pp_outputs.tensors
            or not scheduler.pp_group.is_first_rank
        ):
            return
        if self._is_pipelined_proxy_protocol():
            self.enqueue(
                batch,
                None,
                pp_outputs,
                generation=scheduler._pp_received_output_generation,
                slot_id=scheduler._pp_received_output_slot,
                is_idle=True,
            )
            return
        if self._is_serial_generation_protocol() and get_parallel().pp_size > 2:
            return

        scheduler.forward_stream.wait_stream(scheduler.copy_stream)
        with scheduler.forward_stream_ctx:
            self._run_idle_draft(batch)
        scheduler.copy_stream.wait_stream(scheduler.forward_stream)

    def on_relayed_proposals(
        self,
        batch: ScheduleBatch,
        draft_input: object,
        pp_outputs: PPProxyTensors,
    ) -> None:
        scheduler = self._scheduler
        if self._is_pipelined_proxy_protocol():
            if not scheduler.pp_group.is_first_rank:
                return
            self.enqueue(
                batch,
                draft_input,
                pp_outputs,
                generation=scheduler._pp_received_output_generation,
                slot_id=scheduler._pp_received_output_slot,
            )
            return
        serial_pp4 = (
            self._is_serial_generation_protocol() and get_parallel().pp_size > 2
        )
        local_owner = get_parallel().pp_rank if serial_pp4 else 1
        deferred = (
            f"dspark_next_{local_owner}_identities" not in pp_outputs.tensors
            and self._is_bubble_policy()
        )
        if deferred:
            if (not serial_pp4) and not scheduler.pp_group.is_first_rank:
                raise RuntimeError(
                    "Only the first PP rank may receive a deferred DSpark "
                    "verify result."
                )
            self.enqueue(
                batch,
                draft_input,
                pp_outputs,
                generation=getattr(scheduler, "_pp_received_output_generation", None),
            )
            return

        if scheduler.pp_group.is_first_rank:
            scheduler.forward_stream.wait_stream(scheduler.copy_stream)
            with scheduler.forward_stream_ctx:
                pp_outputs.tensors.update(
                    pack_proposal(0, self._run_draft(batch, draft_input))
                )
            scheduler.copy_stream.wait_stream(scheduler.forward_stream)

        for owner in range(get_parallel().pp_size):
            scheduler.model_worker.install_pp_draft(
                batch,
                owner,
                unpack_proposal(owner, pp_outputs.tensors),
            )

    def enqueue(
        self,
        batch: ScheduleBatch,
        draft_input: object,
        pp_outputs: Optional[PPProxyTensors] = None,
        *,
        generation: Optional[int] = None,
        slot_id: Optional[int] = None,
        is_idle: bool = False,
    ) -> None:
        self._pending.append(
            PPDSparkDraftWork(
                batch=snapshot_pp_dspark_batch(batch),
                draft_input=draft_input,
                pp_outputs=pp_outputs,
                generation=generation,
                slot_id=slot_id,
                is_idle=is_idle,
            )
        )

    def has_pending(self) -> bool:
        return bool(self._pending)

    @staticmethod
    def _identity_key(batch: ScheduleBatch, *, next_round: bool = False) -> tuple:
        identities = [PPDSparkIdentity.from_req(req) for req in batch.reqs]
        if next_round:
            identities = [identity.next_round() for identity in identities]
        return tuple(identity.to_wire() for identity in identities)

    def prepare_proxy_proposal(
        self,
        batch: ScheduleBatch,
        *,
        slot_id: int,
        proxy_tensors: Optional[PPProxyTensors],
    ) -> dict:
        """Install the current content proposal and return its relay envelope."""
        scheduler = self._scheduler
        expected = self._identity_key(batch)
        expected_set = set(expected)
        if scheduler.pp_group.is_first_rank:
            proposal = self._proxy_outbox.pop(slot_id, None)
            if proposal is None:
                if any(identity[2] > 0 for identity in expected):
                    raise RuntimeError(
                        "Missing steady-state PP DSpark proposal for "
                        f"slot={slot_id}, identities={expected}."
                    )
                return {}
            if proposal.slot_id != slot_id:
                raise RuntimeError(
                    "PP DSpark proposal changed scheduler slots: "
                    f"source={proposal.slot_id}, target={slot_id}."
                )
            payload = self._filter_proxy_payload(proposal.payload, expected_set)
            received = tuple(
                (identity[0], tuple(identity[1]), identity[2])
                for identity in payload["identities"]
            )
            self._validate_proxy_identity_coverage(
                expected, expected_set, received, slot_id
            )
            if not received:
                return {}
            scheduler.model_worker.install_pp_draft_subset(batch, payload)
            return pack_proxy_proposal(proposal.generation, proposal.slot_id, payload)

        if proxy_tensors is None:
            raise RuntimeError("Active PP DSpark batch is missing proxy tensors.")
        received = pop_proxy_proposal(proxy_tensors.tensors)
        if received is None:
            if any(identity[2] > 0 for identity in expected):
                raise RuntimeError(
                    "Missing relayed steady-state PP DSpark proposal for "
                    f"slot={slot_id}, identities={expected}."
                )
            return {}
        generation, source_slot, payload = received
        if source_slot != slot_id:
            raise RuntimeError(
                "Stale PP DSpark proxy slot: "
                f"expected={slot_id}, received={source_slot}."
            )
        payload = self._filter_proxy_payload(payload, expected_set)
        received_identities = tuple(
            (identity[0], tuple(identity[1]), identity[2])
            for identity in payload["identities"]
        )
        self._validate_proxy_identity_coverage(
            expected, expected_set, received_identities, slot_id
        )
        if not received_identities:
            return {}
        scheduler.model_worker.install_pp_draft_subset(batch, payload)
        return pack_proxy_proposal(generation, source_slot, payload)

    @staticmethod
    def _filter_proxy_payload(payload: dict, expected_set: set) -> dict:
        identities = [
            (identity[0], tuple(identity[1]), identity[2])
            for identity in payload["identities"]
        ]
        rows = [
            row for row, identity in enumerate(identities) if identity in expected_set
        ]
        filtered = {"identities": [identities[row] for row in rows]}
        for key, value in payload.items():
            if key == "identities":
                continue
            if len(value) != len(identities):
                raise RuntimeError(
                    "PP DSpark proposal field row mismatch: "
                    f"field={key}, rows={len(value)}, identities={len(identities)}."
                )
            index = torch.tensor(rows, dtype=torch.int64, device=value.device)
            filtered[key] = value.index_select(0, index)
        return filtered

    @staticmethod
    def _validate_proxy_identity_coverage(
        expected: tuple,
        expected_set: set,
        received: tuple,
        slot_id: int,
    ) -> None:
        received_set = set(received)
        if len(received_set) != len(received) or not received_set.issubset(
            expected_set
        ):
            raise RuntimeError(
                "Stale or duplicate PP DSpark proxy identities for "
                f"slot={slot_id}: expected={expected}, received={received}."
            )
        missing_steady = [
            identity
            for identity in expected
            if identity[2] > 0 and identity not in received_set
        ]
        if missing_steady:
            raise RuntimeError(
                "Missing steady-state PP DSpark proposal rows for "
                f"slot={slot_id}: missing={missing_steady}."
            )

    def drain_pipelined(self) -> None:
        if not self._is_pipelined_proxy_protocol():
            self.drain()
            return
        scheduler = self._scheduler
        if not scheduler.pp_group.is_first_rank:
            if self._pending:
                raise RuntimeError("Only PP0 may queue pipelined DSpark Draft work.")
            return
        if len(self._pending) > 1:
            raise RuntimeError(
                f"PP0 accumulated {len(self._pending)} pipelined Draft epochs."
            )

        if not self._pending:
            return
        if self._pipelined_ready_event is not None:
            raise RuntimeError(
                "Previous pipelined PP DSpark Draft work was not committed."
            )
        work = self._pending.popleft()
        if work.generation is None or work.slot_id is None:
            raise RuntimeError(
                "Pipelined PP DSpark work is missing generation or slot."
            )
        global_counts = work.batch.draft_global_num_tokens
        if global_counts is None or not any(global_counts):
            raise RuntimeError(
                "Committed PP DSpark Draft epoch has no gathered row counts: "
                f"generation={work.generation}, slot={work.slot_id}, "
                f"counts={global_counts}."
            )

        with scheduler.forward_stream_ctx:
            if not work.is_idle:
                proposal = scheduler.model_worker.prepare_pp_draft(
                    work.batch, work.draft_input, all_rows=True
                )
            else:
                proposal = None
                self._run_idle_draft(
                    SimpleNamespace(
                        draft_global_num_tokens=global_counts,
                        can_run_decode_cuda_graph=True,
                    )
                )
            ready_event = scheduler.device_module.Event()
            ready_event.record(scheduler.device_module.current_stream())
        self._pipelined_ready_event = ready_event

        if not work.is_idle:
            if work.slot_id in self._proxy_outbox:
                raise RuntimeError(
                    f"Duplicate PP DSpark proposal for slot={work.slot_id}."
                )
            self._proxy_outbox[work.slot_id] = PPDSparkProxyProposal(
                generation=work.generation,
                slot_id=work.slot_id,
                payload=proposal,
            )

    def commit_pipelined(self) -> None:
        """Fence Draft work before request-scoped state can be reused."""
        ready_event = self._pipelined_ready_event
        if ready_event is None:
            return
        ready_event.synchronize()
        self._pipelined_ready_event = None

    def drain(self) -> None:
        if not self._pending:
            return

        scheduler = self._scheduler
        work = self._pending.popleft()
        scheduler._pp_commit_comm_work(self._send_work)
        with scheduler.forward_stream_ctx:
            proposal = self._run_draft(work.batch, work.draft_input)
            ready_event = scheduler.device_module.Event()
            ready_event.record(scheduler.device_module.current_stream())

        if self._is_serial_generation_protocol() and get_parallel().pp_size > 2:
            ready_event.synchronize()
            if work.pp_outputs is None:
                raise RuntimeError(
                    "PP4 serial DSpark draft work requires relayed output state."
                )
            work.pp_outputs.tensors.update(
                pack_proposal(get_parallel().pp_rank, proposal)
            )
            return

        if scheduler.pp_group.is_last_rank:
            self._send_work = scheduler._pp_send_dict_to_next_stage(
                pack_proposal(get_parallel().pp_rank, proposal),
                async_send=True,
                msg_type="dspark_draft",
                ready_event=ready_event,
            )
            return

        if not scheduler.pp_group.is_first_rank or work.pp_outputs is None:
            raise RuntimeError("Invalid PP DSpark deferred draft work owner.")
        tensor_dict = scheduler._pp_recv_typed_dict(
            expected_kind="dspark_draft",
            all_gather_group=scheduler.attn_tp_group,
        )
        work.pp_outputs.tensors.update(pack_proposal(0, proposal))
        work.pp_outputs.tensors.update(tensor_dict)
        for owner in range(get_parallel().pp_size):
            scheduler.model_worker.install_pp_draft(
                work.batch,
                owner,
                unpack_proposal(owner, work.pp_outputs.tensors),
            )

    def drain_aligned(self, expected_generation: Optional[int]) -> None:
        """Run one PP4 Draft epoch after the serial output phase fence."""
        if not (self._is_serial_generation_protocol() and get_parallel().pp_size > 2):
            self.drain()
            return
        if len(self._pending) > 1:
            raise RuntimeError(
                f"PP4 DSpark accumulated {len(self._pending)} Draft epochs."
            )

        scheduler = self._scheduler
        work = self._pending.popleft() if self._pending else None
        parallel = get_parallel()
        if work is not None and work.generation is None:
            raise RuntimeError("Aligned PP4 Draft work is missing its generation.")
        local_owner_bs = (
            sum(
                draft_owner(req.rid, parallel.pp_size) == parallel.pp_rank
                for req in work.batch.reqs
            )
            if work is not None
            else 0
        )
        local_state = torch.tensor(
            [
                int(work is not None),
                work.generation if work is not None else -1,
                local_owner_bs,
            ],
            dtype=torch.int64,
        )
        dp_group = get_tp_group()
        gathered_states = [
            torch.empty_like(local_state) for _ in range(dp_group.world_size)
        ]
        torch.distributed.all_gather(
            gathered_states,
            local_state,
            group=dp_group.cpu_group,
        )
        states = [state.tolist() for state in gathered_states]
        generations = {int(state[1]) for state in states if state[0]}
        if not generations:
            return
        if len(generations) != 1:
            raise RuntimeError(
                f"PP4 DSpark DP lanes disagree on Draft generation: {states}."
            )
        generation = generations.pop()
        if expected_generation is not None and generation != expected_generation:
            raise RuntimeError(
                "PP4 DSpark Draft generation mismatch: "
                f"expected={expected_generation}, gathered={generation}."
            )

        global_owner_counts = [int(state[2]) for state in states]
        if work is not None:
            work.batch.draft_global_num_tokens = global_owner_counts
            work.batch.draft_global_num_tokens_for_logprob = global_owner_counts

        scheduler._pp_commit_comm_work(self._send_work)
        with scheduler.forward_stream_ctx:
            proposal = (
                self._run_draft(work.batch, work.draft_input)
                if work is not None
                else None
            )
            if work is None and any(global_owner_counts):
                self._run_idle_draft(
                    SimpleNamespace(
                        draft_global_num_tokens=global_owner_counts,
                        can_run_decode_cuda_graph=True,
                    )
                )
            ready_event = scheduler.device_module.Event()
            ready_event.record(scheduler.device_module.current_stream())
        ready_event.synchronize()
        torch.distributed.barrier(group=dp_group.cpu_group)

        if work is not None:
            if work.pp_outputs is None:
                raise RuntimeError("Aligned PP4 Draft work has no output envelope.")
            work.pp_outputs.tensors.update(pack_proposal(parallel.pp_rank, proposal))

    def relay_serial_proposals(
        self,
        batch: Optional[ScheduleBatch],
        pp_outputs: PPProxyTensors,
        transport_generation: int,
        ready_event: Optional[torch.Event],
    ) -> None:
        """Relay PP0's complete proposal set back to PP1 before Target launch."""
        scheduler = self._scheduler
        active = batch is not None
        if get_parallel().pp_size > 2:
            if not active:
                return
            for owner in range(get_parallel().pp_size):
                local_payload = (
                    pack_proposal(
                        owner,
                        unpack_proposal(owner, pp_outputs.tensors),
                    )
                    if get_parallel().pp_rank == owner
                    else None
                )
                owner_payload = scheduler.pp_group.broadcast_tensor_dict(
                    local_payload,
                    src=owner,
                )
                pp_outputs.tensors.update(owner_payload)
            for owner in range(get_parallel().pp_size):
                scheduler.model_worker.install_pp_draft(
                    batch,
                    owner,
                    unpack_proposal(owner, pp_outputs.tensors),
                )
            return

        send_work = []
        if scheduler.pp_group.is_first_rank:
            tensor_dict = {
                "__pp_generation__": transport_generation,
                "__pp_active__": active,
            }
            if active:
                for owner in range(get_parallel().pp_size):
                    tensor_dict.update(
                        pack_proposal(
                            owner,
                            unpack_proposal(owner, pp_outputs.tensors),
                        )
                    )
            send_work = scheduler._pp_send_dict_to_next_stage(
                tensor_dict,
                async_send=True,
                msg_type="dspark_draft_serial",
                ready_event=ready_event,
            )
        else:
            tensor_dict = scheduler._pp_recv_typed_dict(
                expected_kind="dspark_draft_serial",
                all_gather_group=scheduler.attn_tp_group,
            )
            generation = tensor_dict.pop("__pp_generation__", None)
            received_active = tensor_dict.pop("__pp_active__", None)
            if generation != transport_generation:
                raise RuntimeError(
                    "Stale serial PP DSpark proposal envelope: "
                    f"expected generation {transport_generation}, got {generation}"
                )
            if received_active is not active:
                raise RuntimeError(
                    "Serial PP DSpark proposal activity mismatch for generation "
                    f"{transport_generation}: expected {active}, "
                    f"got {received_active}"
                )
            if active:
                for owner in range(get_parallel().pp_size):
                    scheduler.model_worker.install_pp_draft(
                        batch,
                        owner,
                        unpack_proposal(owner, tensor_dict),
                    )
        scheduler._pp_commit_comm_work(send_work)

    def _run_draft(self, batch: ScheduleBatch, draft_input: object):
        scheduler = self._scheduler
        if not self._is_bubble_policy():
            return scheduler.model_worker.prepare_pp_draft(batch, draft_input)

        with torch.profiler.record_function("pp_dspark_draft_bubble"):
            return scheduler.model_worker.prepare_pp_draft(batch, draft_input)

    def _run_idle_draft(self, batch: ScheduleBatch) -> None:
        with torch.profiler.record_function("pp_dspark_draft_bubble"):
            self._scheduler.model_worker.prepare_pp_idle_draft(batch)

    @staticmethod
    def _is_bubble_policy() -> bool:
        return get_spec().speculative_draft_scheduling_policy == "bubble"

    @staticmethod
    def _is_pipelined_proxy_protocol() -> bool:
        return (
            envs.SGLANG_PP_DSPARK_PIPELINED_PROXY.get()
            and get_parallel().pp_size == 4
            and get_spec().speculative_dspark_pp_replicated_draft
            and get_spec().speculative_draft_scheduling_policy == "bubble"
        )

    @staticmethod
    def _is_serial_generation_protocol() -> bool:
        return envs.SGLANG_PP_DSPARK_SERIAL_GENERATION.get()
