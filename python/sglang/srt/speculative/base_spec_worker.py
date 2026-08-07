from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs


class HiCacheDraftMode(str, Enum):
    NONE = "none"
    PACKED = "packed"
    SIDECAR = "sidecar"


@dataclass(frozen=True, slots=True)
class HiCacheDraftPlan:
    mode: HiCacheDraftMode = HiCacheDraftMode.NONE
    device_pools: tuple[object, ...] = ()


def build_hicache_draft_plan(
    *,
    target_model_runner: ModelRunner,
    draft_runners: tuple[ModelRunner, ...],
    server_args: ServerArgs,
) -> HiCacheDraftPlan:
    target_model_runner.mtp_draft_device_pools = ()
    if not server_args.enable_hierarchical_cache or not draft_runners:
        return HiCacheDraftPlan()

    draft_pools = tuple(runner.token_to_kv_pool for runner in draft_runners)
    spec_algorithm = target_model_runner.spec_algorithm
    is_dsv4_nextn_mtp = (
        spec_algorithm.is_eagle()
        and not spec_algorithm.is_eagle3()
        and all(
            runner.model_config.num_nextn_predict_layers
            and "DeepseekV4ForCausalLMNextN"
            in runner.model_config.hf_config.architectures
            for runner in draft_runners
        )
    )
    if is_dsv4_nextn_mtp:
        if server_args.enable_eic_cache:
            raise NotImplementedError(
                "Packed DeepSeek-V4 NextN draft cache currently supports "
                "standard HiCache L2 only; EIC integration is not included."
            )
        if server_args.hicache_storage_backend is not None:
            raise NotImplementedError(
                "This DeepSeek-V4 backport supports packed built-in NextN "
                "draft cache for HiCache L2 only; L3 draft storage requires "
                "the remaining upstream #30393 storage-backend changes."
            )
        target_model_runner.mtp_draft_device_pools = draft_pools
        return HiCacheDraftPlan(
            mode=HiCacheDraftMode.PACKED,
            device_pools=draft_pools,
        )

    return HiCacheDraftPlan(
        mode=HiCacheDraftMode.SIDECAR,
        device_pools=draft_pools[:1],
    )


class BaseDraftWorker(ABC):
    @property
    def draft_runners(self) -> list[ModelRunner]:
        """Return every draft runner participating in speculative decoding."""
        return getattr(self, "draft_runner_list", [self.draft_runner])

    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass


class BaseSpecWorker(ABC):
    _hicache_draft_plan = HiCacheDraftPlan()

    @property
    def hicache_draft_plan(self) -> HiCacheDraftPlan:
        return self._hicache_draft_plan

    def _draft_model_runners(self) -> tuple[ModelRunner, ...]:
        spec_algorithm = self.target_worker.model_runner.spec_algorithm
        draft_worker = self.draft_worker
        if (
            draft_worker is None
            or spec_algorithm.is_ngram()
            or spec_algorithm.is_frozen_kv_mtp()
        ):
            return ()
        return tuple(draft_worker.draft_runners)

    @property
    def primary_draft_kv_pool(self) -> object | None:
        draft_runners = self._draft_model_runners()
        return draft_runners[0].token_to_kv_pool if draft_runners else None

    def _build_hicache_draft_plan(self) -> HiCacheDraftPlan:
        return build_hicache_draft_plan(
            target_model_runner=self.target_worker.model_runner,
            draft_runners=self._draft_model_runners(),
            server_args=self.server_args,
        )

    def init_hicache_draft_plan(self) -> None:
        self._hicache_draft_plan = self._build_hicache_draft_plan()

    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        pass

    @abstractmethod
    def clear_cache_pool(self):
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        pass

    def on_verify_complete_cpu(self, num_correct_drafts_per_req: list[int]) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU→CPU sync in the worker hot path.
        """
        pass
