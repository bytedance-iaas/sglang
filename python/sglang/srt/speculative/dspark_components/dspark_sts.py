from __future__ import annotations

from pathlib import Path

import msgspec
import torch


class DSparkStsCalibration(msgspec.Struct, frozen=True, omit_defaults=True):
    temperatures: list[float]
    dataset: str = ""
    num_samples: int = 0
    ece_before: list[float] = []
    ece_after: list[float] = []

    def __post_init__(self) -> None:
        if not self.temperatures:
            raise ValueError("DSparkStsCalibration requires at least one temperature.")
        for temperature in self.temperatures:
            if temperature <= 0:
                raise ValueError(
                    "DSparkStsCalibration temperatures must all be > 0, got "
                    f"{self.temperatures}."
                )

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> DSparkStsCalibration:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


def load_sts_calibration_from_path(path: str) -> DSparkStsCalibration:
    with open(path, "r", encoding="utf-8") as f:
        return DSparkStsCalibration.from_json(f.read())


class StsDataRecorder:
    def __init__(self, *, path_stem: str, gamma: int, flush_every: int) -> None:
        self.path_stem = path_stem
        self.gamma = int(gamma)
        self.flush_every = int(flush_every)
        self._logits_buffer: list[torch.Tensor] = []
        self._prefix_mask_buffer: list[torch.Tensor] = []
        self._shard_ct = 0
        self._metadata_buffer = []

    def record(
        self,
        *,
        confidence_raw: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        rids=None,
        prefix_lens=None,
        verify_lens=None,
    ) -> None:
        # Collection is diagnostic: synchronize here and reject censored rows.
        logits = confidence_raw.detach().to(device="cpu", dtype=torch.float32).clone()
        counts = (
            num_correct_drafts.detach().to(device="cpu", dtype=torch.int64).view(-1)
        )
        if logits.ndim != 2 or logits.shape != (counts.numel(), self.gamma):
            raise ValueError("STS logits/acceptance shape mismatch")
        if (
            not torch.isfinite(logits).all()
            or not ((counts >= 0) & (counts <= self.gamma)).all()
        ):
            raise ValueError("Invalid STS logits or acceptance counts")
        keep = torch.ones(counts.numel(), dtype=torch.bool)
        if verify_lens is not None:
            widths = verify_lens.detach().cpu().view(-1)
            if widths.numel() != counts.numel():
                raise ValueError("STS verify length shape mismatch")
            keep = widths == self.gamma + 1
        if not keep.any():
            return
        if rids is not None:
            if len(rids) != counts.numel() or prefix_lens is None:
                raise ValueError("STS request metadata shape mismatch")
            contexts = prefix_lens.detach().cpu().view(-1).tolist()
            if len(contexts) != len(rids):
                raise ValueError("STS prefix length shape mismatch")
            metadata = [
                dict(rid=rid, prefix_len=int(ctx), local_bs=len(rids))
                for rid, ctx, valid in zip(rids, contexts, keep.tolist())
                if valid
            ]
        else:
            metadata = [None] * int(keep.sum())
        positions = torch.arange(self.gamma).view(1, -1)
        self._logits_buffer.append(logits[keep])
        self._prefix_mask_buffer.append((positions < counts[keep, None]).float())
        self._metadata_buffer.extend(metadata)
        if len(self._logits_buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._logits_buffer:
            return
        shard_path = Path(f"{self.path_stem}.{self._shard_ct}.pt")
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "logits": torch.cat(self._logits_buffer, dim=0),
                "prefix_mask": torch.cat(self._prefix_mask_buffer, dim=0),
                "metadata": self._metadata_buffer,
            },
            shard_path,
        )
        self._logits_buffer.clear()
        self._prefix_mask_buffer.clear()
        self._metadata_buffer.clear()
        self._shard_ct += 1
