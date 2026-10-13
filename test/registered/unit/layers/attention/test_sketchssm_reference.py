# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from sglang.srt.layers.attention.linear.sketchssm_reference import (
    SketchKDAReference,
    SketchLifecycle,
    SketchStateError,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _orthogonal_frames(heads: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(heads, dim, dim, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(raw).Q


def _reference(*, ranks=(3, 0, 8), window=16, seed=7, generation=1):
    heads, dim = len(ranks), 8
    generator = torch.Generator().manual_seed(seed)
    return SketchKDAReference(
        state=torch.randn(heads, dim, dim, generator=generator, dtype=torch.float64)
        * 0.1,
        frames=_orthogonal_frames(heads, dim, seed + 1),
        ranks=ranks,
        a_log=torch.randn(heads, generator=generator, dtype=torch.float64) * 0.1,
        dt_bias=torch.randn(heads, dim, generator=generator, dtype=torch.float64) * 0.1,
        window=window,
        slot_generation=generation,
    )


def _inputs(tokens: int, heads=3, dim=8, seed=19):
    generator = torch.Generator().manual_seed(seed)
    q, k, v, gate = (
        torch.randn(tokens, heads, dim, generator=generator, dtype=torch.float64)
        for _ in range(4)
    )
    gate.sub_(3.0)
    beta = torch.randn(tokens, heads, generator=generator, dtype=torch.float64)
    return q, k, v, gate, beta


def _stage(reference, data, proposed_drafts=None):
    return reference.stage(
        q=data[0],
        k=data[1],
        v=data[2],
        gate=data[3],
        beta=data[4],
        proposed_drafts=proposed_drafts,
    )


class TestSketchSSMReference(CustomTestCase):
    def test_target_verify_advances_only_the_persistent_commit_frontier(self):
        candidate = _reference()
        control = _reference()
        verify = _inputs(6)

        transaction = _stage(candidate, verify, proposed_drafts=5)
        receipt = transaction.commit(accepted_drafts=2, committed_bonus=1)
        _stage(control, tuple(value[:3] for value in verify), proposed_drafts=2).commit(
            accepted_drafts=2, committed_bonus=1
        )

        self.assertEqual((receipt.start, receipt.end), (0, 3))
        self.assertEqual(receipt.discarded_rows, 3)
        self.assertEqual(candidate.position, 3)
        self.assertEqual(candidate.committed_tokens, 3)
        torch.testing.assert_close(candidate.state, control.state, rtol=0, atol=0)

    def test_abort_discards_every_staged_state_and_ring_update(self):
        reference = _reference()
        before = reference.state
        transaction = _stage(reference, _inputs(6), proposed_drafts=5)
        transaction.abort()

        self.assertEqual(reference.position, 0)
        self.assertEqual(reference.committed_tokens, 0)
        torch.testing.assert_close(reference.state, before, rtol=0, atol=0)
        with self.assertRaisesRegex(SketchStateError, "already finished"):
            transaction.commit(accepted_drafts=1, committed_bonus=1)

    def test_flush_output_and_exact_state_ignore_corrupted_sketch_map(self):
        clean = _reference(ranks=(3, 4, 0), window=4)
        corrupt = _reference(ranks=(3, 4, 0), window=4)
        data = _inputs(4, seed=31)

        clean_tx = _stage(clean, data, proposed_drafts=3)
        corrupt_tx = _stage(corrupt, data, proposed_drafts=3)
        for value in corrupt_tx.snapshots[-2].maps:
            if value is not None:
                value.fill_(float("nan"))

        # The final staged snapshot was computed from the uncorrupted prior
        # one, so explicitly corrupt the live working map and recompute the
        # flush row through a second transaction rooted at position W - 1.
        clean_tx.commit(accepted_drafts=2, committed_bonus=1)
        corrupt_tx.commit(accepted_drafts=2, committed_bonus=1)
        for value in corrupt._snapshot.maps:
            if value is not None:
                value.fill_(float("nan"))
        flush_clean = _stage(
            clean, tuple(value[3:4] for value in data), proposed_drafts=0
        )
        flush_corrupt = _stage(
            corrupt, tuple(value[3:4] for value in data), proposed_drafts=0
        )

        self.assertTrue(flush_clean.outputs[0].flushed)
        torch.testing.assert_close(
            flush_corrupt.outputs[0].approximate,
            flush_clean.outputs[0].exact,
            rtol=0,
            atol=0,
        )
        flush_clean.commit(accepted_drafts=0, committed_bonus=1)
        flush_corrupt.commit(accepted_drafts=0, committed_bonus=1)
        torch.testing.assert_close(clean.state, corrupt.state, rtol=0, atol=0)

    def test_dense_heads_use_exact_readout_on_non_flush_rows(self):
        reference = _reference(ranks=(0, 8, 3))
        tx = _stage(reference, _inputs(1), proposed_drafts=0)
        output = tx.outputs[0]

        torch.testing.assert_close(
            output.approximate[:2], output.exact[:2], rtol=0, atol=0
        )
        self.assertGreater(float((output.approximate[2] - output.exact[2]).norm()), 0.0)

    def test_stale_transaction_cannot_write_a_reused_slot(self):
        reference = _reference(generation=4)
        transaction = _stage(reference, _inputs(2), proposed_drafts=1)
        replacement = torch.zeros_like(reference.state)
        reference.replace_slot(state=replacement, slot_generation=5)

        with self.assertRaisesRegex(SketchStateError, "stale slot generation"):
            transaction.commit(accepted_drafts=1, committed_bonus=1)
        torch.testing.assert_close(reference.state, replacement, rtol=0, atol=0)

    def test_close_and_release_fail_closed(self):
        reference = _reference()
        reference.close()
        self.assertEqual(reference.lifecycle, SketchLifecycle.CLOSED)
        with self.assertRaisesRegex(SketchStateError, "closed, not live"):
            _stage(reference, _inputs(1), proposed_drafts=0)
        reference.release()
        self.assertEqual(reference.lifecycle, SketchLifecycle.RELEASED)

    def test_long_trajectory_preserves_exact_full_state(self):
        sketch = _reference(ranks=(3, 4, 0), window=16, seed=47)
        dense = _reference(ranks=(0, 0, 0), window=16, seed=47)
        data = _inputs(256, seed=53)
        approximation_errors = []
        flushes = 0

        for start in range(0, 256, 6):
            stop = min(start + 6, 256)
            chunk = tuple(value[start:stop] for value in data)
            proposed = stop - start - 1
            sketch_tx = _stage(sketch, chunk, proposed_drafts=proposed)
            dense_tx = _stage(dense, chunk, proposed_drafts=proposed)
            for output in sketch_tx.outputs:
                self.assertTrue(torch.isfinite(output.approximate).all())
                error = float(
                    (output.approximate - output.exact).norm()
                    / output.exact.norm().clamp_min(1e-30)
                )
                approximation_errors.append(error)
                if output.flushed:
                    flushes += 1
                    self.assertEqual(error, 0.0)
            sketch_tx.commit(accepted_drafts=proposed, committed_bonus=1)
            dense_tx.commit(accepted_drafts=proposed, committed_bonus=1)

        self.assertEqual(sketch.committed_tokens, 256)
        self.assertEqual(sketch.position, 0)
        self.assertEqual(flushes, 16)
        self.assertGreater(max(approximation_errors), 0.0)
        torch.testing.assert_close(sketch.state, dense.state, rtol=0, atol=0)

    def test_invalid_accept_bonus_and_geometry_fail_closed(self):
        transaction = _stage(_reference(), _inputs(6), proposed_drafts=5)
        with self.assertRaisesRegex(SketchStateError, "outside the proposed"):
            transaction.commit(accepted_drafts=6, committed_bonus=0)
        with self.assertRaisesRegex(SketchStateError, "zero or one"):
            transaction.commit(accepted_drafts=1, committed_bonus=2)
        with self.assertRaisesRegex(SketchStateError, "proposed drafts plus one"):
            _stage(_reference(), _inputs(6), proposed_drafts=4)


if __name__ == "__main__":
    unittest.main()
