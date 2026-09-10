"""CPU contracts for the exact-request EAGLE numerical probe."""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.eagle_numerical_probe import (
    EagleNumericalProbe,
    _synchronize_cuda_tensors,
    _tensors_share_storage,
    maybe_record_eagle_numerical_stage,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEagleNumericalProbe(unittest.TestCase):
    @staticmethod
    def probe(expected_rid="probe-rid"):
        return EagleNumericalProbe(
            expected_rid,
            capture_id="capture-test-generation" if expected_rid else None,
            pod_name="probe-pod" if expected_rid else None,
            pod_uid="probe-pod-uid" if expected_rid else None,
        )

    @staticmethod
    def record_target_verify(probe, *, batch_size=1, draft_token_num=2):
        rows = batch_size * draft_token_num
        hidden = torch.arange(rows * 4, dtype=torch.bfloat16).reshape(rows, 4)
        active = probe.record_target_verify_input(
            rids=["probe-rid"],
            draft_token=torch.arange(10, 10 + rows, dtype=torch.int64),
            positions=torch.arange(100, 100 + rows, dtype=torch.int64),
            retrieve_index=torch.arange(rows, dtype=torch.int64).reshape(
                batch_size, draft_token_num
            ),
            retrieve_next_token=torch.full(
                (batch_size, draft_token_num), -1, dtype=torch.int64
            ),
            retrieve_next_sibling=torch.full(
                (batch_size, draft_token_num), -1, dtype=torch.int64
            ),
            batch_size=batch_size,
            draft_token_num=draft_token_num,
        )
        probe.record_target_verify_output(
            logits=torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 6),
            hidden_states=hidden,
            logical_rows=rows,
        )
        probe.record_target_verify_sample(
            predict=torch.arange(20, 20 + rows, dtype=torch.int32),
            logical_rows=rows,
        )
        probe.record_target_verify_accept(
            accept_lens=torch.ones(batch_size, dtype=torch.int32),
            accept_index=torch.zeros((batch_size, 2), dtype=torch.int32),
            batch_size=batch_size,
        )
        probe.record_target_verify_handoff(
            predict=torch.arange(20, 20 + rows, dtype=torch.int32),
            hidden_states=hidden,
            logical_rows=rows,
        )
        return active

    def test_default_off_is_noop(self):
        probe = self.probe(None)
        batch = SimpleNamespace(reqs=[SimpleNamespace(rid="probe-rid")])

        self.assertFalse(probe.can_probe)
        self.assertFalse(probe.matches_schedule_batch(batch))

    def test_capture_identity_is_required_exactly_when_probe_is_armed(self):
        identity = {
            "capture_id": "capture-test-generation",
            "pod_name": "probe-pod",
            "pod_uid": "probe-pod-uid",
        }
        for key in identity:
            for replacement in (None, ""):
                with self.subTest(armed_missing=key, replacement=replacement):
                    incomplete = identity | {key: replacement}
                    with self.assertRaisesRegex(ValueError, "requires capture id"):
                        EagleNumericalProbe("probe-rid", **incomplete)
            with self.subTest(disarmed_with=key):
                with self.assertRaisesRegex(ValueError, "requires an exact request id"):
                    EagleNumericalProbe(None, **{key: identity[key]})

    def test_exact_rid_records_complete_decode_fingerprint(self):
        probe = self.probe()
        forward_batch = SimpleNamespace(
            rids=["probe-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        input_ids = torch.tensor([11, 12, 99], dtype=torch.int64)
        positions = torch.tensor([101, 102, 999], dtype=torch.int64)
        target_hidden = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)

        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.get_parallel",
                return_value=SimpleNamespace(world_rank=13, pp_rank=1, attn_dp_rank=5),
            ),
            self.assertLogs(
                "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
            ) as logs,
        ):
            self.assertTrue(self.record_target_verify(probe))
            with probe.forward_scope(
                forward_batch,
                phase="decode",
                logical_rows=2,
                using_cuda_graph=False,
                input_ids=input_ids,
                target_hidden_states=target_hidden,
                positions=positions,
            ) as active:
                self.assertTrue(active)
                for stage, tensor in (
                    ("nextn_embed", target_hidden + 1),
                    ("nextn_decoder", target_hidden + 2),
                    ("nextn_norm", target_hidden + 3),
                ):
                    maybe_record_eagle_numerical_stage(
                        forward_batch, stage, hidden_states=tensor
                    )
                maybe_record_eagle_numerical_stage(
                    forward_batch,
                    "nextn_logits",
                    logits=torch.arange(18).reshape(3, 6).float(),
                )

            probe.record_proposal(
                phase="decode",
                logical_rows=1,
                topk_index=torch.tensor([[3], [4], [5]]),
                topk_probability=torch.tensor([[0.7], [0.8], [0.9]]),
            )
            probe.finish(rid="probe-rid", natural_stop=True, normal_completion=True)

        self.assertIsNone(forward_batch._eagle_numerical_probe_callback)
        self.assertIsNone(forward_batch._eagle_numerical_probe_phase)
        stage_payloads = [
            json.loads(line.split("EAGLE_NUMERICAL_PROBE_STAGE ", 1)[1])
            for line in logs.output
            if "EAGLE_NUMERICAL_PROBE_STAGE " in line
        ]
        self.assertEqual(
            [payload["stage"] for payload in stage_payloads],
            [
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
            ],
        )
        self.assertEqual(
            [payload["ordinal"] for payload in stage_payloads], list(range(1, 12))
        )
        self.assertTrue(
            all(
                payload["capture"] == probe.capture
                and payload["rank"] == {"world": 13, "pp": 1, "attn_dp": 5}
                for payload in stage_payloads
            )
        )
        payload = json.loads(
            logs.output[-1].split("EAGLE_NUMERICAL_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(
            payload["capture"],
            {
                "id": "capture-test-generation",
                "pod_name": "probe-pod",
                "pod_uid": "probe-pod-uid",
            },
        )
        self.assertTrue(payload["natural_stop"])
        self.assertTrue(payload["normal_completion"])
        stages = payload["phases"]["decode"]
        self.assertEqual(
            set(stages),
            {
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
            },
        )
        self.assertEqual(
            stages["target_verify_input"]["tensors"]["draft_token"]["values"],
            [10, 11],
        )
        self.assertEqual(
            stages["target_verify_sample"]["tensors"]["predict"]["values"],
            [20, 21],
        )
        self.assertEqual(
            stages["target_verify_accept"]["tensors"]["accept_lens"]["values"],
            [1],
        )
        self.assertEqual(
            stages["draft_extend_input"]["tensors"]["input_ids"]["values"],
            [11, 12],
        )
        self.assertEqual(
            stages["proposed_token"]["tensors"]["topk_index"]["values"],
            [3],
        )
        self.assertEqual(stages["draft_extend_input"]["logical_rows"], 2)
        self.assertEqual(stages["target_verify_input"]["logical_rows"], 2)
        self.assertEqual(stages["target_verify_accept"]["logical_rows"], 1)
        self.assertEqual(stages["proposed_token"]["logical_rows"], 1)
        self.assertEqual(
            stages["target_verify_input"]["row_domain"],
            "target_verify_tree_node",
        )
        self.assertEqual(stages["target_verify_accept"]["row_domain"], "request")
        self.assertEqual(
            stages["draft_extend_input"]["row_domain"],
            "dense_request_major_prefix",
        )
        self.assertEqual(stages["proposed_token"]["row_domain"], "request_terminal")
        self.assertEqual(
            stages["nextn_embed"]["tensors"]["hidden_states"]["shape"],
            [2, 4],
        )
        self.assertEqual(
            len(stages["nextn_embed"]["tensors"]["hidden_states"]["sha256"]),
            64,
        )
        self.assertEqual(payload["rank"], {"world": 13, "pp": 1, "attn_dp": 5})

    def test_cuda_stage_syncs_each_distinct_tensor_device(self):
        tensors = {
            "a": SimpleNamespace(is_cuda=True, device=torch.device("cuda:1")),
            "b": SimpleNamespace(is_cuda=True, device=torch.device("cuda:0")),
            "same_as_a": SimpleNamespace(is_cuda=True, device=torch.device("cuda:1")),
            "cpu": SimpleNamespace(is_cuda=False, device=torch.device("cpu")),
            "missing": None,
        }
        streams = {}

        def current_stream(*, device):
            stream = mock.Mock()
            streams[str(device)] = stream
            return stream

        with mock.patch("torch.cuda.current_stream", side_effect=current_stream):
            _synchronize_cuda_tensors(tensors)

        self.assertEqual(list(streams), ["cuda:0", "cuda:1"])
        streams["cuda:0"].synchronize.assert_called_once_with()
        streams["cuda:1"].synchronize.assert_called_once_with()

    def test_target_verify_prefix_survives_cuda_runtime_error(self):
        probe = self.probe()
        tensor = torch.ones((2, 2))
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe._synchronize_cuda_tensors",
                side_effect=[None, None, RuntimeError("CUDA illegal memory access")],
            ),
            self.assertLogs(
                "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
            ) as logs,
            self.assertRaisesRegex(RuntimeError, "CUDA illegal memory access"),
        ):
            self.assertTrue(
                probe.record_target_verify_input(
                    rids=["probe-rid"],
                    draft_token=torch.tensor([10, 11]),
                    positions=torch.tensor([20, 21]),
                    retrieve_index=torch.tensor([[0, 1]]),
                    retrieve_next_token=torch.tensor([[1, -1]]),
                    retrieve_next_sibling=torch.tensor([[-1, -1]]),
                    batch_size=1,
                    draft_token_num=2,
                )
            )
            probe.record_target_verify_output(
                logits=torch.ones((2, 3)),
                hidden_states=tensor,
                logical_rows=2,
            )
            probe.record_target_verify_sample(
                predict=torch.tensor([7, 8]), logical_rows=2
            )

        events = [
            (
                "error" if "EAGLE_NUMERICAL_PROBE_STAGE_ERROR " in line else "stage",
                json.loads(
                    line.split("EAGLE_NUMERICAL_PROBE_STAGE_ERROR ", 1)[1]
                    if "EAGLE_NUMERICAL_PROBE_STAGE_ERROR " in line
                    else line.split("EAGLE_NUMERICAL_PROBE_STAGE ", 1)[1]
                ),
            )
            for line in logs.output
            if "EAGLE_NUMERICAL_PROBE_STAGE" in line
        ]
        self.assertEqual(
            [(kind, payload["stage"]) for kind, payload in events],
            [
                ("stage", "target_verify_input"),
                ("stage", "target_verify_output"),
                ("error", "target_verify_sample"),
            ],
        )
        self.assertEqual(events[-1][1]["ordinal"], 3)
        self.assertEqual(events[-1][1]["capture"], probe.capture)
        self.assertFalse(probe.can_probe)

    def test_co_batched_rid_defers_without_forcing_eager_or_rejecting(self):
        probe = self.probe()
        co_batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="probe-rid"), SimpleNamespace(rid="other")]
        )
        self.assertFalse(probe.needs_eager_for_schedule_batch(co_batch))
        self.assertTrue(probe.can_probe)

        sole_batch = SimpleNamespace(reqs=[SimpleNamespace(rid="probe-rid")])
        self.assertTrue(probe.needs_eager_for_schedule_batch(sole_batch))

        # Starting target verify must not release the eager vote: the same
        # iteration's downstream draft-extend still has to remain outside its
        # CUDA graph before the capture is complete.
        self.assertTrue(self.record_target_verify(probe))
        self.assertTrue(probe.needs_eager_for_schedule_batch(sole_batch))

    def test_wrong_rid_and_second_decode_do_not_capture(self):
        probe = self.probe()
        wrong = SimpleNamespace(
            rids=["other-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        tensor = torch.ones((1, 2))
        with probe.forward_scope(
            wrong,
            phase="decode",
            logical_rows=1,
            using_cuda_graph=False,
            input_ids=torch.ones(1, dtype=torch.int64),
            target_hidden_states=tensor,
            positions=torch.ones(1, dtype=torch.int64),
        ) as active:
            self.assertFalse(active)

        self.assertTrue(self.record_target_verify(probe))
        matching = SimpleNamespace(
            rids=["probe-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        with probe.forward_scope(
            matching,
            phase="decode",
            logical_rows=1,
            using_cuda_graph=False,
            input_ids=torch.ones(1, dtype=torch.int64),
            target_hidden_states=tensor,
            positions=torch.ones(1, dtype=torch.int64),
        ) as active:
            self.assertTrue(active)
        with probe.forward_scope(
            matching,
            phase="decode",
            logical_rows=1,
            using_cuda_graph=False,
            input_ids=torch.ones(1, dtype=torch.int64),
            target_hidden_states=tensor,
            positions=torch.ones(1, dtype=torch.int64),
        ) as active:
            self.assertFalse(active)

    def test_missing_required_tensor_fails_closed(self):
        probe = self.probe()
        self.assertTrue(
            probe.record_target_verify_input(
                rids=["probe-rid"],
                draft_token=torch.ones(1, dtype=torch.int64),
                positions=torch.ones(1, dtype=torch.int64),
                retrieve_index=torch.zeros((1, 1), dtype=torch.int64),
                retrieve_next_token=torch.full((1, 1), -1, dtype=torch.int64),
                retrieve_next_sibling=torch.full((1, 1), -1, dtype=torch.int64),
                batch_size=1,
                draft_token_num=1,
            )
        )
        probe.record_target_verify_output(
            logits=torch.ones((1, 2)), hidden_states=None, logical_rows=1
        )
        self.assertFalse(probe.can_probe)

    def test_out_of_order_target_verify_stage_fails_closed(self):
        probe = self.probe()
        probe.record_target_verify_sample(
            predict=torch.ones(1, dtype=torch.int32), logical_rows=1
        )
        self.assertFalse(probe.can_probe)

    def test_wrong_rid_does_not_start_target_verify_capture(self):
        probe = self.probe()
        self.assertFalse(
            probe.record_target_verify_input(
                rids=["other-rid"],
                draft_token=torch.ones(1, dtype=torch.int64),
                positions=torch.ones(1, dtype=torch.int64),
                retrieve_index=torch.zeros((1, 1), dtype=torch.int64),
                retrieve_next_token=torch.full((1, 1), -1, dtype=torch.int64),
                retrieve_next_sibling=torch.full((1, 1), -1, dtype=torch.int64),
                batch_size=1,
                draft_token_num=1,
            )
        )
        self.assertTrue(probe.can_probe)

    def test_missing_later_required_tensor_fails_closed(self):
        probe = self.probe()
        self.assertTrue(self.record_target_verify(probe))
        forward_batch = SimpleNamespace(
            rids=["probe-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        tensor = torch.ones((1, 2))
        with probe.forward_scope(
            forward_batch,
            phase="decode",
            logical_rows=1,
            using_cuda_graph=False,
            input_ids=torch.ones(1, dtype=torch.int64),
            target_hidden_states=tensor,
            positions=torch.ones(1, dtype=torch.int64),
        ):
            maybe_record_eagle_numerical_stage(
                forward_batch, "nextn_logits", logits=None
            )
        self.assertFalse(probe.can_probe)

    def test_proposal_row_domain_mismatch_fails_closed(self):
        probe = self.probe()
        self.assertTrue(self.record_target_verify(probe))
        forward_batch = SimpleNamespace(
            rids=["probe-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        tensor = torch.ones((2, 2))
        with probe.forward_scope(
            forward_batch,
            phase="decode",
            logical_rows=2,
            using_cuda_graph=False,
            input_ids=torch.ones(2, dtype=torch.int64),
            target_hidden_states=tensor,
            positions=torch.ones(2, dtype=torch.int64),
        ):
            pass
        probe.record_proposal(
            phase="decode",
            logical_rows=2,
            topk_index=torch.ones((2, 1), dtype=torch.int64),
            topk_probability=torch.ones((2, 1)),
        )
        self.assertFalse(probe.can_probe)

    def test_missing_stage_fails_closed(self):
        probe = self.probe()
        probe._seen = True

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.finish(rid="probe-rid", natural_stop=False, normal_completion=True)

        payload = json.loads(
            logs.output[-1].split("EAGLE_NUMERICAL_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["capture"], self.probe().capture)
        self.assertIn("phase decode missing stages", payload["rejection"])

    def test_abnormal_completion_fails_closed_and_preserves_stop_reason(self):
        probe = self.probe()
        probe._seen = True

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.finish(rid="probe-rid", natural_stop=False, normal_completion=False)

        payload = json.loads(
            logs.output[-1].split("EAGLE_NUMERICAL_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["rejection"], "request did not complete normally")
        self.assertFalse(payload["natural_stop"])
        self.assertFalse(payload["normal_completion"])

    def test_prefill_handoff_records_value_change_and_storage_alias(self):
        probe = self.probe()
        sampled = torch.tensor([8451], dtype=torch.int64)

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.record_prefill_target_sample(
                rids=["probe-rid"], next_token_ids=sampled
            )
            bonus_tokens = sampled
            sampled.fill_(71)
            probe.record_prefill_post_draft_extend(
                rids=["probe-rid"],
                next_token_ids=sampled,
                bonus_tokens=bonus_tokens,
            )
            probe.record_prefill_pp_output(
                rids=["probe-rid"],
                next_token_ids=sampled,
                bonus_tokens=bonus_tokens,
                pp_next_token_ids=sampled,
            )

        stage_payloads = [
            json.loads(line.split("EAGLE_PREFILL_HANDOFF_PROBE_STAGE ", 1)[1])
            for line in logs.output
            if "EAGLE_PREFILL_HANDOFF_PROBE_STAGE " in line
        ]
        self.assertEqual(
            [payload["stage"] for payload in stage_payloads],
            ["target_sample", "post_draft_extend", "pp_output"],
        )
        self.assertEqual(
            stage_payloads[0]["fingerprints"]["next_token_ids"]["values"],
            [8451],
        )
        self.assertEqual(
            stage_payloads[1]["fingerprints"]["next_token_ids"]["values"],
            [71],
        )
        self.assertEqual(
            stage_payloads[1]["fingerprints"]["shared_storage"]["values"],
            [1],
        )
        result = json.loads(
            logs.output[-1].split("EAGLE_PREFILL_HANDOFF_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["phase"], "prefill_handoff")
        self.assertEqual(result["capture"], probe.capture)

    def test_prefill_handoff_wrong_order_fails_closed(self):
        probe = self.probe()
        token = torch.tensor([8451], dtype=torch.int64)

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.record_prefill_post_draft_extend(
                rids=["probe-rid"],
                next_token_ids=token,
                bonus_tokens=token,
            )
            probe.finish(rid="probe-rid", natural_stop=True, normal_completion=True)

        self.assertFalse(probe.can_probe_prefill_handoff)
        self.assertIn("out-of-order", probe._prefill_rejection)
        result_lines = [
            line
            for line in logs.output
            if "EAGLE_PREFILL_HANDOFF_PROBE_RESULT " in line
        ]
        self.assertEqual(len(result_lines), 1)
        result = json.loads(
            result_lines[0].split("EAGLE_PREFILL_HANDOFF_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["seen"])
        self.assertIn("out-of-order", result["rejection"])
        self.assertNotIn("EAGLE_NUMERICAL_PROBE_RESULT ", "\n".join(logs.output))

    def test_prefill_only_finish_seals_missing_handoff_without_decode_result(self):
        probe = self.probe()
        token = torch.tensor([8451], dtype=torch.int64)
        probe.record_prefill_target_sample(rids=["probe-rid"], next_token_ids=token)

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.finish(rid="probe-rid", natural_stop=True, normal_completion=True)

        result_lines = [
            line
            for line in logs.output
            if "EAGLE_PREFILL_HANDOFF_PROBE_RESULT " in line
        ]
        self.assertEqual(len(result_lines), 1)
        result = json.loads(
            result_lines[0].split("EAGLE_PREFILL_HANDOFF_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(result["status"], "rejected")
        self.assertIn("missing stages", result["rejection"])
        self.assertNotIn("EAGLE_NUMERICAL_PROBE_RESULT ", "\n".join(logs.output))
        self.assertFalse(probe._sealed)

    def test_storage_alias_detection_handles_views_and_copies(self):
        source = torch.tensor([1, 2, 3], dtype=torch.int64)

        self.assertTrue(_tensors_share_storage(source, source[1:]))
        self.assertFalse(_tensors_share_storage(source, source.clone()))


if __name__ == "__main__":
    unittest.main()
