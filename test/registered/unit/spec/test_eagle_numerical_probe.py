"""CPU contracts for the exact-request EAGLE numerical probe."""

import base64
import contextlib
import hashlib
import json
import stat
import unittest
import zlib
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.models.deepseek_v2 import _pp_target_forward_row_domain
from sglang.srt.speculative.eagle_numerical_probe import (
    EagleNumericalProbe,
    EaglePDHandoffProbe,
    EaglePPSenderProbe,
    _emit_json_record,
    _PPTargetForwardDeviceObserver,
    _synchronize_cuda_tensors,
    _tensor_fingerprint,
    maybe_record_eagle_numerical_stage,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEagleNumericalProbe(unittest.TestCase):
    def test_integer_row_multiset_fingerprint_is_order_independent(self):
        left = torch.tensor([[3, 1, 2], [6, 4, 5]], dtype=torch.int32)
        reordered = torch.tensor([[2, 3, 1], [5, 6, 4]], dtype=torch.int32)
        changed = torch.tensor([[2, 3, 0], [5, 6, 4]], dtype=torch.int32)

        left_fp = _tensor_fingerprint(left, 2, include_row_multiset=True)
        reordered_fp = _tensor_fingerprint(reordered, 2, include_row_multiset=True)
        changed_fp = _tensor_fingerprint(changed, 2, include_row_multiset=True)

        self.assertNotEqual(left_fp["sha256"], reordered_fp["sha256"])
        self.assertEqual(
            left_fp["row_multiset_sha256"],
            reordered_fp["row_multiset_sha256"],
        )
        self.assertNotEqual(
            left_fp["row_multiset_sha256"],
            changed_fp["row_multiset_sha256"],
        )

    def test_row_multiset_fingerprint_rejects_wrong_tensor_kind(self):
        for tensor in (
            torch.zeros(2, dtype=torch.int32),
            torch.zeros((2, 3), dtype=torch.float32),
        ):
            with self.subTest(shape=tuple(tensor.shape), dtype=tensor.dtype):
                with self.assertRaisesRegex(ValueError, "rank-2 integer tensor"):
                    _tensor_fingerprint(tensor, 1, include_row_multiset=True)

    @contextlib.contextmanager
    def capture_atomic_probe_records(self):
        records = []

        def write(fd, value):
            self.assertEqual(fd, 2)
            self.assertLessEqual(len(value), 4096)
            records.append(value)
            return len(value)

        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf",
                return_value=4096,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write",
                side_effect=write,
            ),
        ):
            yield records

    def decode_atomic_probe_record(self, line, marker=b"EAGLE_PP_SENDER_PROBE_RESULT"):
        self.assertTrue(line.endswith(b"\n"))
        actual_marker, encoded = line[:-1].split(b" ", 1)
        self.assertEqual(actual_marker, marker)
        payload = json.loads(encoded)
        if "__eagle_probe_encoding__" not in payload:
            return payload

        self.assertEqual(
            set(payload),
            {"__eagle_probe_encoding__", "payload", "raw_bytes", "sha256"},
        )
        self.assertEqual(payload["__eagle_probe_encoding__"], "zlib+base64")
        compressed = base64.b64decode(payload["payload"], validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, payload["raw_bytes"] + 1)
        raw += decompressor.flush()
        self.assertTrue(decompressor.eof)
        self.assertEqual(decompressor.unused_data, b"")
        self.assertEqual(decompressor.unconsumed_tail, b"")
        self.assertEqual(len(raw), payload["raw_bytes"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), payload["sha256"])
        return json.loads(raw)

    def decode_atomic_probe_records(
        self, lines, marker=b"EAGLE_PP_SENDER_PROBE_RESULT"
    ):
        if len(lines) == 1:
            return self.decode_atomic_probe_record(lines[0], marker=marker)
        envelopes = []
        for line in lines:
            self.assertTrue(line.endswith(b"\n"))
            actual_marker, encoded = line[:-1].split(b" ", 1)
            self.assertEqual(actual_marker, marker)
            envelopes.append(json.loads(encoded))
        self.assertEqual(
            {item["__eagle_probe_encoding__"] for item in envelopes},
            {"zlib+base64-chunk-v1"},
        )
        self.assertEqual(
            {item["record_id"] for item in envelopes},
            {envelopes[0]["record_id"]},
        )
        self.assertEqual(
            {item["chunk_index"] for item in envelopes}, set(range(len(envelopes)))
        )
        self.assertEqual({item["chunks"] for item in envelopes}, {len(envelopes)})
        encoded_payload = "".join(
            item["payload"]
            for item in sorted(envelopes, key=lambda item: item["chunk_index"])
        )
        packed = base64.b64decode(encoded_payload, validate=True)
        self.assertEqual(len(packed), envelopes[0]["packed_bytes"])
        self.assertEqual(
            hashlib.sha256(packed).hexdigest(), envelopes[0]["packed_sha256"]
        )
        raw = zlib.decompress(packed)
        self.assertEqual(len(raw), envelopes[0]["raw_bytes"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), envelopes[0]["sha256"])
        return json.loads(raw)

    def test_small_probe_result_also_uses_one_atomic_write(self):
        payload = {"status": "complete"}
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf",
                return_value=4096,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write",
                side_effect=lambda _fd, value: len(value),
            ) as write,
        ):
            _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", payload)

        write.assert_called_once()
        self.assertEqual(
            write.call_args.args[1],
            b'EAGLE_PP_SENDER_PROBE_RESULT {"status":"complete"}\n',
        )

    def test_large_probe_result_uses_one_checksum_bound_atomic_write(self):
        payload = {"stages": {"large": "x" * 16000}}
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf",
                return_value=4096,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write",
                side_effect=lambda _fd, value: len(value),
            ) as write,
        ):
            _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", payload)

        write.assert_called_once()
        fd, line = write.call_args.args
        self.assertEqual(fd, 2)
        self.assertLessEqual(len(line), 4096)
        marker, encoded = line.rstrip(b"\n").split(b" ", 1)
        self.assertEqual(marker, b"EAGLE_PP_SENDER_PROBE_RESULT")
        envelope = json.loads(encoded)
        raw = zlib.decompress(base64.b64decode(envelope["payload"], validate=True))
        self.assertEqual(envelope["__eagle_probe_encoding__"], "zlib+base64")
        self.assertEqual(envelope["raw_bytes"], len(raw))
        self.assertEqual(envelope["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(json.loads(raw), payload)

    def test_incompressible_probe_result_uses_checksum_bound_atomic_chunks(self):
        payload = {
            "hashes": [hashlib.sha256(str(i).encode()).hexdigest() for i in range(200)]
        }
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf",
                return_value=4096,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write",
                side_effect=lambda _fd, value: len(value),
            ) as write,
        ):
            _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", payload)
        lines = [call.args[1] for call in write.call_args_list]
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(line) <= 4096 for line in lines))
        self.assertEqual(self.decode_atomic_probe_records(lines), payload)

    def test_probe_result_fails_when_chunk_metadata_cannot_fit(self):
        payload = {
            "hashes": [hashlib.sha256(str(i).encode()).hexdigest() for i in range(8)]
        }
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf",
                return_value=128,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write"
            ) as write,
            self.assertRaisesRegex(ValueError, "chunk metadata exceeds"),
        ):
            _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", payload)
        write.assert_not_called()

    def test_probe_result_fails_before_write_when_raw_record_exceeds_safety_limit(self):
        payload = {"value": "x" * ((1 << 20) + 1)}
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat"
            ) as fstat,
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf"
            ) as fpathconf,
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write"
            ) as write,
            self.assertRaisesRegex(ValueError, "raw record.*safety limit"),
        ):
            _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", payload)
        fstat.assert_not_called()
        fpathconf.assert_not_called()
        write.assert_not_called()

    def test_probe_result_fails_before_write_when_packed_record_exceeds_safety_limit(
        self,
    ):
        payload = {"value": "x" * 4096}
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf",
                return_value=4096,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.zlib.compress",
                return_value=b"x" * ((1 << 20) + 1),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write"
            ) as write,
            self.assertRaisesRegex(ValueError, "packed record.*safety limit"),
        ):
            _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", payload)
        write.assert_not_called()

    def test_probe_result_fails_when_stderr_is_not_a_pipe(self):
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFREG),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf",
                return_value=4096,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write"
            ) as write,
            self.assertRaisesRegex(RuntimeError, "cannot establish atomic.*pipe"),
        ):
            _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", {"status": "complete"})
        write.assert_not_called()

    def test_probe_result_fails_when_pipe_limit_is_unavailable(self):
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf",
                side_effect=OSError(22, "invalid argument"),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write"
            ) as write,
            self.assertRaisesRegex(RuntimeError, "cannot establish atomic.*limit"),
        ):
            _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", {"status": "complete"})
        write.assert_not_called()

    def test_probe_result_short_write_fails_closed(self):
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFIFO),
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.fpathconf",
                return_value=4096,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.os.write",
                side_effect=lambda _fd, value: len(value) - 1,
            ) as write,
            self.assertRaisesRegex(RuntimeError, "short atomic.*write"),
        ):
            _emit_json_record("EAGLE_PP_SENDER_PROBE_RESULT", {"status": "complete"})
        write.assert_called_once()

    def test_pp_target_forward_row_domain_uses_communicator_layout(self):
        self.assertEqual(
            _pp_target_forward_row_domain(ScatterMode.SCATTERED),
            "pp_scattered_target_verify_tree_node",
        )
        self.assertEqual(
            _pp_target_forward_row_domain(ScatterMode.TP_ATTN_FULL),
            "pp_attn_group_target_verify_tree_node",
        )
        self.assertEqual(
            _pp_target_forward_row_domain(ScatterMode.FULL),
            "pp_full_target_verify_tree_node",
        )
        with self.assertRaisesRegex(ValueError, "MOE_FULL"):
            _pp_target_forward_row_domain(ScatterMode.MOE_FULL)

    @staticmethod
    def probe(expected_rid="probe-rid", *, require_pp_input=False):
        return EagleNumericalProbe(
            expected_rid,
            capture_id="capture-test-generation" if expected_rid else None,
            pod_name="probe-pod" if expected_rid else None,
            pod_uid="probe-pod-uid" if expected_rid else None,
            require_target_verify_pp_input=require_pp_input,
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
        if "target_verify_pp_input" in probe._required_stage_set:
            probe.record_target_verify_pp_input(
                pp_proxy_tensors={
                    "hidden_states": hidden + 10,
                    "residual": hidden + 20,
                    "topk_indices": torch.arange(rows * 2).reshape(rows, 2),
                    "__msg_type__": "proxy",
                },
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

    def test_pp_sender_probe_records_exact_rid_once(self):
        probe = EaglePPSenderProbe(
            "probe-rid",
            capture_id="capture-test-generation",
            pod_name="probe-pod",
            pod_uid="probe-pod-uid",
        )
        batch = SimpleNamespace(reqs=[SimpleNamespace(rid="probe-rid")])
        hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        rank = SimpleNamespace(
            world_rank=5,
            pp_rank=0,
            pp_size=2,
            tp_rank=5,
            tp_size=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_dp_rank=5,
            attn_dp_size=8,
        )
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.get_parallel",
                return_value=rank,
            ),
            self.capture_atomic_probe_records() as records,
        ):
            self.assertTrue(
                probe.begin_target_verify_pp_output(
                    pp_proxy_tensors={
                        "hidden_states": hidden,
                        "residual": hidden + 1,
                        "__msg_type__": "proxy",
                    },
                    target_world_rank=13,
                    require_attn_tp_allgather=False,
                )
            )
            probe.finalize_target_verify_pp_output()
            self.assertFalse(
                probe.begin_target_verify_pp_output(
                    pp_proxy_tensors={
                        "hidden_states": hidden + 2,
                        "residual": hidden + 3,
                    },
                    target_world_rank=13,
                    require_attn_tp_allgather=False,
                )
            )

        results = [self.decode_atomic_probe_record(line) for line in records]
        self.assertEqual(len(results), 1)
        payload = results[0]
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(
            payload["rank"],
            {
                "world": 5,
                "pp": 0,
                "pp_size": 2,
                "tp_rank": 5,
                "tp_size": 8,
                "attn_tp_rank": 0,
                "attn_tp_size": 1,
                "attn_dp": 5,
                "attn_dp_size": 8,
            },
        )
        stage = payload["stages"]["target_verify_pp_output"]
        self.assertEqual(stage["logical_rows"], 2)
        self.assertEqual(stage["row_domain"], "pp_rank_local_target_verify_tree_node")
        self.assertEqual(set(stage["tensors"]), {"hidden_states", "residual"})
        self.assertEqual(
            payload["transport"],
            {
                "target_world_rank": 13,
                "mode": "direct_rank_local",
                "require_attn_tp_allgather": False,
            },
        )
        self.assertFalse(probe.can_probe)

    def test_pp_target_forward_observer_uses_fixed_slots(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0, 3),
            max_rows=4,
            hidden_size=3,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        hidden = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)

        observer.capture(
            layer_id=0,
            boundary="attn_input",
            hidden_states=hidden,
            residual=hidden + 10,
            hidden_row_domain="pp_attn_group_target_verify_tree_node",
            residual_row_domain="pp_attn_group_target_verify_tree_node",
        )
        observer.capture(
            layer_id=1,
            boundary="attn_input",
            hidden_states=hidden + 20,
            residual=hidden + 30,
            hidden_row_domain="pp_attn_group_target_verify_tree_node",
            residual_row_domain="pp_attn_group_target_verify_tree_node",
        )
        snapshot = observer.snapshot_stages()

        self.assertEqual(
            observer.stage_names,
            (
                "target_verify_layer_00_attn_input",
                "target_verify_layer_00_attn_output",
                "target_verify_layer_00_mlp_input",
                "target_verify_layer_00_layer_return",
                "target_verify_layer_03_attn_input",
                "target_verify_layer_03_attn_output",
                "target_verify_layer_03_mlp_input",
                "target_verify_layer_03_layer_return",
            ),
        )
        self.assertTrue(
            torch.equal(
                snapshot["target_verify_layer_00_attn_input"]["tensors"][
                    "hidden_states"
                ][:2],
                hidden,
            )
        )
        self.assertTrue(
            torch.equal(
                snapshot["target_verify_layer_00_attn_input"]["tensors"]["residual"][
                    :2
                ],
                hidden + 10,
            )
        )
        metadata = snapshot["target_verify_layer_00_attn_input"]["tensor_metadata"]
        self.assertEqual(metadata["hidden_states"]["logical_rows"].item(), 2)
        self.assertEqual(metadata["residual"]["logical_rows"].item(), 2)
        self.assertEqual(
            metadata["hidden_states"]["row_domain"],
            "pp_attn_group_target_verify_tree_node",
        )
        uncaptured = snapshot["target_verify_layer_03_attn_input"]["tensor_metadata"]
        self.assertEqual(uncaptured["hidden_states"]["logical_rows"].item(), 0)
        self.assertEqual(uncaptured["residual"]["logical_rows"].item(), 0)

    def test_pp_target_forward_observer_rejects_shape_or_capacity_drift(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0,),
            max_rows=2,
            hidden_size=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            observer.capture(
                layer_id=0,
                boundary="attn_input",
                hidden_states=torch.ones((1, 4)),
                residual=torch.ones((1, 4)),
                hidden_row_domain="pp_attn_group_target_verify_tree_node",
                residual_row_domain="pp_attn_group_target_verify_tree_node",
            )
        with self.assertRaisesRegex(ValueError, "exceed fixed capacity"):
            observer.capture(
                layer_id=0,
                boundary="attn_input",
                hidden_states=torch.ones((3, 3)),
                residual=torch.ones((3, 3)),
                hidden_row_domain="pp_attn_group_target_verify_tree_node",
                residual_row_domain="pp_attn_group_target_verify_tree_node",
            )
        with self.assertRaisesRegex(ValueError, "invalid residual rows"):
            observer.capture(
                layer_id=0,
                boundary="attn_input",
                hidden_states=torch.ones((2, 3)),
                residual=torch.ones((0, 3)),
                hidden_row_domain="pp_attn_group_target_verify_tree_node",
                residual_row_domain="pp_attn_group_target_verify_tree_node",
            )

    def test_pp_target_forward_observer_tracks_mixed_communicator_domains(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(1,),
            max_rows=32,
            hidden_size=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        hidden = torch.arange(96, dtype=torch.float32).reshape(32, 3)
        residual = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        observer.capture(
            layer_id=1,
            boundary="attn_input",
            hidden_states=hidden,
            residual=residual,
            hidden_row_domain="pp_full_target_verify_tree_node",
            residual_row_domain="pp_attn_group_target_verify_tree_node",
        )
        snapshot = observer.snapshot_stages()["target_verify_layer_01_attn_input"]
        metadata = snapshot["tensor_metadata"]
        self.assertEqual(metadata["hidden_states"]["logical_rows"].item(), 32)
        self.assertEqual(
            metadata["hidden_states"]["row_domain"],
            "pp_full_target_verify_tree_node",
        )
        self.assertEqual(metadata["residual"]["logical_rows"].item(), 4)
        self.assertEqual(
            metadata["residual"]["row_domain"],
            "pp_attn_group_target_verify_tree_node",
        )
        self.assertTrue(torch.equal(snapshot["tensors"]["hidden_states"], hidden))
        self.assertTrue(torch.equal(snapshot["tensors"]["residual"][:4], residual))

        stage = EaglePPSenderProbe(
            "probe-rid",
            capture_id="capture-test-generation",
            pod_name="probe-pod",
            pod_uid="probe-pod-uid",
        )._finalize_stage_snapshot(
            {
                "tensor_metadata": metadata,
                "tensors": snapshot["tensors"],
            }
        )
        self.assertNotIn("logical_rows", stage)
        self.assertNotIn("row_domain", stage)
        self.assertEqual(stage["tensors"]["hidden_states"]["shape"], [32, 3])
        self.assertEqual(stage["tensors"]["hidden_states"]["logical_rows"], 32)
        self.assertEqual(stage["tensors"]["residual"]["shape"], [4, 3])
        self.assertEqual(stage["tensors"]["residual"]["logical_rows"], 4)

    def test_pp_target_forward_observer_captures_attention_internal_cuts(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0,),
            max_rows=4,
            hidden_size=3,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        observer.install_attention_boundaries(
            layer_id=0,
            raw_output_width=6,
            v_projection_width=4,
            row_domain="pp_attn_group_target_verify_tree_node",
            num_q_heads=2,
            padded_num_q_heads=4,
            q_nope_head_dim=2,
            q_rope_head_dim=1,
            topk_width=3,
            max_scheduler_rows=2,
        )
        raw = torch.arange(12, dtype=torch.bfloat16).reshape(2, 2, 3)
        v_projection = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)

        observer.capture_attention(
            layer_id=0, boundary="flashmla_raw_output", output=raw
        )
        observer.capture_attention(
            layer_id=0, boundary="v_projection_output", output=v_projection
        )
        snapshot = observer.snapshot_stages()

        raw_stage = snapshot["target_verify_layer_00_flashmla_raw_output"]
        self.assertEqual(
            raw_stage["tensor_metadata"]["output"]["logical_rows"].item(), 2
        )
        self.assertEqual(
            raw_stage["tensor_metadata"]["output"]["row_domain"],
            "pp_attn_group_target_verify_tree_node",
        )
        self.assertTrue(
            torch.equal(raw_stage["tensors"]["output"][:2], raw.reshape(2, 6))
        )
        self.assertTrue(
            torch.equal(
                snapshot["target_verify_layer_00_v_projection_output"]["tensors"][
                    "output"
                ][:2],
                v_projection,
            )
        )

    def test_pp_target_forward_observer_rejects_attention_shape_drift(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0,),
            max_rows=2,
            hidden_size=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        observer.install_attention_boundaries(
            layer_id=0,
            raw_output_width=6,
            v_projection_width=4,
            row_domain="pp_attn_group_target_verify_tree_node",
            num_q_heads=2,
            padded_num_q_heads=4,
            q_nope_head_dim=2,
            q_rope_head_dim=1,
            topk_width=3,
            max_scheduler_rows=2,
        )
        with self.assertRaisesRegex(ValueError, "width 8 does not match"):
            observer.capture_attention(
                layer_id=0,
                boundary="flashmla_raw_output",
                output=torch.ones((2, 2, 4)),
            )
        with self.assertRaisesRegex(ValueError, "outside fixed capacity"):
            observer.capture_attention(
                layer_id=0,
                boundary="v_projection_output",
                output=torch.ones((3, 4)),
            )

    def test_pp_target_forward_observer_captures_flashmla_inputs(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0,),
            max_rows=4,
            hidden_size=3,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        observer.install_attention_boundaries(
            layer_id=0,
            raw_output_width=6,
            v_projection_width=4,
            row_domain="pp_attn_group_target_verify_tree_node",
            num_q_heads=2,
            padded_num_q_heads=4,
            q_nope_head_dim=2,
            q_rope_head_dim=1,
            topk_width=3,
            max_scheduler_rows=2,
        )
        q_nope = torch.arange(8, dtype=torch.bfloat16).reshape(2, 2, 2)
        q_rope = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2, 1)
        q_input = torch.arange(24, dtype=torch.bfloat16).reshape(2, 1, 4, 3)
        topk = torch.tensor([[40, 41, 42], [80, 81, 82]], dtype=torch.int32)
        indices = topk.unsqueeze(1)
        logical_topk = observer.logical_topk_kernel_output(
            layer_id=0, rows=2, dtype=torch.int32, device=torch.device("cpu")
        )
        logical_topk.copy_(torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int32))
        logical_topk_input = observer.logical_topk_flashmla_input(
            layer_id=0, rows=2, dtype=torch.int32, device=torch.device("cpu")
        )
        self.assertEqual(logical_topk.data_ptr(), logical_topk_input.data_ptr())
        cache_seqlens = torch.tensor([3, 3], dtype=torch.int32)
        num_splits = torch.tensor([0, 1, 2], dtype=torch.int32)
        scheduler = torch.arange(16, dtype=torch.int32).reshape(2, 8)
        pointers_before = {
            name: tensor.data_ptr()
            for name, tensor in observer._flashmla_input_buffers[0].items()
        }

        observer.capture_flashmla_inputs(
            layer_id=0,
            q_nope=q_nope,
            q_rope=q_rope,
            q_input=q_input,
            topk_indices=topk,
            indices=indices,
            logical_topk_indices=logical_topk_input,
            cache_seqlens=cache_seqlens,
            num_splits=num_splits,
            tile_scheduler_metadata=scheduler,
        )
        stage = observer.snapshot_stages()["target_verify_layer_00_flashmla_inputs"]
        self.assertEqual(
            set(stage["tensors"]),
            {
                "q_nope",
                "q_rope",
                "q_input",
                "topk_indices",
                "indices",
                "logical_topk_indices",
                "cache_seqlens",
                "num_splits",
                "tile_scheduler_metadata",
            },
        )
        self.assertTrue(torch.equal(stage["tensors"]["q_input"][:2], q_input))
        self.assertTrue(torch.equal(stage["tensors"]["indices"][:2], indices))
        self.assertTrue(
            torch.equal(stage["tensors"]["logical_topk_indices"][:2], logical_topk)
        )
        self.assertEqual(stage["tensor_metadata"]["q_input"]["logical_rows"].item(), 2)
        self.assertEqual(
            stage["tensor_metadata"]["num_splits"]["logical_rows"].item(), 3
        )
        self.assertEqual(
            stage["tensor_metadata"]["num_splits"]["row_domain"],
            "flashmla_query_split_indptr",
        )
        self.assertEqual(
            stage["tensor_metadata"]["tile_scheduler_metadata"]["row_domain"],
            "flashmla_scheduler_partition",
        )
        self.assertEqual(
            pointers_before,
            {
                name: tensor.data_ptr()
                for name, tensor in observer._flashmla_input_buffers[0].items()
            },
        )

    def test_pp_target_forward_observer_keeps_attention_layers_independent(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0, 1),
            max_rows=2,
            hidden_size=3,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        for layer_id in (0, 1):
            observer.install_attention_boundaries(
                layer_id=layer_id,
                raw_output_width=6,
                v_projection_width=4,
                row_domain="pp_attn_group_target_verify_tree_node",
                num_q_heads=2,
                padded_num_q_heads=4,
                q_nope_head_dim=2,
                q_rope_head_dim=1,
                topk_width=3,
                max_scheduler_rows=2,
            )
            logical_topk = observer.logical_topk_kernel_output(
                layer_id=layer_id,
                rows=1,
                dtype=torch.int32,
                device=torch.device("cpu"),
            )
            logical_topk.copy_(torch.tensor([[layer_id + 2, layer_id + 1, layer_id]]))
            observer.capture_flashmla_inputs(
                layer_id=layer_id,
                q_nope=torch.full((1, 2, 2), layer_id, dtype=torch.bfloat16),
                q_rope=torch.full((1, 2, 1), layer_id, dtype=torch.bfloat16),
                q_input=torch.full((1, 1, 4, 3), layer_id, dtype=torch.bfloat16),
                topk_indices=torch.full((1, 3), layer_id, dtype=torch.int32),
                indices=torch.full((1, 1, 3), layer_id, dtype=torch.int32),
                logical_topk_indices=observer.logical_topk_flashmla_input(
                    layer_id=layer_id,
                    rows=1,
                    dtype=torch.int32,
                    device=torch.device("cpu"),
                ),
                cache_seqlens=torch.full((1,), layer_id, dtype=torch.int32),
                num_splits=torch.full((2,), layer_id, dtype=torch.int32),
                tile_scheduler_metadata=torch.full((2, 8), layer_id, dtype=torch.int32),
            )
            observer.capture_attention(
                layer_id=layer_id,
                boundary="flashmla_raw_output",
                output=torch.full((1, 2, 3), layer_id, dtype=torch.bfloat16),
            )

        snapshot = observer.snapshot_stages()
        for layer_id in (0, 1):
            stage = snapshot[f"target_verify_layer_{layer_id:02d}_flashmla_inputs"]
            self.assertTrue(
                torch.equal(
                    stage["tensors"]["logical_topk_indices"][:1],
                    torch.tensor([[layer_id + 2, layer_id + 1, layer_id]]),
                )
            )
            self.assertEqual(
                stage["tensor_metadata"]["q_input"]["logical_rows"].item(), 1
            )
            raw = snapshot[f"target_verify_layer_{layer_id:02d}_flashmla_raw_output"]
            self.assertTrue(
                torch.equal(
                    raw["tensors"]["output"][:1],
                    torch.full((1, 6), layer_id, dtype=torch.bfloat16),
                )
            )

    def test_pp_target_forward_observer_rejects_logical_topk_output_drift(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0,),
            max_rows=4,
            hidden_size=3,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(ValueError, "uninstalled layer"):
            observer.logical_topk_kernel_output(
                layer_id=0, rows=1, dtype=torch.int32, device=torch.device("cpu")
            )

        observer.install_attention_boundaries(
            layer_id=0,
            raw_output_width=6,
            v_projection_width=4,
            row_domain="pp_attn_group_target_verify_tree_node",
            num_q_heads=2,
            padded_num_q_heads=4,
            q_nope_head_dim=2,
            q_rope_head_dim=1,
            topk_width=3,
            max_scheduler_rows=2,
        )
        invalid_output_cases = (
            ("uninstalled layer", dict(layer_id=1, rows=2, dtype=torch.int32)),
            ("outside fixed capacity", dict(layer_id=0, rows=0, dtype=torch.int32)),
            ("outside fixed capacity", dict(layer_id=0, rows=5, dtype=torch.int32)),
            ("identity changed", dict(layer_id=0, rows=2, dtype=torch.int64)),
        )
        for message, kwargs in invalid_output_cases:
            with self.subTest(message=message, kwargs=kwargs):
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    observer.logical_topk_kernel_output(
                        **kwargs, device=torch.device("cpu")
                    )

        output = observer.logical_topk_kernel_output(
            layer_id=0, rows=2, dtype=torch.int32, device=torch.device("cpu")
        )
        self.assertEqual(output.shape, (2, 3))
        with self.assertRaisesRegex(RuntimeError, "producer/consumer row mismatch"):
            observer.logical_topk_flashmla_input(
                layer_id=0, rows=3, dtype=torch.int32, device=torch.device("cpu")
            )
        consumed = observer.logical_topk_flashmla_input(
            layer_id=0, rows=2, dtype=torch.int32, device=torch.device("cpu")
        )
        self.assertEqual(output.data_ptr(), consumed.data_ptr())

    def test_dsa_topk_backend_rejects_raw_output_outside_fused_v2_paged(self):
        logits = torch.zeros((2, 4), dtype=torch.float32)
        lengths = torch.ones((2,), dtype=torch.int32)
        raw = torch.empty((2, 2), dtype=torch.int32)
        cases = (
            (DSATopKBackend.SGL_KERNEL, False, True),
            (DSATopKBackend.SGL_KERNEL, True, False),
            (DSATopKBackend.TORCH, True, True),
        )
        for backend, fuse_topk, use_v2 in cases:
            with self.subTest(
                backend=backend.value, fuse_topk=fuse_topk, use_v2=use_v2
            ):
                with (
                    envs.SGLANG_DSA_FUSE_TOPK.override(fuse_topk),
                    envs.SGLANG_OPT_USE_TOPK_V2.override(use_v2),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "raw TopK output requires fused DeepSeek-V4 v2 PAGED dispatch",
                    ),
                ):
                    backend.topk_transform(
                        logits=logits,
                        lengths=lengths,
                        topk=2,
                        topk_transform_method=TopkTransformMethod.PAGED,
                        attn_metadata=SimpleNamespace(
                            real_page_table=torch.zeros((2, 1), dtype=torch.int32)
                        ),
                        force_unfused_topk=not fuse_topk,
                        out_raw_indices=raw,
                    )

    def test_pp_target_forward_observer_rejects_flashmla_input_drift_atomically(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0,),
            max_rows=4,
            hidden_size=3,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        observer.install_attention_boundaries(
            layer_id=0,
            raw_output_width=6,
            v_projection_width=4,
            row_domain="pp_attn_group_target_verify_tree_node",
            num_q_heads=2,
            padded_num_q_heads=4,
            q_nope_head_dim=2,
            q_rope_head_dim=1,
            topk_width=3,
            max_scheduler_rows=2,
        )
        valid = {
            "q_nope": torch.zeros((2, 2, 2), dtype=torch.bfloat16),
            "q_rope": torch.zeros((2, 2, 1), dtype=torch.bfloat16),
            "q_input": torch.zeros((2, 1, 4, 3), dtype=torch.bfloat16),
            "topk_indices": torch.zeros((2, 3), dtype=torch.int32),
            "indices": torch.zeros((2, 1, 3), dtype=torch.int32),
            "logical_topk_indices": torch.zeros((2, 3), dtype=torch.int32),
            "cache_seqlens": torch.zeros((2,), dtype=torch.int32),
            "num_splits": torch.zeros((3,), dtype=torch.int32),
            "tile_scheduler_metadata": torch.zeros((2, 8), dtype=torch.int32),
        }
        invalid_cases = {
            "scheduler capacity": (
                "tile_scheduler_metadata",
                torch.zeros((3, 8), dtype=torch.int32),
            ),
            "scheduler width": (
                "tile_scheduler_metadata",
                torch.zeros((2, 7), dtype=torch.int32),
            ),
            "length dtype": (
                "cache_seqlens",
                torch.zeros((2,), dtype=torch.int64),
            ),
        }
        for label, (name, value) in invalid_cases.items():
            with self.subTest(label=label):
                inputs = dict(valid)
                inputs[name] = value
                with self.assertRaisesRegex(ValueError, rf"\.{name} "):
                    observer.capture_flashmla_inputs(layer_id=0, **inputs)
                self.assertTrue(
                    all(
                        count.item() == 0
                        for count in observer._flashmla_input_row_counts[0].values()
                    )
                )

    def test_pp_target_forward_observer_tracks_replayed_bucket_rows(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0,),
            max_rows=4,
            hidden_size=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        stage = "target_verify_layer_00_layer_return"
        observer.capture(
            layer_id=0,
            boundary="layer_return",
            hidden_states=torch.full((4, 3), 4.0),
            residual=torch.full((4, 3), 5.0),
            hidden_row_domain="pp_attn_group_target_verify_tree_node",
            residual_row_domain="pp_attn_group_target_verify_tree_node",
        )
        observer.capture(
            layer_id=0,
            boundary="layer_return",
            hidden_states=torch.full((2, 3), 2.0),
            residual=torch.full((2, 3), 3.0),
            hidden_row_domain="pp_attn_group_target_verify_tree_node",
            residual_row_domain="pp_attn_group_target_verify_tree_node",
        )

        snapshot = observer.snapshot_stages()[stage]
        self.assertEqual(
            snapshot["tensor_metadata"]["hidden_states"]["logical_rows"].item(),
            2,
        )
        self.assertEqual(
            snapshot["tensor_metadata"]["residual"]["logical_rows"].item(),
            2,
        )
        self.assertTrue(
            torch.equal(
                snapshot["tensors"]["hidden_states"][:2], torch.full((2, 3), 2.0)
            )
        )
        self.assertTrue(
            torch.equal(snapshot["tensors"]["residual"][:2], torch.full((2, 3), 3.0))
        )

    def test_pp_target_forward_observer_rejects_cross_variant_domain_drift(self):
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0,),
            max_rows=4,
            hidden_size=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        tensor = torch.ones((2, 3))
        observer.capture(
            layer_id=0,
            boundary="attn_input",
            hidden_states=tensor,
            residual=tensor,
            hidden_row_domain="pp_attn_group_target_verify_tree_node",
            residual_row_domain="pp_attn_group_target_verify_tree_node",
        )
        with self.assertRaisesRegex(ValueError, "row-domain drift"):
            observer.capture(
                layer_id=0,
                boundary="attn_input",
                hidden_states=tensor,
                residual=tensor,
                hidden_row_domain="pp_full_target_verify_tree_node",
                residual_row_domain="pp_attn_group_target_verify_tree_node",
            )

    def test_pp_sender_probe_installs_expected_pp0_layer_anchors(self):
        probe = EaglePPSenderProbe(
            "probe-rid",
            capture_id="capture-test-generation",
            pod_name="probe-pod",
            pod_uid="probe-pod-uid",
        )
        layers = [SimpleNamespace() for _ in range(78)]
        for layer_id in (0, 1):
            layers[layer_id].self_attn = SimpleNamespace(
                num_local_heads=2,
                kv_lora_rank=8,
                qk_rope_head_dim=2,
                v_head_dim=4,
                indexer=SimpleNamespace(),
                attn_mqa=SimpleNamespace(),
            )
        model = type(
            "GlmMoeDsaForCausalLM",
            (),
            {
                "model": SimpleNamespace(layers=layers, start_layer=0, end_layer=40),
                "config": SimpleNamespace(hidden_size=16, index_topk=3),
            },
        )()

        probe.install_target_forward_observer(
            model=model,
            max_rows=32,
            dtype=torch.bfloat16,
            # ModelRunner.device is a string in the serving runtime. Keep this
            # contract covered instead of relying only on torch.device fixtures.
            device="cpu",
        )

        observer = probe.target_forward_observer
        self.assertIsNotNone(observer)
        self.assertEqual(observer.layer_ids, (0, 1, 10, 20, 39))
        for layer_id in observer.layer_ids:
            self.assertIs(layers[layer_id].target_forward_probe, observer)
        self.assertIs(layers[0].self_attn.target_forward_probe, observer)
        self.assertIs(layers[0].self_attn.indexer.target_forward_probe, observer)
        self.assertIs(layers[0].self_attn.attn_mqa.target_forward_probe, observer)
        self.assertIs(layers[1].self_attn.target_forward_probe, observer)
        self.assertIs(layers[1].self_attn.indexer.target_forward_probe, observer)
        self.assertIs(layers[1].self_attn.attn_mqa.target_forward_probe, observer)
        self.assertEqual(
            set(observer._attention_buffers),
            {
                "target_verify_layer_00_flashmla_raw_output",
                "target_verify_layer_00_v_projection_output",
                "target_verify_layer_01_flashmla_raw_output",
                "target_verify_layer_01_v_projection_output",
            },
        )
        self.assertEqual(
            set(observer._flashmla_input_buffers),
            {0, 1},
        )
        self.assertEqual(
            set(observer._flashmla_input_buffers[1]),
            {
                "q_nope",
                "q_rope",
                "q_input",
                "topk_indices",
                "indices",
                "logical_topk_indices",
                "cache_seqlens",
                "num_splits",
                "tile_scheduler_metadata",
            },
        )
        self.assertFalse(
            any(
                hasattr(layer, "target_forward_probe")
                for index, layer in enumerate(layers)
                if index not in observer.layer_ids
            )
        )

    def test_pp_sender_probe_includes_target_forward_sidecar_without_proxy_mutation(
        self,
    ):
        probe = EaglePPSenderProbe(
            "probe-rid",
            capture_id="capture-test-generation",
            pod_name="probe-pod",
            pod_uid="probe-pod-uid",
        )
        observer = _PPTargetForwardDeviceObserver(
            layer_ids=(0,),
            max_rows=2,
            hidden_size=3,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        hidden = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        for boundary, offset in zip(
            ("attn_input", "attn_output", "mlp_input", "layer_return"),
            (10, 20, 30, 40),
        ):
            observer.capture(
                layer_id=0,
                boundary=boundary,
                hidden_states=hidden + offset,
                residual=hidden + offset + 1,
                hidden_row_domain="pp_attn_group_target_verify_tree_node",
                residual_row_domain="pp_attn_group_target_verify_tree_node",
            )
        probe._target_forward_observer = observer
        proxy = {"hidden_states": hidden, "residual": hidden + 1}
        original_keys = tuple(proxy)

        rank = SimpleNamespace(
            world_rank=0,
            pp_rank=0,
            pp_size=2,
            tp_rank=0,
            tp_size=8,
            attn_tp_rank=0,
            attn_tp_size=8,
            attn_dp_rank=0,
            attn_dp_size=1,
        )
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.get_parallel",
                return_value=rank,
            ),
            self.capture_atomic_probe_records() as records,
        ):
            self.assertTrue(
                probe.begin_target_verify_pp_output(
                    pp_proxy_tensors=proxy,
                    target_world_rank=8,
                    require_attn_tp_allgather=False,
                )
            )
            probe.finalize_target_verify_pp_output()

        self.assertEqual(len(records), 1)
        payload = self.decode_atomic_probe_record(records[0])
        self.assertEqual(tuple(proxy), original_keys)
        self.assertEqual(
            set(payload["stages"]),
            {
                "target_verify_pp_output",
                "target_verify_layer_00_attn_input",
                "target_verify_layer_00_attn_output",
                "target_verify_layer_00_mlp_input",
                "target_verify_layer_00_layer_return",
            },
        )
        self.assertEqual(
            set(payload["stages"]["target_verify_layer_00_mlp_input"]["tensors"]),
            {"hidden_states", "residual"},
        )
        self.assertEqual(
            payload["stages"]["target_verify_layer_00_mlp_input"]["tensors"][
                "hidden_states"
            ]["logical_rows"],
            2,
        )

    def test_pp_sender_probe_rejects_stage_rows_beyond_snapshot_capacity(self):
        probe = EaglePPSenderProbe(
            "probe-rid",
            capture_id="capture-test-generation",
            pod_name="probe-pod",
            pod_uid="probe-pod-uid",
        )
        with self.assertRaisesRegex(ValueError, "invalid target-forward metadata"):
            probe._finalize_stage_snapshot(
                {
                    "tensor_metadata": {
                        "hidden_states": {
                            "logical_rows": 3,
                            "row_domain": "pp_rank_local_target_verify_tree_node",
                        },
                        "residual": {
                            "logical_rows": 2,
                            "row_domain": "pp_rank_local_target_verify_tree_node",
                        },
                    },
                    "tensors": {
                        "hidden_states": torch.ones((2, 3)),
                        "residual": torch.ones((2, 3)),
                    },
                }
            )

    def test_pp_sender_probe_wrong_or_cobatched_rid_is_noop(self):
        probe = EaglePPSenderProbe(
            "probe-rid",
            capture_id="capture-test-generation",
            pod_name="probe-pod",
            pod_uid="probe-pod-uid",
        )
        tensor = torch.ones((1, 2))
        for rids in (["other"], ["probe-rid", "other"]):
            with self.subTest(rids=rids):
                self.assertFalse(
                    probe.matches_schedule_batch(
                        SimpleNamespace(reqs=[SimpleNamespace(rid=rid) for rid in rids])
                    )
                )
        self.assertTrue(probe.can_probe)

    def test_pp_sender_probe_missing_or_mismatched_tensors_fail_closed(self):
        batch = SimpleNamespace(reqs=[SimpleNamespace(rid="probe-rid")])
        cases = {
            "missing": {"hidden_states": torch.ones((1, 2))},
            "row_mismatch": {
                "hidden_states": torch.ones((1, 2)),
                "residual": torch.ones((2, 2)),
            },
        }
        for case, tensors in cases.items():
            with self.subTest(case=case):
                probe = EaglePPSenderProbe(
                    "probe-rid",
                    capture_id="capture-test-generation",
                    pod_name="probe-pod",
                    pod_uid="probe-pod-uid",
                )
                with self.capture_atomic_probe_records() as records:
                    self.assertFalse(
                        probe.begin_target_verify_pp_output(
                            pp_proxy_tensors=tensors,
                            target_world_rank=13,
                            require_attn_tp_allgather=False,
                        )
                    )
                self.assertEqual(len(records), 1)
                payload = self.decode_atomic_probe_record(records[0])
                self.assertEqual(payload["status"], "rejected")
                self.assertEqual(payload["stages"], {})
                self.assertFalse(probe.can_probe)

    def test_pp_sender_probe_cuda_snapshot_is_nonblocking_and_finalized_later(self):
        events = []
        stream = object()
        event = SimpleNamespace(
            record=mock.Mock(side_effect=lambda _stream: events.append("record")),
            synchronize=mock.Mock(side_effect=lambda: events.append("synchronize")),
        )

        def cuda_tensor(name):
            tensor = mock.MagicMock(spec=torch.Tensor)
            tensor.detach.return_value = tensor
            tensor.is_cuda = True
            tensor.shape = (2, 4)
            tensor.dtype = torch.bfloat16
            tensor.device = torch.device("cuda:0")
            tensor.record_stream.side_effect = lambda _stream: events.append(
                f"record_stream:{name}"
            )
            return tensor

        snapshots = []
        fingerprint_options = []

        def empty(*_args, **kwargs):
            self.assertEqual(kwargs["device"], "cpu")
            self.assertTrue(kwargs["pin_memory"])
            snapshot = mock.MagicMock(spec=torch.Tensor)
            snapshot.shape = _args[0]
            snapshot.copy_.side_effect = lambda _value, **copy_kwargs: events.append(
                ("copy", copy_kwargs["non_blocking"])
            )
            snapshots.append(snapshot)
            return snapshot

        probe = EaglePPSenderProbe(
            "probe-rid",
            capture_id="capture-test-generation",
            pod_name="probe-pod",
            pod_uid="probe-pod-uid",
        )
        rank = SimpleNamespace(
            world_rank=0,
            pp_rank=0,
            pp_size=2,
            tp_rank=0,
            tp_size=8,
            attn_tp_rank=0,
            attn_tp_size=8,
            attn_dp_rank=0,
            attn_dp_size=1,
        )
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.get_parallel",
                return_value=rank,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.torch.empty",
                side_effect=empty,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.torch.cuda.current_stream",
                return_value=stream,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.torch.cuda.Event",
                return_value=event,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe._tensor_fingerprint",
                side_effect=lambda *_args, **kwargs: (
                    fingerprint_options.append(kwargs),
                    events.append("fingerprint"),
                    {},
                )[2],
            ),
            self.capture_atomic_probe_records() as records,
        ):
            self.assertTrue(
                probe.begin_target_verify_pp_output(
                    pp_proxy_tensors={
                        "hidden_states": cuda_tensor("hidden_states"),
                        "residual": cuda_tensor("residual"),
                    },
                    target_world_rank=8,
                    require_attn_tp_allgather=False,
                )
            )
            self.assertNotIn("synchronize", events)
            self.assertNotIn("fingerprint", events)
            probe.finalize_target_verify_pp_output()

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(len(records), 1)
        self.assertEqual(
            self.decode_atomic_probe_record(records[0])["status"], "complete"
        )
        self.assertEqual(events.count(("copy", True)), 2)
        self.assertEqual(
            sum(bool(item.get("include_row_multiset")) for item in fingerprint_options),
            0,
        )
        self.assertLess(events.index("record"), events.index("synchronize"))
        self.assertLess(events.index("synchronize"), events.index("fingerprint"))

    def test_pp_sender_probe_cuda_queue_error_emits_rejection_and_raises(self):
        tensor = mock.MagicMock(spec=torch.Tensor)
        tensor.detach.return_value = tensor
        tensor.is_cuda = True
        tensor.shape = (1, 2)
        tensor.dtype = torch.bfloat16
        tensor.device = torch.device("cuda:0")
        probe = EaglePPSenderProbe(
            "probe-rid",
            capture_id="capture-test-generation",
            pod_name="probe-pod",
            pod_uid="probe-pod-uid",
        )
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe.torch.empty",
                side_effect=RuntimeError("pinned allocation failed"),
            ),
            self.capture_atomic_probe_records() as records,
            self.assertRaisesRegex(RuntimeError, "pinned allocation failed"),
        ):
            probe.begin_target_verify_pp_output(
                pp_proxy_tensors={"hidden_states": tensor, "residual": tensor},
                target_world_rank=8,
                require_attn_tp_allgather=False,
            )

        self.assertEqual(len(records), 1)
        payload = self.decode_atomic_probe_record(records[0])
        self.assertEqual(payload["status"], "rejected")
        self.assertIn("pinned allocation failed", payload["rejection"])
        self.assertFalse(probe.can_probe)

    def test_non_pipeline_probe_keeps_original_stage_contract(self):
        probe = self.probe()

        self.assertNotIn("target_verify_pp_input", probe._required_stage_set)
        self.assertEqual(probe._required_stages[0], "target_verify_input")
        self.assertEqual(probe._required_stages[1], "target_verify_output")

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
        probe = self.probe(require_pp_input=True)
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
                "target_verify_pp_input",
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
            [payload["ordinal"] for payload in stage_payloads], list(range(1, 13))
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
                "target_verify_pp_input",
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
            stages["target_verify_pp_input"]["tensors"]["hidden_states"]["shape"],
            [2, 4],
        )
        self.assertNotIn("topk_indices", stages["target_verify_pp_input"]["tensors"])
        self.assertNotIn("__msg_type__", stages["target_verify_pp_input"]["tensors"])
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
        self.assertEqual(stages["target_verify_pp_input"]["logical_rows"], 2)
        self.assertEqual(stages["target_verify_accept"]["logical_rows"], 1)
        self.assertEqual(stages["proposed_token"]["logical_rows"], 1)
        self.assertEqual(
            stages["target_verify_input"]["row_domain"],
            "target_verify_tree_node",
        )
        self.assertEqual(
            stages["target_verify_pp_input"]["row_domain"],
            "pp_rank_local_target_verify_tree_node",
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

    def test_pp_probe_requires_real_hidden_and_residual_inputs(self):
        rows = 2
        cases = {
            "missing_proxy": None,
            "missing_residual": {
                "hidden_states": torch.ones((rows, 4)),
            },
            "missing_hidden_states": {
                "residual": torch.ones((rows, 4)),
            },
        }
        for case, pp_proxy_tensors in cases.items():
            with self.subTest(case=case):
                probe = self.probe(require_pp_input=True)
                self.assertTrue(
                    probe.record_target_verify_input(
                        rids=["probe-rid"],
                        draft_token=torch.ones(rows, dtype=torch.int64),
                        positions=torch.ones(rows, dtype=torch.int64),
                        retrieve_index=torch.arange(rows).reshape(1, rows),
                        retrieve_next_token=torch.full(
                            (1, rows), -1, dtype=torch.int64
                        ),
                        retrieve_next_sibling=torch.full(
                            (1, rows), -1, dtype=torch.int64
                        ),
                        batch_size=1,
                        draft_token_num=rows,
                    )
                )
                probe.record_target_verify_pp_input(
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                self.assertFalse(probe.can_probe)

    def test_pp_probe_uses_rank_local_rows_and_rejects_pair_mismatch(self):
        probe = self.probe(require_pp_input=True)
        self.assertTrue(
            probe.record_target_verify_input(
                rids=["probe-rid"],
                draft_token=torch.ones(4, dtype=torch.int64),
                positions=torch.ones(4, dtype=torch.int64),
                retrieve_index=torch.arange(4).reshape(1, 4),
                retrieve_next_token=torch.full((1, 4), -1, dtype=torch.int64),
                retrieve_next_sibling=torch.full((1, 4), -1, dtype=torch.int64),
                batch_size=1,
                draft_token_num=4,
            )
        )
        probe.record_target_verify_pp_input(
            pp_proxy_tensors={
                "hidden_states": torch.ones((1, 4)),
                "residual": torch.ones((2, 4)),
                "__msg_type__": "proxy",
            },
        )
        self.assertFalse(probe.can_probe)
        self.assertIn("row mismatch", probe._rejection)

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

    def test_prefill_pd_handoff_records_and_seals_independently(self):
        probe = EaglePDHandoffProbe(
            "probe-rid",
            role="prefill",
            capture_id="capture-test-generation",
            pod_name="prefill-pod",
            pod_uid="prefill-pod-uid",
        )
        sampled = torch.tensor([8451], dtype=torch.int64)
        serialized = sampled.clone()
        wire = torch.tensor([8451], dtype=torch.int32)

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.record_prefill_pp_output(
                rids=["probe-rid"],
                next_token_ids=sampled,
                serialized_next_token_ids=serialized,
            )
            probe.record_prefill_metadata_write(
                rid="probe-rid", sampled_token=8451, wire_output_id=wire
            )

        stage_payloads = [
            json.loads(line.split("EAGLE_PD_HANDOFF_PROBE_STAGE ", 1)[1])
            for line in logs.output
            if "EAGLE_PD_HANDOFF_PROBE_STAGE " in line
        ]
        self.assertEqual(
            [payload["stage"] for payload in stage_payloads],
            ["pp_output", "metadata_write"],
        )
        self.assertEqual(
            [payload["ordinal"] for payload in stage_payloads],
            [1, 2],
        )
        self.assertEqual(
            stage_payloads[0]["fingerprints"]["next_token_ids"]["values"],
            [8451],
        )
        self.assertEqual(
            stage_payloads[1]["fingerprints"]["wire_output_id"]["values"],
            [8451],
        )
        result = json.loads(
            logs.output[-1].split("EAGLE_PD_HANDOFF_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["phase"], "pd_handoff")
        self.assertEqual(result["role"], "prefill")
        self.assertEqual(result["capture"], probe.capture)

    def test_decode_pd_handoff_records_wire_override_and_prebuilt_bonus(self):
        probe = EaglePDHandoffProbe(
            "probe-rid",
            role="decode",
            capture_id="capture-test-generation",
            pod_name="decode-pod",
            pod_uid="decode-pod-uid",
        )

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.record_decode_metadata_read(
                rid="probe-rid",
                wire_output_id=torch.tensor([8451], dtype=torch.int32),
                committed_output_id=71,
            )
            probe.record_decode_prebuilt_bonus(
                rids=["probe-rid"],
                committed_output_id=torch.tensor([71], dtype=torch.int64),
                bonus_tokens=torch.tensor([71], dtype=torch.int64),
            )

        result = json.loads(
            logs.output[-1].split("EAGLE_PD_HANDOFF_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["role"], "decode")
        self.assertTrue(result["seen"])
        self.assertEqual(
            result["stages"]["metadata_read"]["tensors"]["wire_output_id"]["values"],
            [8451],
        )
        self.assertEqual(
            result["stages"]["metadata_read"]["tensors"]["committed_output_id"][
                "values"
            ],
            [71],
        )

    def test_pd_handoff_wrong_order_fails_closed(self):
        probe = EaglePDHandoffProbe(
            "probe-rid",
            role="prefill",
            capture_id="capture-test-generation",
            pod_name="prefill-pod",
            pod_uid="prefill-pod-uid",
        )

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.record_prefill_metadata_write(
                rid="probe-rid",
                sampled_token=8451,
                wire_output_id=torch.tensor([8451], dtype=torch.int32),
            )

        result = json.loads(
            logs.output[-1].split("EAGLE_PD_HANDOFF_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["seen"])
        self.assertIn("out-of-order", result["rejection"])

    def test_pd_handoff_stage_error_uses_first_ordinal_and_seals(self):
        probe = EaglePDHandoffProbe(
            "probe-rid",
            role="prefill",
            capture_id="capture-test-generation",
            pod_name="prefill-pod",
            pod_uid="prefill-pod-uid",
        )

        with (
            mock.patch(
                "sglang.srt.speculative.eagle_numerical_probe._tensor_fingerprint",
                side_effect=ValueError("invalid fingerprint"),
            ),
            self.assertLogs(
                "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
            ) as logs,
        ):
            probe.record_prefill_pp_output(
                rids=["probe-rid"],
                next_token_ids=torch.tensor([8451]),
                serialized_next_token_ids=torch.tensor([8451]),
            )

        error = json.loads(
            next(
                line.split("EAGLE_PD_HANDOFF_PROBE_STAGE_ERROR ", 1)[1]
                for line in logs.output
                if "EAGLE_PD_HANDOFF_PROBE_STAGE_ERROR " in line
            )
        )
        self.assertEqual(error["ordinal"], 1)
        result = json.loads(
            logs.output[-1].split("EAGLE_PD_HANDOFF_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(result["status"], "rejected")
        self.assertIn("invalid fingerprint", result["rejection"])


if __name__ == "__main__":
    unittest.main()
