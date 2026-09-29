import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _make_dspark_server_args(
    *,
    model_path: str,
    hf_config: SimpleNamespace,
    is_fp4_experts: bool = False,
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args._model_config = SimpleNamespace(
        hf_config=hf_config,
        is_fp4_experts=is_fp4_experts,
    )
    return server_args


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(
            resolution_result(server_args, "speculative_draft_model_path"),
            _BUNDLED_MODEL_PATH,
        )
        self.assertEqual(
            resolution_result(server_args, "speculative_num_draft_tokens"), 6
        )

    def test_plain_target_without_draft_path_raises(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            resolution_result(server_args, "speculative_draft_model_path"),
            "deepseek-ai/some-other-dspark-draft",
        )


class TestDsparkDpAttentionMoeA2aGate(CustomTestCase):
    """Gate contract for DSpark + dp attention + MoE a2a backends."""

    def _dp_server_args(
        self,
        *,
        moe_a2a_backend: str,
        moe_runner_backend: str = "auto",
        is_fp4_experts: bool = False,
        flashinfer_mxfp4_moe_precision: str = "default",
    ) -> ServerArgs:
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH,
            hf_config=_bundled_hf_config(),
            is_fp4_experts=is_fp4_experts,
        )
        server_args.enable_dp_attention = True
        server_args.enable_dp_lm_head = True
        server_args.dp_size = 2
        server_args.tp_size = 2
        server_args.moe_a2a_backend = moe_a2a_backend
        server_args.moe_runner_backend = moe_runner_backend
        server_args.flashinfer_mxfp4_moe_precision = flashinfer_mxfp4_moe_precision
        return server_args

    @patch(
        "sglang.srt.arg_groups.speculative_hook.get_platform",
        return_value=SimpleNamespace(is_sm90=True),
    )
    def test_supported_a2a_backends_are_admitted(self, _):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(self._dp_server_args(moe_a2a_backend="megamoe"))
            _handle_dspark(
                self._dp_server_args(
                    moe_a2a_backend="deepep",
                    moe_runner_backend="deep_gemm",
                )
            )
            _handle_dspark(
                self._dp_server_args(
                    moe_a2a_backend="deepep",
                    moe_runner_backend="flashinfer_mxfp4",
                    flashinfer_mxfp4_moe_precision="fp8",
                    is_fp4_experts=True,
                )
            )

    @patch(
        "sglang.srt.arg_groups.speculative_hook.get_platform",
        return_value=SimpleNamespace(is_sm90=True),
    )
    def test_deepep_rejects_unsupported_runner(self, _):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            for runner, precision, is_fp4_experts in (
                ("triton", "fp8", True),
                ("flashinfer_mxfp4", "bf16", True),
                ("flashinfer_mxfp4", "fp8", False),
            ):
                with self.assertRaisesRegex(ValueError, "supported runner"):
                    _handle_dspark(
                        self._dp_server_args(
                            moe_a2a_backend="deepep",
                            moe_runner_backend=runner,
                            flashinfer_mxfp4_moe_precision=precision,
                            is_fp4_experts=is_fp4_experts,
                        )
                    )

    def test_a2a_backend_with_compact_verify_mode_raises(self):
        server_args = self._dp_server_args(moe_a2a_backend="megamoe")
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            with self.assertRaisesRegex(ValueError, "static"):
                _handle_dspark(server_args)


class TestDsparkReplicatedPPDraft(CustomTestCase):
    def _replicated_args(self, mode: str) -> ServerArgs:
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.pp_size = 2
        server_args.disaggregation_mode = mode
        server_args.speculative_dspark_pp_replicated_draft = True
        server_args.disable_cuda_graph = True
        server_args.pp_async_batch_depth = 0
        server_args.enable_dp_attention = False
        server_args.speculative_use_rejection_sampling = False
        server_args.disable_radix_cache = True
        server_args.attn_cp_size = 1
        server_args.enable_mixed_chunk = True
        return server_args

    def test_prefill_and_decode_are_admitted(self):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            for mode in ("prefill", "decode"):
                args = self._replicated_args(mode)
                _handle_dspark(args)
                self.assertFalse(resolution_result(args, "enable_mixed_chunk"))
                self.assertEqual(args.speculative_draft_scheduling_policy, "tail")

    def test_bubble_policy_requires_replicated_pp_draft(self):
        args = self._replicated_args("decode")
        args.speculative_draft_scheduling_policy = "bubble"
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(args)

        args.speculative_dspark_pp_replicated_draft = False
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            with self.assertRaisesRegex(ValueError, "replicated-draft"):
                _handle_dspark(args)

    def test_decode_cuda_graph_is_admitted(self):
        args = self._replicated_args("decode")
        args.disable_cuda_graph = False
        args.cuda_graph_max_bs_decode = 256
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(args)

    def test_dp4_tp4_builtin_moe_is_admitted(self):
        args = self._replicated_args("decode")
        args.enable_dp_attention = True
        args.enable_dp_lm_head = True
        args.dp_size = 4
        args.tp_size = 4
        args.moe_a2a_backend = "none"
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(args)

        args.dp_size = 2
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            with self.assertRaisesRegex(ValueError, "dp-size == --tp-size"):
                _handle_dspark(args)

        args.dp_size = 4
        args.moe_a2a_backend = "megamoe"
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            with self.assertRaisesRegex(ValueError, "built-in TP MoE"):
                _handle_dspark(args)

    @patch(
        "sglang.srt.arg_groups.speculative_hook.get_platform",
        return_value=SimpleNamespace(is_sm90=True),
    )
    def test_dp8_tp8_flashinfer_humming_deepep_is_admitted(self, _):
        args = self._replicated_args("decode")
        args.enable_dp_attention = True
        args.enable_dp_lm_head = True
        args.dp_size = 8
        args.tp_size = 8
        args.moe_a2a_backend = "deepep"
        args.moe_runner_backend = "flashinfer_mxfp4"
        args.flashinfer_mxfp4_moe_precision = "fp8"
        args.speculative_moe_a2a_backend = "deepep"
        args.speculative_moe_runner_backend = "flashinfer_mxfp4"
        args._model_config.is_fp4_experts = True

        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(args)

    @patch(
        "sglang.srt.arg_groups.speculative_hook.get_platform",
        return_value=SimpleNamespace(is_sm90=True),
    )
    def test_pp_flashinfer_humming_deepep_requires_matching_draft_a2a(self, _):
        args = self._replicated_args("decode")
        args.enable_dp_attention = True
        args.enable_dp_lm_head = True
        args.dp_size = 8
        args.tp_size = 8
        args.moe_a2a_backend = "deepep"
        args.moe_runner_backend = "flashinfer_mxfp4"
        args.flashinfer_mxfp4_moe_precision = "fp8"
        args.speculative_moe_a2a_backend = "none"
        args._model_config.is_fp4_experts = True

        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            with self.assertRaisesRegex(ValueError, "must match"):
                _handle_dspark(args)

    def test_prefill_requires_only_prefill_graph_disabled(self):
        args = self._replicated_args("prefill")
        args.disable_cuda_graph = False
        args.disable_prefill_cuda_graph = True
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(args)

        args.disable_prefill_cuda_graph = False
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            with self.assertRaisesRegex(ValueError, "prefill CUDA graph"):
                _handle_dspark(args)

    def test_non_pd_mode_is_rejected(self):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            with self.assertRaisesRegex(ValueError, "PD disaggregation"):
                _handle_dspark(self._replicated_args("null"))

    def test_unbundled_draft_is_rejected(self):
        args = self._replicated_args("decode")
        args.speculative_draft_model_path = "separate/draft"
        with self.assertRaisesRegex(ValueError, "bundled DeepSeek-V4"):
            _handle_dspark(args)

    def test_radix_cache_is_rejected(self):
        args = self._replicated_args("decode")
        args.disable_radix_cache = False
        with self.assertRaisesRegex(ValueError, "disable-radix-cache"):
            _handle_dspark(args)


if __name__ == "__main__":
    unittest.main()
