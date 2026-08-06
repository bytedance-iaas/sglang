"""CPU coverage for chunked-prefix Full prefill CUDA-graph state."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_executor.model_runner_components.cuda_graph_setup as graph_setup
import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as runner_module
import sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph as bcg_module
from sglang.kernels.ops.attention.dsv4 import gemm as dsv4_gemm
from sglang.srt.distributed import communication_op, parallel_state
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_prefill_graph,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeAttentionBackend:
    supports_full_cuda_graph_chunked_prefix = True

    def __init__(self):
        self.calls = []

    def prepare_full_cuda_graph_chunked_prefix(self, forward_batch, *, in_capture):
        self.calls.append((forward_batch, in_capture))


class _FakeKVIndexKernel:
    def __getitem__(self, grid):
        del grid

        def run(
            req_to_token,
            req_pool_indices,
            starts,
            seq_lens,
            cu_seq_lens,
            output,
            req_to_token_stride,
        ):
            del cu_seq_lens, req_to_token_stride
            cursor = 0
            for row in range(seq_lens.numel()):
                seq_len = int(seq_lens[row])
                start = int(starts[row])
                req = int(req_pool_indices[row])
                output[cursor : cursor + seq_len].copy_(
                    req_to_token[req, start : start + seq_len]
                )
                cursor += seq_len

        return run


class TestPrefillCudaGraphRunnerChunkedPrefix(CustomTestCase):
    def test_multinode_breakable_capture_prewarm_order(self):
        calls = []
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        runner.capture_num_tokens = [4, 2048, 128]
        runner._capture_chunked_prefix = False
        runner._prefix_capture_variants = []
        runner.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(nnodes=2),
        )
        batches = {num_tokens: object() for num_tokens in runner.capture_num_tokens}

        def prepare(num_tokens, *, prefix_num_chunks=0):
            calls.append(("prepare", num_tokens, prefix_num_chunks))
            return batches[num_tokens], object(), ShapeKey(size=num_tokens)

        runner._prepare_capture_shape = prepare
        runner._run_forward = lambda batch, num_tokens: calls.append(
            ("forward", batch, num_tokens)
        )

        class _ReplaySession:
            def __enter__(self):
                calls.append("replay_enter")

            def __exit__(self, *args):
                calls.append("replay_exit")

        runner.backend.replay_session = lambda: _ReplaySession()

        def prewarm_one(shape_key, forward_fn):
            calls.append(("prewarm", shape_key.size))
            forward_fn()

        runner.backend.prewarm_one = prewarm_one

        runner._prewarm_multinode_breakable_capture()

        self.assertEqual(
            calls,
            [
                "replay_enter",
                ("prepare", 2048, 0),
                ("prewarm", 2048),
                ("forward", batches[2048], 2048),
                "replay_exit",
            ],
        )

    def test_breakable_backend_prewarm_retains_last_output(self):
        calls = []
        backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        backend._device_module = SimpleNamespace(
            synchronize=lambda: calls.append("synchronize")
        )
        backend._tp_group = SimpleNamespace(barrier=lambda: calls.append("barrier"))
        backend._prewarmed_outputs = {}
        shape_key = ShapeKey(size=2048)

        backend.prewarm_one(shape_key, lambda: "output")

        self.assertEqual(backend._prewarmed_outputs[shape_key], "output")
        self.assertEqual(calls, ["synchronize", "barrier"])

    def test_collective_break_uses_capture_safe_small_moe_reduce(self):
        with patch.object(
            fused_moe, "is_batch_invariant_mode_enabled", return_value=False
        ):
            self.assertTrue(fused_moe._use_moe_sum_reduce_torch_compile(32))
            with communication_op.cuda_graph_collective_break():
                self.assertFalse(fused_moe._use_moe_sum_reduce_torch_compile(32))
            self.assertTrue(fused_moe._use_moe_sum_reduce_torch_compile(32))

    def test_multinode_collective_break_precompiles_small_moe_reductions(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        runner.backend._enable_collective_break = True
        runner.capture_num_tokens = [4, 16, 32, 48]
        runner.device = torch.device("cpu")
        runner.device_module = SimpleNamespace(synchronize=Mock())
        runner.model_runner = SimpleNamespace(dtype=torch.float32)
        runner.moe_layers = [
            SimpleNamespace(
                moe_runner_config=SimpleNamespace(
                    top_k=9,
                    hidden_size=8,
                    routed_scaling_factor=2.5,
                    no_combine=False,
                )
            ),
            None,
        ]

        target = (
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
            "moe_sum_reduce_torch_compile"
        )
        with patch(target) as compiled_reduce:
            runner._precompile_multinode_breakable_moe_reduce()

        self.assertEqual(compiled_reduce.call_count, 3)
        self.assertEqual(
            [call.args[0].shape for call in compiled_reduce.call_args_list],
            [torch.Size((4, 9, 8)), torch.Size((16, 9, 8)), torch.Size((32, 9, 8))],
        )
        self.assertTrue(
            all(call.args[2] == 2.5 for call in compiled_reduce.call_args_list)
        )
        runner.device_module.synchronize.assert_called_once_with()

    def test_moe_reduce_precompile_is_collective_break_only(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        runner.backend._enable_collective_break = False
        runner.capture_num_tokens = [4]
        runner.moe_layers = []

        with patch.object(torch, "empty") as empty:
            runner._precompile_multinode_breakable_moe_reduce()

        empty.assert_not_called()

    def test_collective_break_routes_moe_gate_gemm_through_bcg_break(self):
        hidden_states = object()
        weight = object()
        with (
            patch.object(dsv4_gemm, "linear_bf16_fp32", return_value="eager") as eager,
            patch.object(
                dsv4_gemm, "bcg_linear_bf16_fp32", return_value="break"
            ) as graph_break,
        ):
            self.assertEqual(
                dsv4_gemm.linear_bf16_fp32_moe_gate(hidden_states, weight), "eager"
            )
            with communication_op.cuda_graph_collective_break():
                self.assertEqual(
                    dsv4_gemm.linear_bf16_fp32_moe_gate(hidden_states, weight),
                    "break",
                )
            self.assertEqual(
                dsv4_gemm.linear_bf16_fp32_moe_gate(hidden_states, weight), "eager"
            )

        self.assertEqual(eager.call_count, 2)
        graph_break.assert_called_once_with(hidden_states, weight)

    def test_bcg_moe_gate_uses_deep_gemm_for_glm52_medium_token_shape(self):
        hidden_states = object()
        weight = object()
        with (
            patch.object(
                dsv4_gemm,
                "_can_use_bcg_deep_gemm_moe_gate",
                return_value=True,
            ),
            patch.object(
                dsv4_gemm,
                "_linear_bf16_fp32_deep_gemm",
                return_value="deep_gemm",
            ) as deep_gemm,
            patch.object(dsv4_gemm, "linear_bf16_fp32") as fallback,
        ):
            self.assertEqual(
                dsv4_gemm._linear_bf16_fp32_moe_gate_bcg(hidden_states, weight),
                "deep_gemm",
            )

        deep_gemm.assert_called_once_with(hidden_states, weight)
        fallback.assert_not_called()

    def test_bcg_deep_gemm_moe_gate_shape_is_exact_and_bounded(self):
        def tensor(shape):
            return SimpleNamespace(
                dim=lambda: 2,
                is_cuda=True,
                dtype=torch.bfloat16,
                is_contiguous=lambda: True,
                shape=shape,
            )

        weight = tensor((256, 6144))
        with patch("sglang.srt.layers.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM", True):
            self.assertTrue(
                dsv4_gemm._can_use_bcg_deep_gemm_moe_gate(tensor((28, 6144)), weight)
            )
            for num_tokens in (16, 33):
                self.assertFalse(
                    dsv4_gemm._can_use_bcg_deep_gemm_moe_gate(
                        tensor((num_tokens, 6144)), weight
                    )
                )
            self.assertFalse(
                dsv4_gemm._can_use_bcg_deep_gemm_moe_gate(
                    tensor((28, 7168)), tensor((256, 7168))
                )
            )
            self.assertFalse(
                dsv4_gemm._can_use_bcg_deep_gemm_moe_gate(
                    tensor((28, 6144)), tensor((257, 6144))
                )
            )

    def test_bcg_moe_gate_preserves_existing_fallback_dispatch(self):
        hidden_states = object()
        weight = object()
        with (
            patch.object(
                dsv4_gemm,
                "_can_use_bcg_deep_gemm_moe_gate",
                return_value=False,
            ),
            patch.object(
                dsv4_gemm, "linear_bf16_fp32", return_value="fallback"
            ) as fallback,
            patch.object(dsv4_gemm, "_linear_bf16_fp32_deep_gemm") as deep_gemm,
        ):
            self.assertEqual(
                dsv4_gemm._linear_bf16_fp32_moe_gate_bcg(hidden_states, weight),
                "fallback",
            )

        fallback.assert_called_once_with(hidden_states, weight)
        deep_gemm.assert_not_called()

    def test_non_collective_bcg_break_skips_capture_barrier(self):
        calls = []
        capture = SimpleNamespace(
            _barrier_fn=Mock(),
            _end_current_segment=lambda: calls.append("end"),
            _begin_new_segment=lambda: calls.append("begin"),
            cuda_graph=SimpleNamespace(_break_fns=[]),
        )

        wrapped = bcg_module.eager_on_graph(True, synchronize_ranks=False)(
            lambda value: value
        )
        token = bcg_module._current_capture_var.set(capture)
        try:
            self.assertEqual(wrapped("output"), "output")
        finally:
            bcg_module._current_capture_var.reset(token)

        self.assertEqual(calls, ["end", "begin"])
        capture._barrier_fn.assert_not_called()
        self.assertEqual(len(capture.cuda_graph._break_fns), 1)

    def test_collective_break_skips_capture_session_warmup_after_largest_shape(self):
        backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        backend._enable_collective_break = True
        backend._prewarmed_outputs = {ShapeKey(size=2048): "largest"}
        backend._shared_output_buffer = None
        forward_fn = Mock()

        self.assertEqual(
            backend._warmup_capture_one(ShapeKey(size=2048), forward_fn, None),
            "largest",
        )
        backend._shared_output_buffer = object()
        self.assertIsNone(
            backend._warmup_capture_one(ShapeKey(size=1792), forward_fn, None)
        )
        forward_fn.assert_not_called()

    def test_collective_break_context_is_scoped(self):
        tensor = object()
        with (
            patch.object(
                communication_op,
                "_tensor_model_parallel_all_reduce",
                return_value="eager",
            ) as eager,
            patch.object(
                communication_op,
                "bcg_tensor_model_parallel_all_reduce",
                return_value="break",
            ) as graph_break,
        ):
            self.assertEqual(
                communication_op.tensor_model_parallel_all_reduce(tensor), "eager"
            )
            with communication_op.cuda_graph_collective_break():
                self.assertEqual(
                    communication_op.tensor_model_parallel_all_reduce(tensor),
                    "break",
                )
            self.assertEqual(
                communication_op.tensor_model_parallel_all_reduce(tensor), "eager"
            )

        self.assertEqual(eager.call_count, 2)
        graph_break.assert_called_once_with(tensor)

    def test_gather_collective_break_context_is_scoped(self):
        coordinator = parallel_state.GroupCoordinator.__new__(
            parallel_state.GroupCoordinator
        )
        coordinator.unique_name = "test"
        output = object()
        input_ = object()
        with (
            patch.object(parallel_state, "_is_npu", False),
            patch.object(parallel_state, "reg_all_gather_into_tensor") as eager_gather,
            patch.object(
                parallel_state, "bcg_reg_all_gather_into_tensor"
            ) as graph_break_gather,
        ):
            coordinator.all_gather_into_tensor(output, input_)
            with parallel_state.cuda_graph_collective_break():
                coordinator.all_gather_into_tensor(output, input_)
            coordinator.all_gather_into_tensor(output, input_)

        self.assertEqual(eager_gather.call_count, 2)
        graph_break_gather.assert_called_once_with(output, input_, group_name="test")

    def test_multinode_prefill_bcg_sets_nccl_launch_order_only_for_prefill(self):
        args = ServerArgs.__new__(ServerArgs)
        args.nnodes = 2
        args.enable_cuda_graph_collective_break = True
        args.cuda_graph_config = SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.BREAKABLE)
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NCCL_LAUNCH_ORDER_IMPLICIT", None)
            args._handle_multinode_prefill_bcg_collectives()
            self.assertEqual(os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"], "1")

        args.enable_cuda_graph_collective_break = False
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NCCL_LAUNCH_ORDER_IMPLICIT", None)
            args._handle_multinode_prefill_bcg_collectives()
            self.assertNotIn("NCCL_LAUNCH_ORDER_IMPLICIT", os.environ)

    def test_capture_prewarm_skips_single_node_and_non_breakable(self):
        for backend, nnodes in (
            (BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend), 1),
            (SimpleNamespace(), 2),
        ):
            with self.subTest(backend=type(backend).__name__, nnodes=nnodes):
                runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
                runner.backend = backend
                runner.model_runner = SimpleNamespace(
                    server_args=SimpleNamespace(nnodes=nnodes)
                )
                runner._prepare_capture_shape = Mock()

                runner._prewarm_multinode_breakable_capture()

                runner._prepare_capture_shape.assert_not_called()

    def test_eagle_target_tc_piecewise_skips_last_mode_capture(self):
        eager_runner = object()
        model_runner = SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            server_args=SimpleNamespace(
                enable_return_hidden_states=True,
                return_hidden_states_mode="last",
            ),
        )

        with patch.object(
            graph_setup,
            "check_cuda_graph_backend",
            return_value=False,
        ):
            capture = capture_prefill_graph(
                model_runner=model_runner,
                eager_runner=eager_runner,
            )

        self.assertIs(capture.runner, eager_runner)

    def test_prefix_chunk_capacity_is_aggregate_and_can_be_overridden(self):
        model_runner = SimpleNamespace(
            server_args=SimpleNamespace(
                chunked_prefill_size=16,
                context_length=None,
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(
                        full_prefill_prefix_chunk_tokens=None, max_bs=8
                    )
                ),
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.empty((1, 32), dtype=torch.int32)
            ),
        )

        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (4, 16),
        )

        model_runner.server_args.chunked_prefill_size = -1
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (2, 8),
        )
        model_runner.server_args.chunked_prefill_size = 16

        model_runner.server_args.cuda_graph_config.prefill.full_prefill_prefix_chunk_tokens = (
            24
        )
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (6, 24),
        )

        model_runner.server_args.cuda_graph_config.prefill.full_prefill_prefix_chunk_tokens = (
            256
        )
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (32, 128),
        )

        # At least one token is reserved per request lane even if the requested
        # aggregate capacity is smaller than the fixed request-slot count.
        model_runner.server_args.cuda_graph_config.prefill.full_prefill_prefix_chunk_tokens = (
            2
        )
        self.assertEqual(
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4),
            (1, 4),
        )

        model_runner.server_args.cuda_graph_config.prefill.full_prefill_prefix_chunk_tokens = (
            0
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PrefillCudaGraphRunner._resolve_prefix_chunk_shape(model_runner, 4)

    def test_buffers_are_shared_across_token_buckets(self):
        backend = _FakeAttentionBackend()
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._capture_req_slots = 3
        runner._prefix_chunk_len = 2
        runner._prefix_chunk_capacity = 6
        runner._prefix_max_len = 8
        runner._prefix_capture_variants = (1, 2, 4)
        runner.device = torch.device("cpu")
        runner._prefill_static_buffers = {
            "extend_prefix_lens": torch.zeros(3, dtype=torch.int64),
            "req_pool_indices": torch.tensor([2, 0, 1], dtype=torch.int64),
        }
        runner._prefix_capture_batches = {}
        runner._prefix_capture_buffers = runner._create_chunked_prefix_buffers()
        runner.model_runner = SimpleNamespace(
            attn_backend=backend,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(24, dtype=torch.int32).view(3, 8)
            ),
        )

        first = SimpleNamespace()
        second = SimpleNamespace()
        first_key = ShapeKey(size=8, variant_label="chunked_prefix:4")
        second_key = ShapeKey(size=16, variant_label="chunked_prefix:4")

        with patch.object(
            runner_module,
            "create_chunked_prefix_cache_kv_indices",
            _FakeKVIndexKernel(),
        ):
            runner._prepare_chunked_prefix_capture(first, first_key, 4)
            runner._prepare_chunked_prefix_capture(second, second_key, 4)

            buffers = runner._prefix_capture_buffers
            self.assertIsNotNone(buffers)
            # Chunk starts are constant and prefilled at allocation.
            self.assertEqual(
                buffers.starts_cpu.tolist(),
                [[0, 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6]],
            )
            self.assertEqual(first.extend_prefix_lens_cpu, [8, 8, 8])
            self.assertEqual(first.prefix_chunk_num_tokens, [6, 6, 6, 6])
            self.assertIs(first.prefix_chunk_starts, buffers.starts)
            self.assertIs(first.prefix_chunk_seq_lens, buffers.seq_lens)
            self.assertIs(first.prefix_chunk_cu_seq_lens, buffers.cu_seq_lens)
            self.assertIs(first.prefix_chunk_starts, second.prefix_chunk_starts)
            self.assertIs(first.prefix_chunk_seq_lens, second.prefix_chunk_seq_lens)
            self.assertIs(
                first.prefix_chunk_cu_seq_lens,
                second.prefix_chunk_cu_seq_lens,
            )
            # Per-chunk KV indices are views of one shared 2-D buffer; what
            # capture bakes into the graph is the address, so compare pointers.
            for kv_chunk_idx in (0, 3):
                self.assertEqual(
                    first.prefix_chunk_kv_indices[kv_chunk_idx].data_ptr(),
                    buffers.kv_indices[kv_chunk_idx].data_ptr(),
                )
                self.assertEqual(
                    first.prefix_chunk_kv_indices[kv_chunk_idx].data_ptr(),
                    second.prefix_chunk_kv_indices[kv_chunk_idx].data_ptr(),
                )

            runner._prepare_chunked_prefix_replay(
                second_key,
                SimpleNamespace(batch_size=2, extend_prefix_lens_cpu=[5, 1]),
            )

        self.assertEqual(
            second.prefix_chunk_seq_lens.tolist(),
            [[2, 1, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]],
        )
        self.assertEqual(
            second.prefix_chunk_kv_indices[0].tolist(),
            [16, 17, 0, 0, 0, 0],
        )
        self.assertEqual(
            second.prefix_chunk_kv_indices[1].tolist(),
            [18, 19, 0, 0, 0, 0],
        )
        self.assertEqual(
            second.prefix_chunk_kv_indices[2].tolist(),
            [20, 0, 0, 0, 0, 0],
        )
        self.assertEqual(second.prefix_chunk_kv_indices[3].tolist(), [0] * 6)
        self.assertEqual(
            backend.calls,
            [(first, True), (second, True), (second, False)],
        )

    def test_prefix_gate_only_applies_to_chunked_prefix_variant(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._capture_req_slots = 4
        runner.enable_lora = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.max_num_tokens = 32
        runner.capture_num_tokens = [4]
        runner.backend = SimpleNamespace()
        runner.prefill_backend_name = Backend.FULL
        runner.has_mha_companion_layers = False
        runner._prefix_chunk_len = 2
        runner._prefix_capture_variants = (1, 2, 4)

        forward_batch = SimpleNamespace(
            batch_size=1,
            input_ids=torch.zeros(4, dtype=torch.int64),
            input_embeds=None,
            replace_embeds=None,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            extend_prefix_lens_cpu=[8],
        )

        # Prefix hits in BCG/TC-piecewise and ordinary non-MLA FullCG use the
        # normal graph topology and must retain their existing eligibility.
        runner._capture_chunked_prefix = False
        for is_full_backend in (False, True):
            with self.subTest(is_full_backend=is_full_backend):
                runner._is_full_backend = is_full_backend
                self.assertTrue(runner.can_run_graph(forward_batch))

        # The dedicated chunked-prefix topology has a fixed captured capacity.
        runner._is_full_backend = True
        runner._capture_chunked_prefix = True
        self.assertTrue(runner.can_run_graph(forward_batch))
        self.assertEqual(
            runner._shape_key(4, forward_batch).variant_label,
            "chunked_prefix:4",
        )
        forward_batch.batch_size = 2
        # Capacity is per request, not a sum: three real chunks round up to the
        # four-chunk graph even though the aggregate prefix has eight tokens.
        forward_batch.extend_prefix_lens_cpu = [5, 3]
        self.assertTrue(runner.can_run_graph(forward_batch))
        self.assertEqual(
            runner._shape_key(4, forward_batch).variant_label,
            "chunked_prefix:4",
        )
        forward_batch.extend_prefix_lens_cpu = [9, 1]
        self.assertFalse(runner.can_run_graph(forward_batch))


if __name__ == "__main__":
    unittest.main()
