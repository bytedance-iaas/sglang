import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.draft_worker_common import build_draft_tp_worker
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDeferredDeviceGraphInit(CustomTestCase):
    def test_model_runner_default_does_not_defer_graph_capture(self):
        parameter = inspect.signature(ModelRunner.__init__).parameters[
            "defer_device_graph_init"
        ]
        self.assertIs(parameter.default, False)
        helper_parameter = inspect.signature(build_draft_tp_worker).parameters[
            "defer_device_graph_init"
        ]
        self.assertIs(helper_parameter.default, False)

    def test_capture_runs_only_after_deferred_state_is_cleared(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner._device_graph_init_deferred = True
        capture = Mock(
            side_effect=lambda: self.assertFalse(runner._device_graph_init_deferred)
        )
        runner.init_device_graphs = capture

        runner.finish_deferred_device_graph_init(capture=True)

        capture.assert_called_once_with()
        self.assertFalse(runner._device_graph_init_deferred)

    def test_capture_can_be_cancelled_without_mutating_server_args(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner._device_graph_init_deferred = True
        runner.server_args = SimpleNamespace(disable_cuda_graph=False)
        runner.init_device_graphs = Mock()

        runner.finish_deferred_device_graph_init(capture=False)

        runner.init_device_graphs.assert_not_called()
        self.assertFalse(runner._device_graph_init_deferred)
        self.assertFalse(runner.server_args.disable_cuda_graph)

    def test_non_deferred_runner_rejects_finish(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner._device_graph_init_deferred = False

        with self.assertRaisesRegex(RuntimeError, "was not deferred"):
            runner.finish_deferred_device_graph_init(capture=True)


class TestDsparkGraphLifecycle(CustomTestCase):
    @patch("sglang.srt.speculative.draft_worker_common.TpModelWorker")
    @patch("sglang.srt.speculative.draft_worker_common.get_global_server_args")
    @patch(
        "sglang.srt.speculative.draft_worker_common."
        "set_global_server_args_for_scheduler"
    )
    def test_draft_worker_construction_defers_graph_capture(
        self, set_global_args, get_global_args, tp_worker_cls
    ):
        get_global_args.return_value = object()
        draft_model_runner = SimpleNamespace(model=object())
        tp_worker_cls.return_value = SimpleNamespace(model_runner=draft_model_runner)
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(context_len=131072),
                memory_pool_config=object(),
            ),
            get_memory_pool=Mock(return_value=(object(), object())),
        )
        server_args = SimpleNamespace(
            speculative_draft_attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
            attention_backend="dsv4",
        )

        build_draft_tp_worker(
            server_args=server_args,
            gpu_id=0,
            ps=SimpleNamespace(
                tp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                dp_rank=0,
            ),
            nccl_port=12345,
            target_worker=target_worker,
            algo_label="DSPARK",
            attention_backend_override="dsv4",
            defer_device_graph_init=True,
        )

        self.assertTrue(tp_worker_cls.call_args.kwargs["defer_device_graph_init"])
        inner_draft_args = tp_worker_cls.call_args.kwargs["server_args"]
        self.assertEqual(
            set_global_args.call_args_list,
            [
                call(inner_draft_args),
                call(get_global_args.return_value),
            ],
        )

    @patch("sglang.srt.speculative.draft_worker_common.TpModelWorker")
    @patch("sglang.srt.speculative.draft_worker_common.get_global_server_args")
    @patch(
        "sglang.srt.speculative.draft_worker_common."
        "set_global_server_args_for_scheduler"
    )
    def test_inner_draft_args_restore_after_worker_failure(
        self, set_global_args, get_global_args, tp_worker_cls
    ):
        saved_args = object()
        get_global_args.return_value = saved_args
        tp_worker_cls.side_effect = RuntimeError("draft worker failed")
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(context_len=131072),
                memory_pool_config=object(),
            ),
            get_memory_pool=Mock(return_value=(object(), object())),
        )
        server_args = SimpleNamespace(
            speculative_draft_attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
            attention_backend="dsv4",
        )

        with self.assertRaisesRegex(RuntimeError, "draft worker failed"):
            build_draft_tp_worker(
                server_args=server_args,
                gpu_id=0,
                ps=SimpleNamespace(
                    tp_rank=0,
                    moe_ep_rank=0,
                    pp_rank=0,
                    attn_cp_rank=0,
                    moe_dp_rank=0,
                    dp_rank=0,
                ),
                nccl_port=12345,
                target_worker=target_worker,
                algo_label="DSPARK",
                attention_backend_override="dsv4",
            )

        inner_draft_args = tp_worker_cls.call_args.kwargs["server_args"]
        self.assertEqual(
            set_global_args.call_args_list,
            [call(inner_draft_args), call(saved_args)],
        )

    @patch(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.is_cuda",
        return_value=False,
    )
    def test_dspark_finishes_capture_after_shared_modules_are_attached(self, _):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._is_context_only_pp_prefill_rank = False
        worker._decode_graph_allowed = True
        worker._draft_is_moe = True
        worker.device = "cuda"
        worker.gpu_id = 0
        worker._draft_sampler = None
        worker._maybe_build_draft_sampler = Mock(return_value=None)
        worker._proposer = SimpleNamespace(attach_draft_sampler=Mock())
        worker._draft_context = lambda: nullcontext()
        worker._draft_worker = SimpleNamespace()
        worker.draft_model = SimpleNamespace(
            embed_tokens=object(),
            lm_head=object(),
        )
        runner = SimpleNamespace(graph_runner=None)

        def finish_deferred(*, capture):
            self.assertTrue(capture)
            self.assertIsNotNone(worker.draft_model.embed_tokens)
            self.assertIsNotNone(worker.draft_model.lm_head)
            runner.graph_runner = object()

        runner.finish_deferred_device_graph_init = Mock(side_effect=finish_deferred)
        worker.draft_model_runner = runner

        worker.init_cuda_graphs()

        runner.finish_deferred_device_graph_init.assert_called_once_with(capture=True)

    @patch(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.is_cuda",
        return_value=False,
    )
    def test_dspark_rejects_capture_before_shared_modules(self, _):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._is_context_only_pp_prefill_rank = False
        worker._decode_graph_allowed = True
        worker._draft_is_moe = True
        worker.device = "cuda"
        worker.gpu_id = 0
        worker.draft_model = SimpleNamespace(embed_tokens=None, lm_head=None)

        with self.assertRaisesRegex(RuntimeError, "shared modules"):
            worker.init_cuda_graphs()

    @patch(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.is_cuda",
        return_value=False,
    )
    def test_disabled_graph_completes_deferred_lifecycle_without_capture(self, _):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._is_context_only_pp_prefill_rank = False
        worker._decode_graph_allowed = False
        worker._draft_is_moe = True
        worker.device = "cuda"
        worker.gpu_id = 0
        worker._draft_context = lambda: nullcontext()
        worker._draft_worker = SimpleNamespace()
        worker.draft_model = SimpleNamespace(embed_tokens=None, lm_head=None)
        runner = SimpleNamespace(
            graph_runner=None,
            finish_deferred_device_graph_init=Mock(),
        )
        worker.draft_model_runner = runner

        worker.init_cuda_graphs()

        runner.finish_deferred_device_graph_init.assert_called_once_with(capture=False)


class TestDraftServerArgsIsolation(CustomTestCase):
    def _scheduler(self, draft_worker_cls):
        target_args = SimpleNamespace(
            speculative_draft_load_format="draft-format",
            load_format="target-format",
            context_length=131072,
            attention_backend="dsv4",
        )
        algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_dspark=lambda: True,
            is_ngram=lambda: False,
            create_worker=Mock(return_value=draft_worker_cls),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = target_args
        scheduler.spec_algorithm = algorithm
        scheduler.ps = SimpleNamespace(
            gpu_id=0,
            tp_rank=0,
            moe_ep_rank=0,
            dp_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
        )
        scheduler.nccl_port = 12345
        scheduler.tp_worker = object()
        return scheduler, target_args, algorithm

    def test_draft_overrides_do_not_mutate_target_args(self):
        def build_worker(**kwargs):
            draft_args = kwargs["server_args"]
            draft_args.context_length = 4096
            draft_args.attention_backend = "draft-backend"
            return object()

        draft_worker_cls = Mock(side_effect=build_worker)
        scheduler, target_args, algorithm = self._scheduler(draft_worker_cls)
        saved_args = object()

        with (
            patch(
                "sglang.srt.managers.scheduler.get_global_server_args",
                return_value=saved_args,
            ),
            patch(
                "sglang.srt.managers.scheduler."
                "set_global_server_args_for_scheduler"
            ) as set_global,
        ):
            scheduler.maybe_init_draft_worker()

        draft_args = algorithm.create_worker.call_args.args[0]
        self.assertIsNot(draft_args, target_args)
        self.assertIs(draft_worker_cls.call_args.kwargs["server_args"], draft_args)
        self.assertEqual(draft_args.load_format, "draft-format")
        self.assertEqual(target_args.load_format, "target-format")
        self.assertEqual(target_args.context_length, 131072)
        self.assertEqual(target_args.attention_backend, "dsv4")
        self.assertEqual(
            set_global.call_args_list,
            [call(draft_args), call(saved_args)],
        )

    def test_global_target_args_are_restored_when_draft_init_fails(self):
        draft_worker_cls = Mock(side_effect=RuntimeError("draft init failed"))
        scheduler, _, algorithm = self._scheduler(draft_worker_cls)
        saved_args = object()

        with (
            patch(
                "sglang.srt.managers.scheduler.get_global_server_args",
                return_value=saved_args,
            ),
            patch(
                "sglang.srt.managers.scheduler."
                "set_global_server_args_for_scheduler"
            ) as set_global,
            self.assertRaisesRegex(RuntimeError, "draft init failed"),
        ):
            scheduler.maybe_init_draft_worker()

        draft_args = algorithm.create_worker.call_args.args[0]
        self.assertEqual(
            set_global.call_args_list,
            [call(draft_args), call(saved_args)],
        )


if __name__ == "__main__":
    unittest.main()
