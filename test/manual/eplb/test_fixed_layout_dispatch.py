"""CPU regression for both model callers that bypass the MegaMoE adapter.

Extract the actual forward methods to avoid importing optional attention/GPU
backends. Stop at TopK: this checks that fixed layouts reach dispatch even when
periodic EPLB is off, without pretending to test the expert math on CPU.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class ReachedTopK(Exception):
    pass


def load_function(path, name, globals_, class_name=None):
    tree = ast.parse(path.read_text())
    if class_name:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    method = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    method.returns = None
    method.decorator_list = []
    for arg in method.args.args:
        arg.annotation = None
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec"), globals_)
    return globals_[name]


class FixedLayoutDispatch(unittest.TestCase):
    def test_normal_and_dual_stream_pass_required_mapping(self):
        root = Path(__file__).resolve().parents[3] / "python/sglang/srt"
        stream = SimpleNamespace(wait_stream=lambda other: None)
        for method in ("forward_normal", "forward_normal_dual_stream"):
            for layout, redundant, periodic, nextn, expected in (
                ("trivial", 0, False, False, False),
                ("fixed.pt", 8, False, False, True),
                ("trivial", 8, False, False, True),
                ("trivial", 0, True, False, True),
                ("fixed.pt", 8, False, True, False),
            ):
                with self.subTest(
                    method=method,
                    layout=layout,
                    redundant=redundant,
                    periodic=periodic,
                    nextn=nextn,
                ):
                    cfg = SimpleNamespace(
                        enable_eplb=periodic,
                        init_expert_location=layout,
                        ep_num_redundant_experts=redundant,
                    )
                    scope = dict(
                        get_server_args=lambda: cfg,
                        get_exec=lambda: SimpleNamespace(moe=cfg),
                    )
                    remap = load_function(
                        root / "layers/moe/topk.py", "_eplb_remap_enabled", scope
                    )
                    mapping = object()
                    dispatch = Mock(return_value=mapping)
                    topk = Mock(side_effect=ReachedTopK)
                    obj = SimpleNamespace(
                        layer_id=2,
                        is_nextn=nextn,
                        is_hash=False,
                        num_fused_shared_experts=0,
                        alt_stream=stream,
                        experts=SimpleNamespace(
                            moe_runner_config=SimpleNamespace(inplace=False)
                        ),
                        _maybe_quant_moe_input_once=lambda _: None,
                        gate=lambda *args, **kwargs: object(),
                        topk=topk,
                    )
                    scope.update(
                        _eplb_remap_enabled=remap,
                        ExpertLocationDispatchInfo=SimpleNamespace(init_new=dispatch),
                        get_forward=lambda: SimpleNamespace(
                            flashinfer_trtllm_bypass=False,
                            cp_moe_precomputed_shared_output=None,
                        ),
                        torch=SimpleNamespace(
                            cuda=SimpleNamespace(current_stream=lambda: stream)
                        ),
                    )
                    forward = load_function(
                        root / "models/deepseek_v2.py",
                        method,
                        scope,
                        class_name="DeepseekV2MoE",
                    )
                    with self.assertRaises(ReachedTopK):
                        forward(obj, SimpleNamespace(shape=(1, 16)))
                    self.assertIs(
                        topk.call_args.kwargs["expert_location_dispatch_info"],
                        mapping if expected else None,
                    )
                    self.assertEqual(dispatch.call_count, int(expected))


if __name__ == "__main__":
    unittest.main()
