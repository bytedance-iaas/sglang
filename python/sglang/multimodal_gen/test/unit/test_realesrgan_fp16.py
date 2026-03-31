import ast
import unittest
from pathlib import Path


class TestRealESRGANHalfPrecisionSafety(unittest.TestCase):
    def setUp(self):
        multimodal_gen_dir = Path(__file__).resolve().parents[2]
        self.src_path = (
            multimodal_gen_dir / "runtime" / "postprocess" / "realesrgan_upscaler.py"
        )
        self.tree = ast.parse(self.src_path.read_text(encoding="utf-8"))

    def _get_method(self, class_name: str, method_name: str) -> ast.FunctionDef:
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return item
        raise AssertionError(
            f"Could not find {class_name}.{method_name} in {self.src_path}"
        )

    def test_upscale_casts_input_to_model_dtype(self):
        """Regression test: fp16 weights require fp16 inputs (no float/half mismatch)."""
        upscale_fn = self._get_method("UpscalerModel", "upscale")

        found_dtype_kw = False
        for node in ast.walk(upscale_fn):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
            ):
                continue

            dtype_kw = None
            for kw in node.keywords:
                if kw.arg == "dtype":
                    dtype_kw = kw.value
                    break
            if dtype_kw is None:
                continue

            # Expect `.to(..., dtype=self.dtype)` (or equivalent)
            if (
                isinstance(dtype_kw, ast.Attribute)
                and isinstance(dtype_kw.value, ast.Name)
                and dtype_kw.value.id == "self"
                and dtype_kw.attr == "dtype"
            ):
                found_dtype_kw = True
                break

        self.assertTrue(
            found_dtype_kw,
            "Expected UpscalerModel.upscale to cast input tensor dtype to match model dtype.",
        )

    def test_model_cache_key_includes_dtype(self):
        """Regression test: fp16/fp32 loads must not collide in the global cache."""
        ensure_fn = self._get_method("ImageUpscaler", "_ensure_model_loaded")

        found_cache_key = False
        for node in ast.walk(ensure_fn):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            if node.targets[0].id != "cache_key":
                continue
            if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 3:
                continue

            third = node.value.elts[2]
            if isinstance(third, ast.Name) and third.id == "target_dtype":
                found_cache_key = True
                break

        self.assertTrue(
            found_cache_key,
            "Expected ImageUpscaler._ensure_model_loaded cache_key to include target_dtype.",
        )
