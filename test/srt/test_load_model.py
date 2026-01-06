# /sgl-workspace/sglang/test/srt/test_load_model.py
import os
import sys
import inspect
import importlib
from pathlib import Path
from sglang.srt.configs.model_config import ModelConfig

def set_global_server_args_compat(server_args):
    """
    Your SGLang version doesn't export set_global_server_args().
    This sets the module-global variable that get_global_server_args() uses.
    """
    import sglang.srt.server_args as sa

    # If a setter exists in your branch, use it.
    for fn in [
        "set_global_server_args",
        "init_global_server_args",
        "set_server_args",
        "set_global_args",
        "set_global_server_args_for_test",
    ]:
        if hasattr(sa, fn):
            getattr(sa, fn)(server_args)
            return

    # Otherwise, detect internal variable name used by get_global_server_args()
    src = inspect.getsource(sa.get_global_server_args)
    candidates = [
        "_GLOBAL_SERVER_ARGS",
        "GLOBAL_SERVER_ARGS",
        "_global_server_args",
        "global_server_args",
    ]

    for v in candidates:
        if v in src:
            setattr(sa, v, server_args)
            return

    for v in candidates:
        if hasattr(sa, v):
            setattr(sa, v, server_args)
            return

    raise RuntimeError("Cannot find how to set global server args in this SGLang version.")


def find_module_defining_class(pkg_root: Path, class_name: str):
    """
    Scan sglang package sources to find the first .py file that contains:
      'class <class_name>'
    Return (module_name, file_path).
    """
    needle = f"class {class_name}"
    candidates = []
    for py in pkg_root.rglob("*.py"):
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if needle in txt:
            candidates.append(py)

    if not candidates:
        return None, None

    # Prefer more specific locations if multiple hits exist
    # (e.g., model_runner.py, scheduler.py, runner.py)
    def score(p: Path):
        name = p.name
        s = 0
        if "model_runner" in name:
            s += 10
        if "runner" in name:
            s += 5
        if "scheduler" in name:
            s += 3
        # shallower paths often are the “real” definition
        s -= len(p.parts)
        return -s  # smaller is better

    candidates.sort(key=score)
    chosen = candidates[0]

    # Convert file path -> module import path
    # Example: /.../site-packages/sglang/srt/managers/scheduler.py
    # module = "sglang.srt.managers.scheduler"
    pkg_parent = pkg_root.parent  # directory containing 'sglang'
    rel = chosen.relative_to(pkg_parent).with_suffix("")  # sglang/srt/.../scheduler
    module_name = ".".join(rel.parts)

    return module_name, chosen


def import_class_by_scan(class_name: str):
    import sglang  # ensure package import
    pkg_root = Path(sglang.__file__).resolve().parent  # .../sglang/__init__.py's folder

    module_name, file_path = find_module_defining_class(pkg_root, class_name)
    if module_name is None:
        raise ImportError(f"Could not find 'class {class_name}' in {pkg_root}")

    mod = importlib.import_module(module_name)
    if not hasattr(mod, class_name):
        # Sometimes class is imported into __init__.py; try importing the file module directly anyway
        raise ImportError(
            f"Found '{class_name}' in {file_path}, but module '{module_name}' "
            f"does not export attribute '{class_name}'."
        )
    return getattr(mod, class_name), module_name, str(file_path)

def main():
    repo_root = "/sgl-workspace/sglang"
    py_root = os.path.join(repo_root, "python")
    if py_root not in sys.path:
        sys.path.insert(0, py_root)

    from sglang.srt.server_args import prepare_server_args

    argv = [
        "--model", "deepseek-ai/DeepSeek-V3.2-Exp",
        "--tp", "1",
        "--dp", "1",
        "--enable-dp-attention",
        "--load-format", "dummy",
        "--moe-runner-backend", "deep_gemm",
        "--json-model-override-args", '{"num_hidden_layers": 2}',
        "--context-length", "8192",
        "--max-total-tokens", "8192",
        "--chunked-prefill-size", "2048",
        "--mem-fraction-static", "0.90",
    ]

    server_args = prepare_server_args(argv)
    set_global_server_args_compat(server_args)

    # ---- Dynamically locate ModelRunner & ModelConfig in your tree ----
    ModelRunner, mr_mod, mr_file = import_class_by_scan("ModelRunner")
    print(f"✅ ModelRunner found in: {mr_mod}  ({mr_file})")

    model_config = ModelConfig(server_args)

    # ---- Instantiate ModelRunner to trigger loader.py paths ----
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0, tp_size=1,
        moe_ep_rank=0, moe_ep_size=1,
        pp_rank=0, pp_size=1,
        nccl_port=getattr(server_args, "nccl_port", 29500),
        server_args=server_args,
    )

    print("✅ ModelRunner created. If you put ipdb in loader.py, it should hit in this process.")
    print("Runner:", type(runner))


if __name__ == "__main__":
    main()