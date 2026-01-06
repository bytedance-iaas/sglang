import os, sys, inspect, importlib, ast
from pathlib import Path


# ------------------ compat: set global server args ------------------
def set_global_server_args_compat(server_args):
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

    # Otherwise, set module-global variable used by get_global_server_args()
    src = inspect.getsource(sa.get_global_server_args)
    candidates = ["_GLOBAL_SERVER_ARGS", "GLOBAL_SERVER_ARGS", "_global_server_args", "global_server_args"]
    for v in candidates:
        if v in src:
            setattr(sa, v, server_args)
            return
    for v in candidates:
        if hasattr(sa, v):
            setattr(sa, v, server_args)
            return
    raise RuntimeError("Cannot set global server args in this SGLang version.")


# ------------------ AST scan: find a real top-level class ------------------
def _has_top_level_class(pyfile: Path, class_name: str) -> bool:
    try:
        src = pyfile.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src, filename=str(pyfile))
    except Exception:
        return False
    return any(isinstance(n, ast.ClassDef) and n.name == class_name for n in tree.body)


def _pyfile_to_module(pkg_root: Path, pyfile: Path) -> str:
    pkg_parent = pkg_root.parent  # dir containing 'sglang'
    rel = pyfile.relative_to(pkg_parent).with_suffix("")  # sglang/...
    return ".".join(rel.parts)


def import_class_by_ast_scan(class_name: str):
    import sglang
    pkg_root = Path(sglang.__file__).resolve().parent  # .../sglang

    candidates = [py for py in pkg_root.rglob("*.py") if _has_top_level_class(py, class_name)]
    if not candidates:
        raise ImportError(f"Could not find top-level class {class_name} under {pkg_root}")

    def rank(p: Path):
        n = p.name.lower()
        s = 0
        if class_name.lower() in n: s += 10
        if "model" in n: s += 4
        if "runner" in n: s += 4
        if "config" in n: s += 3
        if "/test/" in str(p): s -= 5
        s -= len(p.parts) * 0.1
        return -s

    candidates.sort(key=rank)

    last_err = None
    for py in candidates:
        modname = _pyfile_to_module(pkg_root, py)
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, class_name):
                return getattr(mod, class_name), modname, str(py)
        except Exception as e:
            last_err = (modname, py, e)

    if last_err:
        modname, py, e = last_err
        raise ImportError(f"None exported {class_name}. Last tried {modname} ({py}) -> {repr(e)}")
    raise ImportError(f"Found candidates for {class_name}, but could not import/export it.")


# ------------------ build ModelRunner single-process ------------------
def build_model_runner(server_args):
    ModelRunner, mr_mod, mr_file = import_class_by_ast_scan("ModelRunner")
    print(f"[debug] ModelRunner from {mr_mod} ({mr_file})")

    # Try to build model_config with a few known patterns
    model_config = None

    # Pattern A: ModelConfig(server_args)
    try:
        ModelConfig, mc_mod, mc_file = import_class_by_ast_scan("ModelConfig")
        print(f"[debug] ModelConfig from {mc_mod} ({mc_file})")
        try:
            model_config = ModelConfig(server_args)
        except TypeError:
            # Pattern B: ModelConfig.from_server_args(server_args)
            if hasattr(ModelConfig, "from_server_args"):
                model_config = ModelConfig.from_server_args(server_args)
    except Exception as e:
        print("[debug] No ModelConfig class usable via scan:", repr(e))

    # Pattern C: helper function returns model_config
    if model_config is None:
        for modpath, fn in [
            ("sglang.srt.configs.model_config", "get_model_config"),
            ("sglang.srt.config.model_config", "get_model_config"),
            ("sglang.srt.model_config", "get_model_config"),
        ]:
            try:
                m = importlib.import_module(modpath)
                if hasattr(m, fn):
                    model_config = getattr(m, fn)(server_args)
                    print(f"[debug] model_config via {modpath}.{fn}")
                    break
            except Exception:
                pass

    if model_config is None:
        raise RuntimeError(
            "Could not construct model_config. "
            "Print your ModelRunner signature file and we can wire the right builder."
        )

    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=getattr(server_args, "tp", 1) if isinstance(getattr(server_args, "tp", 1), int) else 1,
        moe_ep_rank=0,
        moe_ep_size=getattr(server_args, "ep_size", 1) if hasattr(server_args, "ep_size") else 1,
        pp_rank=0,
        pp_size=getattr(server_args, "pp", 1) if hasattr(server_args, "pp") else 1,
        nccl_port=getattr(server_args, "nccl_port", 29500),
        server_args=server_args,
        dp_rank=None,
        is_draft_worker=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        draft_model_idx=None,
    )
    return runner

def force_single_process_dist(server_args):
    import os

    # pick a port
    port = getattr(server_args, "nccl_port", None)
    if port in (None, "None", "", 0):
        port = 29500
        if hasattr(server_args, "nccl_port"):
            server_args.nccl_port = port

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    # some forks use an explicit init method / address
    if hasattr(server_args, "dist_init_method") and not getattr(server_args, "dist_init_method"):
        server_args.dist_init_method = "env://"
    if hasattr(server_args, "dist_init_addr") and not getattr(server_args, "dist_init_addr"):
        server_args.dist_init_addr = f"127.0.0.1:{port}"

def main():
    repo_root = "/sgl-workspace/sglang"
    py_root = os.path.join(repo_root, "python")
    if py_root not in sys.path:
        sys.path.insert(0, py_root)

    from sglang.srt.server_args import prepare_server_args

    # Accept same CLI args as launch_server
    server_args = prepare_server_args(sys.argv[1:])
    set_global_server_args_compat(server_args)
    force_single_process_dist(server_args)   # <-- ADD THIS

    # Add your breakpoint anywhere, or in loader.py
    # import ipdb; ipdb.set_trace()

    runner = build_model_runner(server_args)

    print("✅ ModelRunner constructed in single process.")
    print("Runner type:", type(runner))
    # You can now inspect runner internals to confirm loader.py behavior:
    for attr in ["model", "model_worker", "worker", "engine"]:
        if hasattr(runner, attr):
            print(f"runner.{attr} =", type(getattr(runner, attr)))


if __name__ == "__main__":
    main()