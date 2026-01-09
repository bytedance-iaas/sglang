import os, sys

print("[sglang_debugpy_hook] imported from", __file__, flush=True)
print("[sglang_debugpy_hook] pid=", os.getpid(),
      "SGLANG_DEBUGPY=", os.getenv("SGLANG_DEBUGPY"),
      "RANK=", os.getenv("RANK"),
      "LOCAL_RANK=", os.getenv("LOCAL_RANK"),
      flush=True)

# If not enabled, do nothing (NO exit)
if os.getenv("SGLANG_DEBUGPY", "0") != "1":
    print("[sglang_debugpy_hook] disabled in this process", flush=True)
    # Important: return-like behavior
else:
    try:
        import debugpy
        print("[sglang_debugpy_hook] debugpy imported", flush=True)
    except Exception as e:
        print("[sglang_debugpy_hook] debugpy import failed:", repr(e), flush=True)
        raise

    rank = int(os.getenv("LOCAL_RANK") or os.getenv("RANK") or "0")
    base_port = int(os.getenv("DEBUGPY_BASE_PORT", "5678"))
    host = os.getenv("DEBUGPY_HOST", "0.0.0.0")
    port = base_port + rank

    try:
        debugpy.listen((host, port))
        print(f"[debugpy] listening on {host}:{port} (rank={rank}, pid={os.getpid()})", flush=True)
    except Exception as e:
        print(f"[debugpy] listen failed on {host}:{port}: {repr(e)}", flush=True)
        raise

    if os.getenv("DEBUGPY_WAIT", "0") == "1":
        wait_rank = int(os.getenv("DEBUGPY_WAIT_RANK", "0"))
        if rank == wait_rank:
            print(f"[debugpy] waiting for client on rank={rank} pid={os.getpid()} port={port}", flush=True)
            import debugpy as _d
            _d.wait_for_client()
            _d.breakpoint()