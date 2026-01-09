import os

# Only enable debugpy if requested; otherwise do nothing (NO SystemExit)
if os.getenv("SGLANG_DEBUGPY", "0") == "1":
    import debugpy

    rank = int(os.getenv("LOCAL_RANK") or os.getenv("RANK") or "0")
    base_port = int(os.getenv("DEBUGPY_BASE_PORT", "5678"))
    host = os.getenv("DEBUGPY_HOST", "0.0.0.0")
    port = base_port + rank

    debugpy.listen((host, port))
    print(f"[debugpy] listening on {host}:{port} rank={rank}", flush=True)

    if os.getenv("DEBUGPY_WAIT", "0") == "1":
        wait_rank = int(os.getenv("DEBUGPY_WAIT_RANK", "0"))
        if rank == wait_rank:
            print(f"[debugpy] waiting for client on rank={rank}", flush=True)
            debugpy.wait_for_client()
            debugpy.breakpoint()