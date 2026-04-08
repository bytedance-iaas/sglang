import os
import torch
import torch.distributed as dist

from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.distributed.parallel_state import (
    initialize_model_parallel,
    get_tensor_model_parallel_world_size,
)
import argparse

def run_debug(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    initialize_model_parallel(world_size, 1)

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size < 2:
        print(f"[Rank {rank}] Warning: TP size is {tp_size}. TP overlap needs TP >= 2.")
        
    # Try to import triton_dist
    try:
        from triton_dist.layers.nvidia import GemmARLayer
        from triton_dist.utils import init_nvshmem_by_torch_process_group
    except ImportError:
        print(f"[Rank {rank}] Failed to import triton_dist. Please ensure it is installed.")
        dist.destroy_process_group()
        return

    # 1. Init NVSHMEM
    _TP_OVERLAP_GROUP = torch.distributed.new_group(backend="gloo")
    torch.distributed.barrier(_TP_OVERLAP_GROUP)
    init_nvshmem_by_torch_process_group(_TP_OVERLAP_GROUP)

    M = args.batch_size * args.seq_len
    N = args.hidden_size
    K_full = args.intermediate_size
    K_per_rank = K_full // tp_size
    
    dtype = getattr(torch, args.dtype)

    print(f"[Rank {rank}] Initializing GemmARLayer... M_max={args.max_m}, N={N}, K_per_rank={K_per_rank}")
    gemm_ar_op = GemmARLayer(
        tp_group=_TP_OVERLAP_GROUP,
        max_M=args.max_m,
        N=N,
        K=K_per_rank,
        input_dtype=dtype,
        output_dtype=dtype,
        local_world_size=tp_size,
        persistent=True,
        copy_to_local=False,
        use_ll_kernel=args.use_ll_kernel,
        NUM_COMM_SMS=2,
    )
    
    print(f"[Rank {rank}] Initializing RowParallelLinear...")
    layer = RowParallelLinear(
        input_size=K_full,
        output_size=N,
        bias=args.bias,
        input_is_parallel=args.input_is_parallel,
        skip_bias_add=args.skip_bias_add,
        reduce_results=args.reduce_results,
        params_dtype=dtype,
    ).cuda()
    
    # Assign operator
    if args.is_attn:
        layer.gemm_ar_attn_op = gemm_ar_op
        layer.gemm_ar_mlp_op = None
    else:
        layer.gemm_ar_attn_op = None
        layer.gemm_ar_mlp_op = gemm_ar_op
        
    # Create input. 
    torch.manual_seed(42)
    full_input = torch.randn(M, K_full, dtype=dtype, device="cuda")
    dist.broadcast(full_input, src=0)
    
    # Sync weights
    dist.broadcast(layer.weight, src=0)
    if layer.bias is not None:
        dist.broadcast(layer.bias, src=0)
        
    if args.input_is_parallel:
        local_input = full_input.chunk(tp_size, dim=-1)[rank].contiguous()
    else:
        local_input = full_input.clone()

    # Wait for all
    torch.cuda.synchronize()
    dist.barrier()

    # --- Baseline ---
    print(f"[Rank {rank}] Running baseline (SGL_USE_TP_OVERLAP=0)...")
    os.environ["SGL_USE_TP_OVERLAP"] = "0"
    baseline_output, baseline_bias = layer.forward(local_input, skip_all_reduce=args.skip_all_reduce)
    torch.cuda.synchronize()

    # --- Overlap ---
    print(f"[Rank {rank}] Running overlap (SGL_USE_TP_OVERLAP=1)...")
    os.environ["SGL_USE_TP_OVERLAP"] = "1"
    overlap_output, overlap_bias = layer.forward(local_input, skip_all_reduce=args.skip_all_reduce)
    torch.cuda.synchronize()

    # Compare
    # Check shapes first
    if baseline_output.shape != overlap_output.shape:
        print(f"[Rank {rank}] ❌ SHAPE MISMATCH: Baseline {baseline_output.shape} vs Overlap {overlap_output.shape}")
        
        # Explain potential bug:
        if args.skip_all_reduce or not args.reduce_results:
            print(f"[Rank {rank}]   -> Bug Alert: TP Overlap ignores `skip_all_reduce` and `reduce_results`, always doing All-Reduce!")
    else:
        diff = torch.abs(baseline_output - overlap_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"==================================================")
        print(f"[Rank {rank}] Results:")
        print(f"[Rank {rank}] Max diff:  {max_diff}")
        print(f"[Rank {rank}] Mean diff: {mean_diff}")
        
        if max_diff > 1e-3:
            print(f"[Rank {rank}] ❌ SIGNIFICANT NUMERICAL DIFFERENCE DETECTED!")
            idx = torch.where(diff == torch.max(diff))
            print(f"[Rank {rank}] Max diff at: {idx}")
            print(f"[Rank {rank}] Baseline: {baseline_output[idx]}")
            print(f"[Rank {rank}] Overlap:  {overlap_output[idx]}")
            
            # Explain potential bug:
            if not args.input_is_parallel:
                print(f"[Rank {rank}]   -> Bug Alert: TP Overlap ignores `input_is_parallel=False` and passes the full tensor to GemmARLayer without splitting it.")
        else:
            print(f"[Rank {rank}] ✅ Outputs match numerically.")
        print(f"==================================================")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End TP Overlap Debug Script")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--max-m", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    
    # Important testing toggles
    parser.add_argument("--bias", action="store_true", help="Initialize with bias")
    parser.add_argument("--skip-bias-add", action="store_true", help="Set skip_bias_add in layer")
    parser.add_argument("--input-is-parallel", action="store_true", default=True, help="Simulate input being already partitioned")
    parser.add_argument("--not-input-is-parallel", dest="input_is_parallel", action="store_false", help="Simulate input not being partitioned")
    parser.add_argument("--skip-all-reduce", action="store_true", help="Pass skip_all_reduce=True to forward()")
    parser.add_argument("--no-reduce-results", dest="reduce_results", action="store_false", default=True, help="Set reduce_results=False in init")
    
    # Triton dist specific
    parser.add_argument("--is-attn", action="store_true", help="Test attn overlap op instead of mlp")
    parser.add_argument("--use-ll-kernel", action="store_true", help="Use low-latency kernel for triton_dist")
    args = parser.parse_args()
    
    run_debug(args)
