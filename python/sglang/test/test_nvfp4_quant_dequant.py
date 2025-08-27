#!/usr/bin/env python3

import torch
import time
# from sglang.srt.layers.quantization.nvfp4_tensor import NVFP4QuantizeUtil
import numpy as np

# from flashinfer import nvfp4_batched_quantize
# from sglang.srt.layers.quantization.nvfp4_tensor import NVFP4QuantizeUtil
from sglang.srt.layers.quantization.nvfp4_tensor import KVFP4QuantizeUtil


def calculate_accuracy_metrics(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> dict[str, float]:
    """Calculate accuracy metrics between original and reconstructed tensors."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(original - reconstructed)).item()

    # PSNR calculation
    max_val = torch.max(torch.abs(original)).item()
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float("inf")

    # Relative error
    rel_error = torch.mean(
        torch.abs(original - reconstructed) / (torch.abs(original) + 1e-8)
    ).item()

    return {
        "MSE": mse,
        "MAE": mae,
        "PSNR": psnr,
        "Relative Error": rel_error,
    }

def benchmark_quantization_performance(b, m, n) -> None:
    """Benchmark FP8 vs KVFP4 quantization performance and accuracy."""
    # Test configuration
    num_runs = 100

    print(f"Testing tensor size: [{b}, {m}, {n}]")
    print(f"Auto-selecting optimal kernel based on tensor shape...")
    tensor_bf16 = torch.randn(b, m, n, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(10):
        _ = tensor_bf16 * 2
    torch.cuda.synchronize()

    # FP8 Benchmark
    print("\n=== FP8 Quant/Dequant ===")

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp8 = tensor_bf16.to(torch.float8_e4m3fn)
    torch.cuda.synchronize()
    fp8_quant_time = (time.time() - start_time) / num_runs

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp8_dequant = tensor_fp8.to(torch.bfloat16)
    torch.cuda.synchronize()
    fp8_dequant_time = (time.time() - start_time) / num_runs

    print(f"FP8 Quant: {fp8_quant_time * 1000:.2f} ms")
    print(f"FP8 Dequant: {fp8_dequant_time * 1000:.2f} ms")

    fp8_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp8_dequant)
    print(
        f"FP8 Accuracy - MSE: {fp8_metrics['MSE']:.6f}, "
        f"MAE: {fp8_metrics['MAE']:.6f}, PSNR: {fp8_metrics['PSNR']:.2f}dB"
    )

    del tensor_fp8, tensor_fp8_dequant

    # KVFP4 Optimized Benchmark
    print("\n=== KVFP4 Optimized Quant/Dequant ===")

    # Warmup
    tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    _ = KVFP4QuantizeUtil.batched_dequantize(tensor_fp4, scale_factors)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp4, scale_factors = KVFP4QuantizeUtil.batched_quantize(tensor_bf16)
    torch.cuda.synchronize()
    fp4_quant_time = (time.time() - start_time) / num_runs

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        tensor_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(tensor_fp4, scale_factors)
    torch.cuda.synchronize()
    fp4_dequant_time = (time.time() - start_time) / num_runs

    # print(tensor_fp4.shape)
    # print(scale_factors.shape)

    print(f"KVFP4 Quant: {fp4_quant_time * 1000:.2f} ms")
    print(f"KVFP4 Dequant: {fp4_dequant_time * 1000:.2f} ms")

    fp4_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp4_dequant)
    print(
        f"KVFP4 Accuracy - MSE: {fp4_metrics['MSE']:.6f}, "
        f"MAE: {fp4_metrics['MAE']:.6f}, PSNR: {fp4_metrics['PSNR']:.2f}dB"
    )

    # Comparison
    print("\n=== Performance Comparison ===")
    fp8_total = fp8_quant_time + fp8_dequant_time
    fp4_total = fp4_quant_time + fp4_dequant_time
    speedup = fp8_total / fp4_total
    print(f"FP8 vs FP4 Total Time: {fp8_total * 1000:.2f} ms vs {fp4_total * 1000:.2f} ms")
    print(f"FP4 Speedup: {speedup:.2f}x")

    print("\n=== Accuracy Comparison ===")
    mse_ratio = fp8_metrics["MSE"] / fp4_metrics["MSE"]
    psnr_diff = fp8_metrics["PSNR"] - fp4_metrics["PSNR"]
    print(f"FP8 vs FP4 - MSE Ratio: {mse_ratio:.2f}x")
    print(f"FP8 vs FP4 - PSNR Diff: {psnr_diff:.2f}dB")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}")
    # b, m, n = 256, 8192, 4096
    # b, m, n = 220477, 1, 576
    # b, m, n = 2868672, 8, 128
    b, m, n = 2868672, 8, 64
    benchmark_quantization_performance(b, m, n)



# def calculate_accuracy_metrics(
#     original: torch.Tensor, reconstructed: torch.Tensor
# ) -> dict[str, float]:
#     """Calculate accuracy metrics between original and reconstructed tensors."""
#     mse = torch.mean((original - reconstructed) ** 2).item()
#     mae = torch.mean(torch.abs(original - reconstructed)).item()

#     # PSNR calculation
#     max_val = torch.max(torch.abs(original)).item()
#     psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float("inf")

#     # Relative error
#     rel_error = torch.mean(
#         torch.abs(original - reconstructed) / (torch.abs(original) + 1e-8)
#     ).item()

#     return {
#         "MSE": mse,
#         "MAE": mae,
#         "PSNR": psnr,
#         "Relative Error": rel_error,
#     }

# def benchmark_quantization_performance(e, m, n) -> None:
#     """Benchmark FP8 vs NVFP4 quantization performance and accuracy."""
#     # Test configuration
#     num_runs = 100

#     print(f"Testing tensor size: [{e}, {m}, {n}]")
#     print(f"Auto-selecting optimal kernel based on tensor shape...")
#     tensor_bf16 = torch.randn(e, m, n, dtype=torch.bfloat16, device="cuda")

#     # Warmup
#     for _ in range(10):
#         _ = tensor_bf16 * 2
#     torch.cuda.synchronize()

#     # FP8 Benchmark
#     print("\n=== FP8 Quant/Dequant ===")

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp8 = tensor_bf16.to(torch.float8_e4m3fn)
#     torch.cuda.synchronize()
#     fp8_quant_time = (time.time() - start_time) / num_runs

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp8_dequant = tensor_fp8.to(torch.bfloat16)
#     torch.cuda.synchronize()
#     fp8_dequant_time = (time.time() - start_time) / num_runs

#     print(f"FP8 Quant: {fp8_quant_time * 1000:.2f} ms")
#     print(f"FP8 Dequant: {fp8_dequant_time * 1000:.2f} ms")

#     fp8_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp8_dequant)
#     print(
#         f"FP8 Accuracy - MSE: {fp8_metrics['MSE']:.6f}, "
#         f"MAE: {fp8_metrics['MAE']:.6f}, PSNR: {fp8_metrics['PSNR']:.2f}dB"
#     )

#     del tensor_fp8, tensor_fp8_dequant

#     # NVFP4 Optimized Benchmark
#     print("\n=== NVFP4 Optimized Quant/Dequant ===")

#     global_sf = (448 * 6) / tensor_bf16.abs().max().float()
#     sf_vec_size = 16

#     # Warmup
#     tensor_fp4, scale_factors = NVFP4QuantizeUtil.batched_quantize(
#         tensor_bf16, global_sf, sf_vec_size
#     )
#     _ = NVFP4QuantizeUtil.batched_dequantize(
#         tensor_fp4, scale_factors, global_sf, torch.bfloat16
#     )

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp4, scale_factors = NVFP4QuantizeUtil.batched_quantize(
#             tensor_bf16, global_sf, sf_vec_size
#         )
#     torch.cuda.synchronize()
#     fp4_quant_time = (time.time() - start_time) / num_runs

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp4_dequant = NVFP4QuantizeUtil.batched_dequantize(
#             tensor_fp4, scale_factors, global_sf, torch.bfloat16
#         )
#     torch.cuda.synchronize()
#     fp4_dequant_time = (time.time() - start_time) / num_runs

#     print(tensor_fp4.shape)
#     print(scale_factors.shape)

#     print(f"NVFP4 Quant: {fp4_quant_time * 1000:.2f} ms")
#     print(f"NVFP4 Dequant: {fp4_dequant_time * 1000:.2f} ms")

#     fp4_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp4_dequant)
#     print(
#         f"FP4 Accuracy - MSE: {fp4_metrics['MSE']:.6f}, "
#         f"MAE: {fp4_metrics['MAE']:.6f}, PSNR: {fp4_metrics['PSNR']:.2f}dB"
#     )

#     # Comparison
#     print("\n=== Performance Comparison ===")
#     fp8_total = fp8_quant_time + fp8_dequant_time
#     fp4_total = fp4_quant_time + fp4_dequant_time
#     speedup = fp8_total / fp4_total
#     print(f"FP8 vs FP4 Total Time: {fp8_total * 1000:.2f} ms vs {fp4_total * 1000:.2f} ms")
#     print(f"FP4 Speedup: {speedup:.2f}x")

#     print("\n=== Accuracy Comparison ===")
#     mse_ratio = fp8_metrics["MSE"] / fp4_metrics["MSE"]
#     psnr_diff = fp8_metrics["PSNR"] - fp4_metrics["PSNR"]
#     print(f"FP8 vs FP4 - MSE Ratio: {mse_ratio:.2f}x")
#     print(f"FP8 vs FP4 - PSNR Diff: {psnr_diff:.2f}dB")


# if __name__ == "__main__":
#     print(f"GPU: {torch.cuda.get_device_name()}")
#     # e, m, n = 256, 8192, 4096
#     e, m, n = 220477, 1, 576
#     #e, m, n = 220477, 36, 16
#     #e, m, n = 220477, 64,16
#     #e, m, n = 220544, 1, 576
#     benchmark_quantization_performance(e, m, n)








# def calculate_accuracy_metrics(
#     original: torch.Tensor, reconstructed: torch.Tensor
# ) -> dict[str, float]:
#     """Calculate accuracy metrics between original and reconstructed tensors."""
#     mse = torch.mean((original - reconstructed) ** 2).item()
#     mae = torch.mean(torch.abs(original - reconstructed)).item()

#     # PSNR calculation
#     max_val = torch.max(torch.abs(original)).item()
#     psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float("inf")

#     # Relative error
#     rel_error = torch.mean(
#         torch.abs(original - reconstructed) / (torch.abs(original) + 1e-8)
#     ).item()

#     return {
#         "MSE": mse,
#         "MAE": mae,
#         "PSNR": psnr,
#         "Relative Error": rel_error,
#     }

# def benchmark_quantization_performance(e, m, n) -> None:
#     """Benchmark FP8 vs NVFP4 quantization performance and accuracy."""
#     # Test configuration
#     num_runs = 100

#     print(f"Testing tensor size: [{e}, {m}, {n}]")
#     print(f"Auto-selecting optimal kernel based on tensor shape...")
#     tensor_bf16 = torch.randn(e, m, n, dtype=torch.bfloat16, device="cuda")

#     # Warmup
#     for _ in range(10):
#         _ = tensor_bf16 * 2
#     torch.cuda.synchronize()

#     # FP8 Benchmark
#     print("\n=== FP8 Quant/Dequant ===")

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp8 = tensor_bf16.to(torch.float8_e4m3fn)
#     torch.cuda.synchronize()
#     fp8_quant_time = (time.time() - start_time) / num_runs

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp8_dequant = tensor_fp8.to(torch.bfloat16)
#     torch.cuda.synchronize()
#     fp8_dequant_time = (time.time() - start_time) / num_runs

#     print(f"FP8 Quant: {fp8_quant_time * 1000:.2f} ms")
#     print(f"FP8 Dequant: {fp8_dequant_time * 1000:.2f} ms")

#     fp8_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp8_dequant)
#     print(
#         f"FP8 Accuracy - MSE: {fp8_metrics['MSE']:.6f}, "
#         f"MAE: {fp8_metrics['MAE']:.6f}, PSNR: {fp8_metrics['PSNR']:.2f}dB"
#     )

#     del tensor_fp8, tensor_fp8_dequant

#     # NVFP4 Optimized Benchmark
#     print("\n=== NVFP4 Optimized Quant/Dequant ===")

#     global_sf = (448 * 6) / tensor_bf16.abs().max().float()
#     sf_vec_size = 16

#     # Warmup
#     tensor_fp4, scale_factors = NVFP4QuantizeUtil.batched_quantize(
#         tensor_bf16, global_sf, sf_vec_size
#     )
#     print(f"[horenc] scale_factors.shape = {scale_factors.shape}")
#     _ = NVFP4QuantizeUtil.batched_dequantize(
#         tensor_fp4, scale_factors, global_sf, torch.bfloat16
#     )

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp4, scale_factors = NVFP4QuantizeUtil.batched_quantize(
#             tensor_bf16, global_sf, sf_vec_size
#         )
#     torch.cuda.synchronize()
#     fp4_quant_time = (time.time() - start_time) / num_runs

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp4_dequant = NVFP4QuantizeUtil.batched_dequantize(
#             tensor_fp4, scale_factors, global_sf, torch.bfloat16
#         )
#     torch.cuda.synchronize()
#     fp4_dequant_time = (time.time() - start_time) / num_runs

#     print(f"NVFP4 Quant: {fp4_quant_time * 1000:.2f} ms")
#     print(f"NVFP4 Dequant: {fp4_dequant_time * 1000:.2f} ms")

#     fp4_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp4_dequant)
#     print(
#         f"FP4 Accuracy - MSE: {fp4_metrics['MSE']:.6f}, "
#         f"MAE: {fp4_metrics['MAE']:.6f}, PSNR: {fp4_metrics['PSNR']:.2f}dB"
#     )

#     # Comparison
#     print("\n=== Performance Comparison ===")
#     fp8_total = fp8_quant_time + fp8_dequant_time
#     fp4_total = fp4_quant_time + fp4_dequant_time
#     speedup = fp8_total / fp4_total
#     print(f"FP8 vs FP4 Total Time: {fp8_total * 1000:.2f} ms vs {fp4_total * 1000:.2f} ms")
#     print(f"FP4 Speedup: {speedup:.2f}x")

#     print("\n=== Accuracy Comparison ===")
#     mse_ratio = fp8_metrics["MSE"] / fp4_metrics["MSE"]
#     psnr_diff = fp8_metrics["PSNR"] - fp4_metrics["PSNR"]
#     print(f"FP8 vs FP4 - MSE Ratio: {mse_ratio:.2f}x")
#     print(f"FP8 vs FP4 - PSNR Diff: {psnr_diff:.2f}dB")


# if __name__ == "__main__":
#     print(f"GPU: {torch.cuda.get_device_name()}")
#     #e, m, n = 256, 8192, 4096
#     e, m, n = 220477, 1, 576
#     benchmark_quantization_performance(e, m, n)


# def calculate_accuracy_metrics(original, reconstructed):
#     """Calculate accuracy metrics between original and reconstructed tensors."""
#     mse = torch.mean((original - reconstructed) ** 2).item()
#     mae = torch.mean(torch.abs(original - reconstructed)).item()
    
#     # PSNR calculation
#     max_val = torch.max(torch.abs(original)).item()
#     psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')
    
#     # Relative error
#     rel_error = torch.mean(torch.abs(original - reconstructed) / (torch.abs(original) + 1e-8)).item()
    
#     return {
#         'MSE': mse,
#         'MAE': mae, 
#         'PSNR': psnr,
#         'Relative Error': rel_error
#     }

# def benchmark_quantization_performance():
#     num_runs = 1
#     # E, M, N = 256, 8192, 4096
#     E, M, N = 220477, 1, 288
#     print(f"Tensor size: [{E}, {M}, {N}]")
#     tensor_bf16 = torch.randn(E, M, N, dtype=torch.bfloat16, device='cuda')

#     # warm_up
#     for _ in range(10):
#         _ = tensor_bf16 * 2
#     torch.cuda.synchronize()

#     print("\n=== FP8 Quant/Dequant ===")
#     _ = tensor_bf16.to(torch.float8_e4m3fn)
#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp8 = tensor_bf16.to(torch.float8_e4m3fn)
#     torch.cuda.synchronize()
#     fp8_quant_time = (time.time() - start_time) / num_runs

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         tensor_fp8_dequant = tensor_fp8.to(torch.bfloat16)
#     torch.cuda.synchronize()
#     fp8_dequant_time = (time.time() - start_time) / num_runs

#     print(f"FP8 Quant: {fp8_quant_time*1000:.2f} ms")
#     print(f"FP8 Dequant: {fp8_dequant_time*1000:.2f} ms")
#     # FP8 Accuracy
#     fp8_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp8_dequant)
#     print(f"FP8 Accuracy - MSE: {fp8_metrics['MSE']:.6f}, MAE: {fp8_metrics['MAE']:.6f}, PSNR: {fp8_metrics['PSNR']:.2f}dB")
#     del tensor_fp8
#     del tensor_fp8_dequant

#     print("\n=== NVFP4 Quant/Dequant ===")
#     global_sf = (448 * 6) / tensor_bf16.abs().max().float()
#     sf_vec_size = 16
#     _, _ = nvfp4_batched_quantize(
#             tensor_bf16, global_sf, sf_vec_size
#     )
#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         # tensor_fp4, scale_factors = NVFP4QuantizeUtil.batched_quantize(
#         tensor_fp4, scale_factors = nvfp4_batched_quantize(
#             tensor_bf16, global_sf, sf_vec_size
#         )        
#         """
#             print(f"tensor_bf16.shape = {tensor_bf16.shape}")
#             print(f"global_sf.shape = {global_sf.shape} global_sf = {global_sf} global_sf.dtype = {global_sf.dtype}")

#             print(f"tensor_fp4.shape = {tensor_fp4.shape}")
#             print(f"tensor_fp4.dtype = {tensor_fp4.dtype}")
#             print(f"type(tensor_fp4) = {type(tensor_fp4)}")
#             print(f"scale_factors.shape = {scale_factors.shape}")
#             print(f"scale_factors.dtype = {scale_factors.dtype}")
#             print(f"type(scale_factors) = {type(scale_factors)}")

#             print(f"global_sf.shape = {global_sf.shape}")
#             print(f"global_sf.dtype = {global_sf.dtype}")
#             print(f"type(global_sf) = {type(global_sf)}")

#             tensor_bf16.shape = torch.Size([256, 8192, 4096])
#             global_sf.shape = torch.Size([]) global_sf = 407.6587829589844 global_sf.dtype = torch.float32

            

#             tensor_fp4.shape = torch.Size([256, 8192, 2048])
#             tensor_fp4.dtype = torch.uint8
#             type(tensor_fp4) = <class 'torch.Tensor'>
#             scale_factors.shape = torch.Size([256, 2097152])
#             scale_factors.dtype = torch.uint8
#             type(scale_factors) = <class 'torch.Tensor'>

#             [horenc](0) 1 cache_k_nope_fp4.dtype = torch.uint8 cache_k_nope_fp4.shape = torch.Size([220477, 1, 288]) type(cache_k_nope_fp4) = <class 'torch.Tensor'>
#             [horenc](0) 2 cache_k_nope_fp4_sf.dtype = torch.uint8 cache_k_nope_fp4_sf.shape = torch.Size([220477, 4608]) type(cache_k_nope_fp4_sf) = <class 'torch.Tensor'>
#             [horenc](0) 3-1 global_sf = 24144.841796875 global_sf.shape = torch.Size([]) global_sf.dtype = torch.float32 type(global_sf) = <class 'torch.Tensor'>
            

#             global_sf.shape = torch.Size([])
#             global_sf.dtype = torch.float32
#             type(global_sf) = <class 'torch.Tensor'>
#         """
#     torch.cuda.synchronize()
#     fp4_quant_time = (time.time() - start_time) / num_runs

#     # torch.compile() warm up
#     #_ = NVFP4QuantizeUtil.batched_dequantize(
#     _ = NVFP4Dequantizer.batched_dequantize(
#         tensor_fp4, scale_factors, global_sf, torch.bfloat16
#     )
#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(num_runs):
#         # tensor_fp4_dequant = NVFP4QuantizeUtil.batched_dequantize(
#         tensor_fp4_dequant = NVFP4Dequantizer.batched_dequantize(
#             tensor_fp4, scale_factors, global_sf, torch.bfloat16
#         )
#     torch.cuda.synchronize()
#     fp4_dequant_time = (time.time() - start_time) / num_runs

#     print(f"NVFP4 Quant: {fp4_quant_time*1000:.2f} ms")
#     print(f"NVFP4 Dequant: {fp4_dequant_time*1000:.2f} ms")

#     # FP4 Accuracy
#     fp4_metrics = calculate_accuracy_metrics(tensor_bf16, tensor_fp4_dequant)
#     print(f"FP4 Accuracy - MSE: {fp4_metrics['MSE']:.6f}, MAE: {fp4_metrics['MAE']:.6f}, PSNR: {fp4_metrics['PSNR']:.2f}dB")
    
#     print("\n=== Accuracy Comparison ===")
#     print(f"FP8 vs FP4 - MSE Ratio: {fp8_metrics['MSE']/fp4_metrics['MSE']:.2f}x")
#     print(f"FP8 vs FP4 - PSNR Diff: {fp8_metrics['PSNR']-fp4_metrics['PSNR']:.2f}dB")


# if __name__ == "__main__":
#     print(f"Use GPU: {torch.cuda.get_device_name()}")
#     benchmark_quantization_performance()
