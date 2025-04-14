# Grouped GEMM Triton Kernel Tuning

This directory contains scripts for tuning the `grouped_gemm_triton` kernel in the SGLang library.

## Overview

The `grouped_gemm_triton` function is a Triton kernel implementation for grouped matrix multiplication, which is used in the Mixture of Experts (MoE) layers. The performance of this kernel depends on various configuration parameters, such as block sizes for different dimensions.

This tuning script helps find the optimal configuration parameters for different input sizes and data types.

## Usage

You can use the `run_tuning.sh` script to run the tuning process:

```bash
./run_tuning.sh [options]
```

### Options

- `--model MODEL`: The model name or path to load configuration from (optional)
  - If provided, the script will use the model's hidden size and data type
  - The script will also check for quantization configuration in the model
- `--hidden-size HIDDEN_SIZE`: The hidden size of the model (default: 4096)
  - Ignored if `--model` is provided
- `--dtype DTYPE`: The data type to use (default: float16)
  - Supported values: auto, float16, bfloat16, float32, fp8_w8a8, int8_w8a16, int8_w8a8
  - If set to "auto" and `--model` is provided, the script will use the model's data type
- `--batch-size BATCH_SIZE`: The batch size to tune for (optional)
  - If not specified, the script will tune for multiple batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
- `--no-tune`: Disable tuning mode (only benchmark the default configuration)
- `--block-shape BLOCK_SHAPE`: The block shape for quantization (optional)
  - Format: "block_n,block_k" (e.g., "128,128")
  - Ignored if `--model` is provided and the model has quantization configuration

### Examples

1. Tune for default settings (hidden_size=4096, dtype=float16):
   ```bash
   ./run_tuning.sh
   ```

2. Tune for a specific model:
   ```bash
   ./run_tuning.sh --model mistralai/Mistral-7B-v0.1
   ```

3. Tune for a specific batch size:
   ```bash
   ./run_tuning.sh --batch-size 64
   ```

4. Tune for a specific data type:
   ```bash
   ./run_tuning.sh --dtype bfloat16
   ```

5. Tune for a specific hidden size:
   ```bash
   ./run_tuning.sh --hidden-size 2048
   ```

6. Tune for a specific block shape:
   ```bash
   ./run_tuning.sh --block-shape 128,128
   ```

7. Benchmark the default configuration without tuning:
   ```bash
   ./run_tuning.sh --no-tune
   ```

## Output

The tuning script will generate a JSON file with the optimal configuration for each batch size. The file will be named according to the following pattern:

```
grouped_gemm_config_{hidden_size}_{dtype}{block_str}.json
```

For example:
```
grouped_gemm_config_4096_float16.json
grouped_gemm_config_4096_float16_block128x128.json
```

The `grouped_gemm_triton` function will automatically load the appropriate configuration from this file when it is called.

## Implementation Details

The tuning process works as follows:

1. Generate a set of possible configurations with different block sizes
2. For each configuration, benchmark the kernel with the specified parameters
3. Find the configuration with the best performance
4. Save the configuration to a JSON file

The benchmarking process uses CUDA graphs to measure the performance of the kernel with minimal overhead. 