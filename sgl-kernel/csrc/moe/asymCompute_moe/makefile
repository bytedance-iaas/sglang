# -----------------------------
# Makefile: asymCompKernelMain.cu + asymComputeKernel.cu
# -----------------------------

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

TARGET    ?= asym_moe

SRCS      := asymCompKernelMain.cu asymComputeKernel.cu

# Paths (match your environment)
FLASH_ATTN_SRC ?= /sgl-workspace/sglang/sgl-kernel/csrc/moe/asymCompute_moe/flash-attention/csrc/flash_attn/src
CUTLASS_INC    ?= /sgl-workspace/sglang/sgl-kernel/csrc/moe/asymCompute_moe/flash-attention/csrc/cutlass/include

INCLUDES  := -I$(FLASH_ATTN_SRC) -I$(CUTLASS_INC) -I$(CUDA_HOME)/include -I/usr/lib/python3.12/site-packages/torch/include -I//usr/local/lib/python3.12/dist-packages/torch/include

# GPU arch (override: make SM=90, SM=80, SM=89, SM=100, etc.)
SM ?= 100

# Use gencode instead of -arch to be explicit/portable
ARCHFLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)

# Toggle debug
DEBUG ?= 0

# Common NVCC flags (match your extension-ish setup)
NVCC_COMMON := -std=c++17 $(INCLUDES) --expt-relaxed-constexpr --extended-lambda -lineinfo
NVCC_COMMON += -DASYMCOMPUTE_STANDALONE=1

# Optional: fast math (disable by default for correctness debugging)
FAST_MATH ?= 0
ifeq ($(FAST_MATH),1)
  NVCC_COMMON += --use_fast_math
endif

# Optimization / debug flags
ifeq ($(DEBUG),1)
  NVCC_COMMON += -O0 -g -G
else
  NVCC_COMMON += -O2
endif

# Link flags
LDFLAGS := -L$(CUDA_HOME)/lib64 -L/usr/local/lib/python3.12/dist-packages/torch/lib -L/usr/local/cuda/lib64 -lcudart -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

# If you use other libs, uncomment as needed:
# LDFLAGS += -lcublas -lcublasLt

# You can add extra flags at build time:
# make EXTRA_NVCCFLAGS="--ptxas-options=-v"
EXTRA_NVCCFLAGS ?=

.PHONY: all clean print run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_COMMON) $(ARCHFLAGS) $(EXTRA_NVCCFLAGS) $^ -o $@ $(LDFLAGS)

print:
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "FLASH_ATTN_SRC=$(FLASH_ATTN_SRC)"
	@echo "CUTLASS_INC=$(CUTLASS_INC)"
	@echo "SM=$(SM)"
	@echo "DEBUG=$(DEBUG)"
	@echo "FAST_MATH=$(FAST_MATH)"
	@echo "EXTRA_NVCCFLAGS=$(EXTRA_NVCCFLAGS)"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

# export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH