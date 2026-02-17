PYTHON ?= python3
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17

# Query PyTorch’s C++ extension helper for include/lib paths
TORCH_INC := $(shell $(PYTHON) - <<'PY'\nimport torch\nfrom torch.utils.cpp_extension import include_paths\nprint(' '.join(['-I'+p for p in include_paths()]))\nPY)
TORCH_LIB := $(shell $(PYTHON) - <<'PY'\nimport torch\nfrom torch.utils.cpp_extension import library_paths\nprint(' '.join(['-L'+p for p in library_paths()]))\nPY)

# rpath so the executable finds libtorch at runtime
TORCH_RPATH := $(shell $(PYTHON) - <<'PY'\nimport torch\nfrom torch.utils.cpp_extension import library_paths\nprint(' '.join(['-Wl,-rpath,'+p for p in library_paths()]))\nPY)

# Link libs (torch_cuda may not apparently exist on some builds; keep both patterns)
LDLIBS := -ltorch -ltorch_cpu -lc10 -ldl

# Try to link CUDA libs if available
HAS_CUDA := $(shell $(PYTHON) - <<'PY'\nimport torch\nprint(int(torch.cuda.is_available()))\nPY)

ifeq ($(HAS_CUDA),1)
  LDLIBS += -ltorch_cuda -lc10_cuda
endif

all: run_asym_gemm

run_asym_gemm: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $@ $(TORCH_INC) $(TORCH_LIB) $(TORCH_RPATH) $(LDLIBS)

clean:
	rm -f run_asym_gemm
