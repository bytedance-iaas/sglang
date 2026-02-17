#include <torch/torch.h>
#include <torch/script.h>        // for torch::jit::getAllOperators()
#include <dlfcn.h>
#include <iostream>
#include <cstdlib>
#include <string>

// Helper: dlopen the DeepGEMM extension .so so its static registration runs
static void load_asym_gemm_so(const char* so_path) {
    void* handle = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "dlopen failed: " << dlerror() << "\n";
        std::exit(1);
    }
    std::cout << "Loaded DeepGEMM extension: " << so_path << "\n";
}

// Helper: print all registered ops containing a substring
static void dump_ops_containing(const std::string& needle) {
    std::cout << "=== Registered ops containing '" << needle << "' ===\n";
    for (const auto& op : torch::jit::getAllOperators()) {
        const auto name = op->schema().name();
        if (name.find(needle) != std::string::npos) {
            std::cout << "  " << name << " :: " << op->schema() << "\n";
        }
    }
    std::cout << "=== end ===\n";
}

int main(int argc, char** argv) {
    // 1) Load asym_gemm._C*.so (set env var DEEP_GEMM_SO to the exact path)
    const char* so_path = std::getenv("DEEP_GEMM_SO");
    if (!so_path) {
        std::cerr
            << "Please set DEEP_GEMM_SO to the path of asym_gemm/_C*.so\n"
            << "Example:\n"
            << "  export DEEP_GEMM_SO=/path/to/site-packages/asym_gemm/_C.cpython-*.so\n";
        return 1;
    }
    load_asym_gemm_so(so_path);

    // 2) Dump op names so you can find the exact dispatcher schema name
    dump_ops_containing("asym_gemm");
    dump_ops_containing("m_grouped");

    // 3) TODO: set this to the REAL op name printed above
    // Common patterns look like:
    //   "asym_gemm::m_grouped_fp8_asym_gemm_nt_contiguous"
    // or "asym_gemm::m_grouped_fp8_asym_gemm_contiguous_1d1d"
    const std::string op_name =
        (argc >= 2) ? argv[1] : "asym_gemm::m_grouped_fp8_asym_gemm_nt_contiguous";

    // 4) Build test tensors (match your Python test as closely as possible)
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available in this build/runtime.\n";
        return 1;
    }

    const int64_t m = 8192;
    const int64_t n = 4096;
    const int64_t k = 4096;
    const int64_t num_groups = 4;
    const bool disable_ue8m0_cast = true;

    auto dev = torch::Device(torch::kCUDA, 0);

    // NOTE: FP8 creation in C++ depends on your PyTorch version.
    // Many builds support kFloat8E4M3FN directly; if yours doesn't, create fp16 and cast.
#ifdef TORCH_VERSION
#endif

    // a: [m, k] FP8
    auto a_fp16 = torch::randn({m, k}, torch::TensorOptions().device(dev).dtype(torch::kFloat16));
    auto a = a_fp16.to(torch::kFloat8E4M3FN);

    // b: simplest guess [num_groups, n, k] FP8
    auto b_fp16 = torch::randn({num_groups, n, k}, torch::TensorOptions().device(dev).dtype(torch::kFloat16));
    auto b = b_fp16.to(torch::kFloat8E4M3FN);

    // d: [m, n] FP16 output
    auto d = torch::empty({m, n}, torch::TensorOptions().device(dev).dtype(torch::kFloat16));

    // m_indices: [m] int32 (values [0..num_groups-1] or -1)
    auto m_indices = torch::randint(
        0, num_groups, {m},
        torch::TensorOptions().device(dev).dtype(torch::kInt32));
    m_indices.index_put_({torch::indexing::Slice(0, 128)}, -1);

    std::cout << "Calling op '" << op_name << "'\n";

    // 5) Call the dispatcher op (positional args only)
    // If your op signature differs, adjust the argument list accordingly.
    // Many ops are "in-place" on d and return void / None.
    auto& dispatcher = c10::Dispatcher::singleton();

    c10::OperatorHandle handle;
    try {
        handle = dispatcher.findSchemaOrThrow(op_name, "");
    } catch (const c10::Error& e) {
        std::cerr << "Could not find op schema for '" << op_name << "'.\n"
                  << "Use the printed list above to pick the correct name.\n"
                  << e.what() << "\n";
        return 1;
    }

    // Build IValue args
    std::vector<c10::IValue> args;
    args.emplace_back(a);
    args.emplace_back(b);
    args.emplace_back(d);
    args.emplace_back(m_indices);
    args.emplace_back(disable_ue8m0_cast);

    // Call
    try {
        handle.callBoxed(args);
    } catch (const c10::Error& e) {
        std::cerr << "Op call failed:\n" << e.what() << "\n";
        return 1;
    }

    // Basic sanity
    std::cout << "Done. d.mean=" << d.to(torch::kFloat32).mean().item<float>() << "\n";
    return 0;
}