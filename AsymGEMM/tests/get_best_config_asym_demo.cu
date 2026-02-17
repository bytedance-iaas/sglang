#include "../csrc/jit_kernels/heuristics/sm100.hpp"
#include "../csrc/jit/device_runtime.hpp"

int main() {
    using namespace asym_gemm;
    auto cfg = get_best_config<SM100ArchSpec>(
        GemmType::MGroupedContiguous,
        KernelType::Kernel1D1D,
        /*m=*/1024, /*n=*/1024, /*k=*/1024, /*num_groups=*/1,
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        torch::kBFloat16, torch::kBFloat16,
        /*with_accumulation=*/false,
        /*num_sms=*/device_runtime->get_num_sms()
    );
    printf("block_m=%d block_n=%d block_k=%d stages=%d\n",
           cfg.block_m, cfg.block_n, cfg.block_k, cfg.num_stages);
    return 0;
}
