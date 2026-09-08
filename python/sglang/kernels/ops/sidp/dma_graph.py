"""Conditional DMA: standalone eager graphs or in-place model capture."""

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def load_sidp_dma_graph_module():
    # A separate translation unit keeps SWITCH/CUDA-version requirements out
    # of the default DMA and SM backends. CUDA 13 compiles the conditional
    # setter with the standard JIT whole-program compilation path.
    return load_jit(
        "sidp_dma_graph",
        cuda_files=["sidp/dma_graph.cuh"],
        cuda_wrappers=[
            ("create", "&sidp::SidpDmaGraphKernels::build<false>"),
            ("append_cycle_to_capture", "&sidp::SidpDmaGraphKernels::build<true>"),
            ("launch", "&sidp::SidpDmaGraphKernels::launch"),
            ("destroy", "&sidp::SidpDmaGraphKernels::destroy"),
        ],
    )
