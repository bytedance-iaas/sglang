import torch

from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerBackend
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher import (
    StandardCombineInput,
    StandardDispatchOutput,
)

def main():
    torch.cuda.set_device(0)

    # match your DeepSeek layer dims
    M = 2048          # tokens
    H = 7168          # hidden size
    E = 256
    topk = 8

    a = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)

    # fake topk routing outputs (shape will depend on SGLang type)
    # Many implementations store (topk_ids, topk_weights) inside triton_topk_output
    topk_ids = torch.randint(0, E, (M, topk), device="cuda", dtype=torch.int32)
    topk_w   = torch.rand(M, topk, device="cuda", dtype=torch.float16)

    triton_topk_output = (topk_ids, topk_w, None)  # adjust to expected structure

    dispatch_output = StandardDispatchOutput(
        hidden_states=a,
        hidden_states_scale=None,
        topk_output=triton_topk_output,
    )


    # w13_weight = layer.w13_weight
    # w2_weight = layer.w2_weight

    quant_info = None  # or whatever your runner expects for fp8 scales

    # quant_info = DeepGemmMoeQuantInfo(
    #             w13_weight=w13_weight,
    #             w2_weight=w2_weight,
    #             use_fp8=True,
    #             w13_scale=w13_scale,
    #             w2_scale=w2_scale,
    #             block_shape=block_shape,
    #         )

    config = MoeRunnerConfig(inplace=False)
    runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, config)

    # warmup
    for _ in range(10):
        _ = runner.run(dispatch_output, quant_info)
        import ipdb
        ipdb.set_trace()

    torch.cuda.synchronize()
    print(out.hidden_states.shape, out.hidden_states.dtype)

if __name__ == "__main__":
    main()
