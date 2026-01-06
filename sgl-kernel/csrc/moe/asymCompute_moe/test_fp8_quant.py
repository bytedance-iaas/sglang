import torch
import fp8_quant_ext

def ref_dynamic_fp8_quant(x, fp8_max):
    # per-tensor dynamic scale (matches your ref)
    amax = x.abs().max()
    s = (amax / fp8_max).clamp(min=1e-12)  # scalar
    y = (x / s).to(torch.float8_e4m3fn)
    return y, s

@torch.no_grad()
def main():
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(4096, 4096, device=device, dtype=torch.float16) * 0.3

    # torch.float8_e4m3fn
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    import ipdb
    ipdb.set_trace()
    
    y_ref, s_ref = ref_dynamic_fp8_quant(x, fp8_max)

    # pre-allocate outputs for extension (OUT style)
    y_ext = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s_ext = torch.empty((1,), device=device, dtype=torch.float32)  # scalar scale

    # call extension: this should WRITE into y_ext and s_ext
    # Adjust the function name to whatever you exported in PYBIND11_MODULE
    fp8_quant_ext.dynamic_scaled_fp8_quant(x, y_ext, s_ext, fp8_max)

    # compare dequantized reconstruction (common in real pipelines)
    xr_ref = y_ref.float() * s_ref
    xr_ext = y_ext.float() * s_ext[0]

    abs_max = (xr_ref - xr_ext).abs().max().item()
    abs_mean = (xr_ref - xr_ext).abs().mean().item()

    print("scale ref:", float(s_ref))
    print("scale ext:", float(s_ext))
    print("dequant diff abs_max:", abs_max)
    print("dequant diff abs_mean:", abs_mean)

if __name__ == "__main__":
    main()
