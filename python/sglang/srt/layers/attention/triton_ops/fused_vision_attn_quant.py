import torch
import triton
import triton.language as tl
import json
import os

# 更新持久化配置文件名，增加 _tma 后缀
CONFIG_FILE = "/tmp/fused_vision_attn_quant_configs_tma.json"

def load_configs():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_configs(configs):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(configs, f)
    except:
        pass

# 全局配置缓存
_CONFIG_CACHE = load_configs()

@triton.autotune(
    configs=[
        # 重新精简配置空间，优先尝试较大分块
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_qkv_quant_pad_tma_kernel(
    q_ptr, k_ptr, v_ptr,
    qp_ptr, kp_ptr, vp_ptr,
    sq_ptr, sk_ptr, sv_ptr,
    M, N, aligned_N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Load scales (scalars)
    sq = tl.load(sq_ptr)
    sk = tl.load(sk_ptr)
    sv = tl.load(sv_ptr)
    
    iq = 1.0 / (sq + 1e-12)
    ik = 1.0 / (sk + 1e-12)
    iv = 1.0 / (sv + 1e-12)

    # 定义 Q, K, V 的输入 Block Pointers (TMA 核心逻辑)
    # 输入数据的形状是 (M, N)，使用边界检查
    q_in_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(M, N),
        strides=(stride_im, stride_in),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    k_in_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(M, N),
        strides=(stride_im, stride_in),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    v_in_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(M, N),
        strides=(stride_im, stride_in),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # 定义 Q, K, V 的输出 Block Pointers
    # 输出数据的形状是 (M, aligned_N)
    qp_out_block_ptr = tl.make_block_ptr(
        base=qp_ptr,
        shape=(M, aligned_N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    kp_out_block_ptr = tl.make_block_ptr(
        base=kp_ptr,
        shape=(M, aligned_N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    vp_out_block_ptr = tl.make_block_ptr(
        base=vp_ptr,
        shape=(M, aligned_N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # 使用 tl.load 异步加载数据块，Triton 3.5 在 Hopper 架构下会自动编译为 TMA 指令
    xq = tl.load(q_in_block_ptr, boundary_check=(0, 1), padding_option="zero")
    xk = tl.load(k_in_block_ptr, boundary_check=(0, 1), padding_option="zero")
    xv = tl.load(v_in_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # 量化与存储
    tl.store(qp_out_block_ptr, (xq * iq).to(tl.float8e4nv), boundary_check=(0, 1))
    tl.store(kp_out_block_ptr, (xk * ik).to(tl.float8e4nv), boundary_check=(0, 1))
    tl.store(vp_out_block_ptr, (xv * iv).to(tl.float8e4nv), boundary_check=(0, 1))

def fused_qkv_per_tensor_quant_pad(q, k, v, qp, kp, vp, sq, sk, sv):
    """
    q, k, v: (tokens, heads, head_dim) - BF16
    qp, kp, vp: (tokens, heads, aligned_head_dim) - FP8
    sq, sk, sv: (1,) - Float32
    """
    # 1. Calculate scales in Python
    sq.copy_(torch.max(torch.abs(q)) / 448.0)
    sk.copy_(torch.max(torch.abs(k)) / 448.0)
    sv.copy_(torch.max(torch.abs(v)) / 448.0)
    
    # 确保连续性，避免 TMA 访存跨页或 stride 错误
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    M = q.shape[0] * q.shape[1]
    N = q.shape[2]
    aligned_N = qp.shape[2]
    
    # 展平为 2D 方便处理
    q_2d = q.view(-1, N)
    k_2d = k.view(-1, N)
    v_2d = v.view(-1, N)
    qp_2d = qp.view(-1, aligned_N)
    kp_2d = kp.view(-1, aligned_N)
    vp_2d = vp.view(-1, aligned_N)
    
    cache_key = f"{M}_{N}"
    
    # 2. 持久化缓存检查
    if cache_key in _CONFIG_CACHE:
        best_config = _CONFIG_CACHE[cache_key]
        BLOCK_SIZE_M = best_config['BLOCK_SIZE_M']
        BLOCK_SIZE_N = best_config['BLOCK_SIZE_N']
        num_warps = best_config['num_warps']
        
        print(f"[Triton Persistence] Hit TMA Cache for {cache_key}: {best_config}", flush=True)
        
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(aligned_N, BLOCK_SIZE_N))
        
        fused_qkv_quant_pad_tma_kernel.fn[grid](
            q_2d, k_2d, v_2d, qp_2d, kp_2d, vp_2d, sq, sk, sv,
            M, N, aligned_N,
            q_2d.stride(0), q_2d.stride(1),
            qp_2d.stride(0), qp_2d.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=num_warps,
        )
    else:
        # 3. 调优
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']),
            triton.cdiv(aligned_N, META['BLOCK_SIZE_N']),
        )
        
        fused_qkv_quant_pad_tma_kernel[grid](
            q_2d, k_2d, v_2d, qp_2d, kp_2d, vp_2d, sq, sk, sv,
            M, N, aligned_N,
            q_2d.stride(0), q_2d.stride(1),
            qp_2d.stride(0), qp_2d.stride(1),
        )
        
        # 4. 保存最优配置
        best_config_obj = fused_qkv_quant_pad_tma_kernel.best_config
        _CONFIG_CACHE[cache_key] = {
            'BLOCK_SIZE_M': best_config_obj.kwargs['BLOCK_SIZE_M'],
            'BLOCK_SIZE_N': best_config_obj.kwargs['BLOCK_SIZE_N'],
            'num_warps': best_config_obj.num_warps,
        }
        print(f"[Triton TMA Tuning] {cache_key} tuned: {_CONFIG_CACHE[cache_key]}", flush=True)
        save_configs(_CONFIG_CACHE)
