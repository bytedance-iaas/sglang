from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import triton
import triton.language as tl

from sglang.srt.custom_op import CustomOp
from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_int_env_var,
    is_cpu,
    is_hip,
    set_weight_attrs,
    use_intel_amx_backend,
)

from sglang.srt.utils.kernel_selector import get_kernel_selector, append_string_if_not_exists
from logging import Logger


if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

L20_DEVICE_NAME = 'NVIDIA L20'
H20_DEVICE_NAME = 'NVIDIA H20'

DEEP_GEMM_AVAILABLE = True
try:
    from deep_gemm import bf16_gemm_nt
except:
    DEEP_GEMM_AVAILABLE = False

_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight



def deep_gemm_matmul(a, b):
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = b.shape
    d = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    if a.dtype == torch.bfloat16:
        bf16_gemm_nt(a, b, d, c=None)
    return d


@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        bias_ptr,
        M, N, K,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):

    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    
    # we assume that each program_id is  solving a  (BLOCK_SIZE_M, BLOCK_SIZE_N) size problem
    # then pid will range from  0 ~ (M/ BLOCK_SIZE_M) *  (N/BLOCK_SIZE_N)
    # size_m = M / BLOCK_SIZE__M
    # size_n = N / BLOCK_SIZE_N
    
    # here we need to get every  prgram  will deal which block in output
    # in a normal case: can just use  pid // size_n, pid % size_n
    # then can compute each block's start as (pid // size_n *  BLOCK_SIZE_M) * N, (pid * size_n) * BLOCK_SIZE_N
    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    # now access block in zig-zag order: need to decide blong to which group first:
    
    
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # decide this group has a total group size or less than group_size_m
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m) # get which row the block was in (in a size_m * size_n matrix)
    pid_n = (pid % num_pid_in_group) // group_size_m # get which column the block was in (in a size_m * size_n matrix)

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
     
    # this is for all threads in the block , so we need to use a tl.arrange to  get an offset for every threads in the block
    # in case of OOB need to %M
    # the same for col (offset_bn)
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M 
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    
    # since we use a size_k tile each time need to plus this stride in a/b matrix
    offs_k = tl.arange(0, BLOCK_SIZE_K) 
    
    
    # here we not only compute a  pointer's start (address) but also need to express a (BLOCK_SIZE_M, BLOCK_SIZE_K) input matrix (also for B)
    # so use None to get (BLOCK_SIZE_M,1) also use None to get (1, BLOLCK_SIZE_K) and the final size will be (BLOCK_SIZE_M, BLOCK_SIZE_K) tile
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    
    # used to store output
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        
        # will load a (BLOCK_SIZE_M, BLOCK_SIZE_K) tile automatically
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # will load a (BLOCK_SIZE_K, BLOCK_SIZE_N) tile automatically
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        
        # only need to move left-top corner of pointer
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!

    if ACTIVATION == "add_bias":
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs)
        # add none @ need to broadcast dim
        accumulator+=bias[None, :]
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_matmul(a, b, bias = None, launch_configs= [64,64,64,8,8,4,1]):
    if len(launch_configs)!= 7:
        Logger.info("invalid luanch config , use default config")
        launch_configs =  [64,64,64,8,8,4,1]
    
    activation = ""
    if not (bias is None):
        activation = "add_bias"
      
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    need_unsequeeze = False
    squeeze_position = 0
    if len(a.shape) == 3:
        if a.shape[0] == 1:
            squeeze_position = 0
        elif a.shape[1] == 1:
            squeeze_position = 1
        a = a.squeeze(squeeze_position)
        need_unsequeeze = True
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16).contiguous()
    # print("c.shape {}".format(c.shape))
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    #BLOCK_SIZE_M: 512, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 8, num_ctas: 1, num_stages: 4
    matmul_kernel[grid](
        a, b, c, bias,#
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(1), b.stride(0),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=launch_configs[0],
        BLOCK_SIZE_N=launch_configs[1],
        BLOCK_SIZE_K= launch_configs[2],
        GROUP_SIZE_M= launch_configs[3],
        ACTIVATION=activation,
        num_warps = launch_configs[4],
        num_stages = launch_configs[5],
        num_ctas = launch_configs[6],
    )
    
    if need_unsequeeze:
        c.unsqueeze_(squeeze_position)
    return c

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.jit
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    FLATTEN: tl.constexpr,
    ACTIVATION: tl.constexpr,
    
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    tl.assume(num_pid_m >= 0)
    tl.assume(num_pid_n >= 0)


    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        

            
        tile_id_c += NUM_SMS
        #GLOBAL
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N
        
        offs_bn_bias = (pid_n * BLOCK_SIZE_N + tl.range(0, BLOCK_SIZE_N)) % N
        if ACTIVATION == "add_bias":
            bias_ptrs = bias_ptr + offs_bn_bias
            bias = tl.load(bias_ptrs)
            accumulator+=bias[None, :]

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


def triton_matmul_persistent_tma(is_hopper, sm_num, a, b, bias = None, launch_configs= [64,64,64,8, False, True, 4, 4, 1]):
    if len(launch_configs)!= 9:
        Logger.info("invalid luanch config , use default config")
        launch_configs =  [64,64,64,8, False, True, 4, 4, 1]
    
    
    activation = ""
    if not (bias is None):
        activation = "add_bias"
      
    assert a.is_contiguous(), "Matrix A must be contiguous"
    need_unsequeeze = False
    squeeze_position = 0
    if len(a.shape) == 3:
        if a.shape[0] == 1:
            squeeze_position = 0
        elif a.shape[1] == 1:
            squeeze_position = 1
        a = a.squeeze(squeeze_position)
        need_unsequeeze = True
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16).contiguous()
    # print("c.shape {}".format(c.shape))
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    #BLOCK_SIZE_M: 512, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8, num_warps: 8, num_ctas: 1, num_stages: 4
    
    warp_specialize = launch_configs[5]
    flatten = False if (warp_specialize and is_hopper) else True
    matmul_kernel_descriptor_persistent[grid](
        a, b, c, bias,#
        M, N, K,  ##
        BLOCK_SIZE_M=launch_configs[0],
        BLOCK_SIZE_N=launch_configs[1],
        BLOCK_SIZE_K= launch_configs[2],
        GROUP_SIZE_M= launch_configs[3],
        EPILOGUE_SUBTILE= launch_configs[4],
        WARP_SPECIALIZE= launch_configs[5],
        num_warps = launch_configs[6],
        num_stages = launch_configs[7],
        num_ctas = launch_configs[8],
        NUM_SMS= sm_num,
        FLATTEN= flatten,
        ACTIVATION= activation
    )
    
    if need_unsequeeze:
        c.unsqueeze_(squeeze_position)
    return c

class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output
        bias_shape = "none" if  bias is None else bias.shape
        # print("check_shape weight {} input {} bias {} ".format(layer.weight.shape, x.shape,  bias_shape))
        
        shape_collect = []
        if len(x.shape) == 3:
            for s in x.shape:
                if s==1:
                    continue
                shape_collect.append(s)
        elif len(x.shape) == 2 :
            shape_collect = x.shape
        
        do_kernel_selection = get_int_env_var("SGLANG_GEMM_KERNEL_SELC")
        
        if len(shape_collect) == 2  and do_kernel_selection:
            # print("try to opt")
            if not hasattr(self, "device_name"):
                setattr(self, "device_name", torch.cuda.get_device_name())
                
            kernel_selector = get_kernel_selector()
            
            is_h20 = self.device_name == H20_DEVICE_NAME
            is_l20 = self.device_name == L20_DEVICE_NAME
            if not hasattr(self, "is_hopper"):
                is_hopper = torch.cuda.get_device_capability()[0] == 9
                setattr(self, "is_hopper", is_hopper)
            
            if not hasattr(self, "sm_num"):
                sm_num = torch.cuda.get_device_properties("cuda").multi_processor_count
                setattr(self, "sm_num", sm_num)
                
            # if not hasattr(self, "shape_table"):
            #     shape_table = set()
            #     setattr(self, "shape_table", shape_table)
                
            hash_device_name = self.device_name.split(" ")
            if hash_device_name[0] == "NVIDIA" and len(hash_device_name) == 2:
                hash_device_name = hash_device_name[1]
            
            
            hash_dtype = ""
            if x.dtype == torch.float32:
                hash_dtype = "FP32"
            elif x.dtype == torch.bfloat16:
                hash_dtype = "BF16"
            elif x.dtype == torch.float16:
                hash_dtype = "FP16"
            
            M, N, K = shape_collect[0], layer.weight.shape[0], shape_collect[1]    
            
            call_in_graph = torch.cuda.is_current_stream_capturing()

            if get_bool_env_var("SGLANG_RECORD_TUNE"):
                save_path = os.getenv("gemm_tune_record_path")
                file_name = "GEMM_graph_{}_BF16.txt".format(call_in_graph) 
                file_name = os.path.join(save_path, file_name)
                
                to_save_mnk ="{}_{}_{}".format(M, N, K)
                append_string_if_not_exists(file_name, to_save_mnk)
                
            
            # ori_len = len(self.shape_table)
            # self.shape_table.add((M,N,K))
            # after_len = len(self.shape_table)
            
            # if ori_len !=after_len:
            #     print("all shape {}".format(self.shape_table))
            selection_rets = kernel_selector.query_kernel_data(hash_device_name, (M,N,K), hash_dtype, "GEMM", call_in_graph)
            # all case use default
            if not isinstance(selection_rets, list) or len(selection_rets) == 0:
                pass
            else:
                selection_ret = selection_rets[-1]
                kernel_type = selection_ret["kernel_type"]
                kernel_config = selection_ret["kernel_config"]
                
                if kernel_type == "deep_gemm":
                    if (bias is None) and DEEP_GEMM_AVAILABLE:
                        return deep_gemm_matmul(x, layer.weight)
                    else:
                        if len(selection_rets)!=0:
                            selection_ret = selection_rets[-2]
                            kernel_type = selection_ret["kernel_type"]
                            kernel_config = selection_ret["kernel_config"]
                
                if kernel_type == "triton":
                    print("p")
                    return triton_matmul(x, layer.weight, bias, kernel_config)
                elif kernel_type == "triton_tma":
                    return triton_matmul_persistent_tma(self.is_hopper, self.sum_num, x, layer.weight, bias, kernel_config)
        # else:
            # print("check this case x.shape {} shape_collect {}".format(x.shape, shape_collect))                
        print("F")
        ret =  F.linear(x, layer.weight, bias)
        # print("ret shape {}".format(ret.shape))
        return ret


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, use_triton_kernels: bool = False):
        super().__init__()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_weight_n, w13_weight_k = 2 * intermediate_size_per_partition, hidden_size
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _use_aiter:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

        return

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        backend = (
            MoeRunnerBackend.TRITON_KERNELS
            if self.use_triton_kernels
            else MoeRunnerBackend.TRITON
        )
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            if _use_aiter:
                assert not moe_runner_config.no_combine, "unsupported"
                topk_weights, topk_ids, _ = topk_output
                if moe_runner_config.apply_router_weight_on_input:
                    assert (
                        topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
                    _, topk = topk_weights.shape
                    assert (
                        topk == 1
                    ), "Only support topk=1 when `apply_router_weight_on_input` is True"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(
                        topk_weights, dtype=torch.float32
                    )  # topk_weights must be FP32 (float32)
                output = fused_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation=(
                        ActivationType.Silu
                        if moe_runner_config.activation == "silu"
                        else ActivationType.Gelu
                    ),
                )
                return StandardCombineInput(hidden_states=output)
            else:
                quant_info = TritonMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    b13=getattr(layer, "w13_weight_bias", None),
                    b2=getattr(layer, "w2_weight_bias", None),
                )
                return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if (
            use_intel_amx_backend(layer)
            and not moe_runner_config.apply_router_weight_on_input
        ):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                False,  # use_int8_w8a8
                False,  # use_fp8_w8a16
                None,  # w1_scale
                None,  # w2_scale
                None,  # block_size
                None,  # a1_scale
                None,  # a2_scale
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        import torch_npu

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)
        num_experts = layer.num_experts
        top_k = layer.top_k
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
            .view(top_k, -1)
            .permute(1, 0)
            .contiguous()
        )

        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch_npu.npu_moe_init_routing(
                x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
            )
        )

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )

        expert_tokens = expert_tokens.to(torch.int64)
        if layer.w13_weight.shape[-1] == layer.hidden_size:
            w13 = layer.w13_weight.transpose(1, 2)
            w2 = layer.w2_weight.transpose(1, 2)

        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w13],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # act_fn:
        if self.moe_runner_config.activation == "silu":
            hidden_states = torch_npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul

            hidden_states = GeluAndMul()(hidden_states)

        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )

        return StandardCombineInput(hidden_states=final_hidden_states)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cpu
