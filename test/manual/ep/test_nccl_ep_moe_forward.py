import ctypes
import os
import random

import torch
import torch.distributed as dist

from sglang.test.test_deepep_utils import calc_diff, init_dist


def _import_nccl_ep():
    from nccl_ep.nccl_wrapper import (
        NCCLLibrary,
        ncclDataTypeEnum,
        ncclEpAlgorithm_t,
        ncclEpDispatchConfig_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
        get_nccl_comm_from_group,
    )

    return (
        NCCLLibrary,
        ncclDataTypeEnum,
        ncclEpAlgorithm_t,
        ncclEpDispatchConfig_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
        get_nccl_comm_from_group,
    )


def _torch_dtype_to_nccl(dtype, ncclDataTypeEnum):
    dtype_map = {
        torch.float16: ncclDataTypeEnum.ncclFloat16,
        torch.bfloat16: ncclDataTypeEnum.ncclBfloat16,
        torch.float32: ncclDataTypeEnum.ncclFloat32,
        torch.int32: ncclDataTypeEnum.ncclInt32,
        torch.int64: ncclDataTypeEnum.ncclInt64,
        torch.int8: ncclDataTypeEnum.ncclInt8,
        torch.uint8: ncclDataTypeEnum.ncclUint8,
    }
    if dtype == torch.float8_e4m3fn:
        return ncclDataTypeEnum.ncclUint8
    return dtype_map[dtype]


def _create_ep_tensor(lib, ep_group, tensor, tag, ncclDataTypeEnum):
    nccl_dtype = _torch_dtype_to_nccl(tensor.dtype, ncclDataTypeEnum)
    sizes = list(tensor.shape) + [1] * (5 - tensor.ndim)
    data = ctypes.c_void_p(tensor.data_ptr())
    return lib.ncclEpTensorCreate(
        ep_group, tensor.ndim, nccl_dtype, tag,
        data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
    )


def _make_handle_array(handles):
    n = len(handles)
    arr = (ctypes.c_void_p * n)()
    for i, h in enumerate(handles):
        arr[i] = h
    return arr, n


def _destroy_tensor_handles(lib, ep_group, handles):
    for h in handles:
        if h:
            lib.ncclEpTensorDestroy(ep_group, h)


def reference_moe_forward(
    x, topk_idx, topk_weights, expert_weights_w1, expert_weights_w2,
    num_experts, group, rank,
):
    num_ranks = dist.get_world_size(group)
    num_tokens, hidden = x.shape
    num_topk = topk_idx.shape[1]
    intermediate_size = expert_weights_w1[0].shape[0]

    all_x = [torch.empty_like(x) for _ in range(num_ranks)]
    dist.all_gather(all_x, x, group=group)
    all_x = torch.cat(all_x, dim=0)

    all_topk_idx = [torch.empty_like(topk_idx) for _ in range(num_ranks)]
    dist.all_gather(all_topk_idx, topk_idx, group=group)
    all_topk_idx = torch.cat(all_topk_idx, dim=0)

    all_topk_weights = [torch.empty_like(topk_weights) for _ in range(num_ranks)]
    dist.all_gather(all_topk_weights, topk_weights, group=group)
    all_topk_weights = torch.cat(all_topk_weights, dim=0)

    total_tokens = all_x.shape[0]
    output = torch.zeros((total_tokens, hidden), dtype=x.dtype, device=x.device)

    for token_i in range(total_tokens):
        for k in range(num_topk):
            expert_id = all_topk_idx[token_i, k].item()
            if expert_id < 0 or expert_id >= num_experts:
                continue
            weight = all_topk_weights[token_i, k].item()
            token = all_x[token_i].float()
            h = torch.nn.functional.silu(token @ expert_weights_w1[expert_id].float().T)
            h = h @ expert_weights_w2[expert_id].float().T
            output[token_i] += (h * weight).to(x.dtype)

    local_output = output[rank * num_tokens : (rank + 1) * num_tokens]
    return local_output


def test_moe_forward_ll(
    rank, num_ranks, group, lib, ep_group,
    num_tokens, hidden, num_experts, num_topk,
    num_max_dispatch_tokens_per_rank, intermediate_size, nccl_types, seed=0,
):
    (
        ncclEpAlgorithm_t,
        ncclEpDispatchConfig_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
    ) = nccl_types
    (_, ncclDataTypeEnum, _, _, _, _, _, _) = _import_nccl_ep()

    torch.manual_seed(seed)
    random.seed(seed)

    num_local_experts = num_experts // num_ranks
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    expert_weights_w1 = []
    expert_weights_w2 = []
    for _ in range(num_experts):
        w1 = torch.randn(intermediate_size, hidden, dtype=torch.bfloat16, device="cuda") * 0.02
        w2 = torch.randn(hidden, intermediate_size, dtype=torch.bfloat16, device="cuda") * 0.02
        expert_weights_w1.append(w1)
        expert_weights_w2.append(w2)

    torch.manual_seed(seed + rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(
        torch.int64
    )
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    ).abs()

    ref_output = reference_moe_forward(
        x, topk_idx, topk_weights,
        expert_weights_w1, expert_weights_w2, num_experts, group, rank,
    )

    topk_handle = _create_ep_tensor(
        lib, ep_group, topk_idx,
        ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX,
        ncclDataTypeEnum,
    )

    handle = lib.ncclEpCreateHandle(
        ep_group, topk_handle, None, stream_ptr, use_fp8=False,
    )

    dispatch_handles = []
    try:
        in_token_handle = _create_ep_tensor(
            lib, ep_group, x,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            ncclDataTypeEnum,
        )
        dispatch_handles.append(in_token_handle)

        num_recv = num_max_dispatch_tokens_per_rank * num_ranks
        out_shape = (num_local_experts, num_recv, hidden)
        out_hidden = torch.empty(out_shape, dtype=x.dtype, device="cuda")
        out_token_handle = _create_ep_tensor(
            lib, ep_group, out_hidden,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            ncclDataTypeEnum,
        )
        dispatch_handles.append(out_token_handle)

        recv_expert_counter = torch.zeros(
            num_local_experts, dtype=torch.int32, device="cuda"
        )
        recv_counter_handle = _create_ep_tensor(
            lib, ep_group, recv_expert_counter,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
            ncclDataTypeEnum,
        )
        dispatch_handles.append(recv_counter_handle)

        in_ptr, n_in = _make_handle_array([in_token_handle])
        out_ptr, n_out = _make_handle_array([out_token_handle])
        local_ptr, n_local = _make_handle_array([recv_counter_handle])

        dispatch_config = ncclEpDispatchConfig_t()
        dispatch_config.round_scales = 0

        lib.ncclEpDispatch(
            handle, in_ptr, n_in, out_ptr, n_out, local_ptr, n_local,
            0, dispatch_config, stream_ptr,
        )
        torch.cuda.synchronize()
    finally:
        _destroy_tensor_handles(lib, ep_group, dispatch_handles)

    mlp_output = torch.zeros_like(out_hidden)
    for i in range(num_local_experts):
        expert_id = rank * num_local_experts + i
        cnt = recv_expert_counter[i].item()
        if cnt > 0:
            tokens = out_hidden[i, :cnt].float()
            h = torch.nn.functional.silu(tokens @ expert_weights_w1[expert_id].float().T)
            h = h @ expert_weights_w2[expert_id].float().T
            mlp_output[i, :cnt] = h.to(x.dtype)

    combine_handles = [topk_handle]
    try:
        combine_in_handle = _create_ep_tensor(
            lib, ep_group, mlp_output,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            ncclDataTypeEnum,
        )
        combine_handles.append(combine_in_handle)

        combine_out = torch.empty(
            (num_tokens, hidden), dtype=x.dtype, device="cuda"
        )
        combine_out_handle = _create_ep_tensor(
            lib, ep_group, combine_out,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
            ncclDataTypeEnum,
        )
        combine_handles.append(combine_out_handle)

        topk_weights_handle = _create_ep_tensor(
            lib, ep_group, topk_weights,
            ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
            ncclDataTypeEnum,
        )
        combine_handles.append(topk_weights_handle)

        cin_ptr, cn_in = _make_handle_array([combine_in_handle])
        cout_ptr, cn_out = _make_handle_array([combine_out_handle])
        clocal_ptr, cn_local = _make_handle_array([topk_weights_handle])

        lib.ncclEpCombine(
            handle, cin_ptr, cn_in, cout_ptr, cn_out,
            clocal_ptr, cn_local, 0, None, stream_ptr,
        )
        torch.cuda.synchronize()
    finally:
        _destroy_tensor_handles(lib, ep_group, combine_handles)

    lib.ncclEpHandleDestroy(handle)

    diff = calc_diff(combine_out, ref_output)
    assert diff < 1e-3, (
        f"LL MoE forward: diff={diff:.6f} exceeds threshold 1e-3"
    )

    if rank == 0:
        print(f"  [rank {rank}] LL dispatch+MLP+combine diff={diff:.8f}: PASSED", flush=True)


def test_loop(local_rank, num_local_ranks):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    (
        NCCLLibrary,
        ncclDataTypeEnum,
        ncclEpAlgorithm_t,
        ncclEpDispatchConfig_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
        get_nccl_comm_from_group,
    ) = _import_nccl_ep()

    nccl_types = (
        ncclEpAlgorithm_t,
        ncclEpDispatchConfig_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
    )

    num_tokens, hidden, num_topk = 32, 2560, 4
    num_experts = (64 // num_ranks) * num_ranks
    num_max_dispatch_tokens_per_rank = 128
    intermediate_size = 512

    nccl_ep_so = os.environ.get("NCCL_EP_SO", None)
    lib = NCCLLibrary(nccl_ep_so)
    assert lib.ep_available, "NCCL EP symbols not found in library"

    comm = get_nccl_comm_from_group(group, lib)
    token_size_bytes = hidden * 2
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    if rank == 0:
        print("=" * 60)
        print("Testing dispatch + MLP + combine (LL mode) ...")
        print("=" * 60)

    ll_config = ncclEpGroupConfig_t()
    ll_config.version = 1
    ll_config.algorithm = ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY
    ll_config.num_experts = num_experts
    ll_config.max_tokens_per_rank = num_max_dispatch_tokens_per_rank
    ll_config.token_size_bytes = token_size_bytes
    ll_config.rdma_buffer_size = 0
    ll_config.num_qp_per_rank = 0
    ll_config.num_channels = 0

    ll_ep_group = lib.ncclEpCreateGroup(comm, ll_config, stream_ptr)

    test_moe_forward_ll(
        rank, num_ranks, group, lib, ll_ep_group,
        num_tokens, hidden, num_experts, num_topk,
        num_max_dispatch_tokens_per_rank, intermediate_size, nccl_types, seed=42,
    )

    lib.ncclEpGroupDestroy(ll_ep_group, stream_ptr)
    dist.barrier(group)

    if rank == 0:
        print("")
        print("=" * 60)
        print("All NCCL-EP MoE forward tests PASSED!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    num_processes = int(os.environ.get("NCCL_EP_TEST_NPROCS", "8"))
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)
