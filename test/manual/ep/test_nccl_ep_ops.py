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
    """Create an opaque ncclNDTensor_t handle wrapping a PyTorch tensor."""
    ndim = tensor.ndim
    dtype = _torch_dtype_to_nccl(tensor.dtype, ncclDataTypeEnum)
    data = ctypes.c_void_p(tensor.data_ptr())
    sizes = list(tensor.shape) + [1] * (5 - ndim)
    return lib.ncclEpTensorCreate(
        ep_group, ndim, dtype, tag, data,
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
    )


def _make_handle_array(handles):
    """Create a C array of opaque tensor handles."""
    n = len(handles)
    if n == 0:
        return None, 0
    arr = (ctypes.c_void_p * n)()
    for i, h in enumerate(handles):
        arr[i] = h
    return arr, n


def _destroy_tensor_handles(lib, ep_group, handles):
    """Destroy a list of ncclNDTensor_t handles."""
    for h in handles:
        if h:
            lib.ncclEpTensorDestroy(ep_group, h)


def per_token_cast_to_fp8(x):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def per_token_cast_back(x_fp8, x_scales):
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


def test_low_latency(
    rank,
    num_ranks,
    group,
    lib,
    ep_group,
    num_tokens,
    hidden,
    num_experts,
    num_topk,
    num_max_dispatch_tokens_per_rank,
    nccl_types,
    seed=0,
):
    (
        ncclEpAlgorithm_t,
        ncclEpDispatchConfig_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
    ) = nccl_types
    (
        _,
        ncclDataTypeEnum,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _import_nccl_ep()

    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    num_local_experts = num_experts // num_ranks
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

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

    for use_fp8 in (False, True):
        # LL input must always be BF16; the C kernel handles BF16->FP8 internally
        dispatch_hidden = x

        handles_to_destroy = []
        try:
            topk_handle = _create_ep_tensor(
                lib, ep_group, topk_idx,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOPK_IDX,
                ncclDataTypeEnum,
            )

            handle = lib.ncclEpCreateHandle(
                ep_group, topk_handle, None, stream_ptr, use_fp8=use_fp8
            )

            in_token_handle = _create_ep_tensor(
                lib, ep_group, dispatch_hidden,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
                ncclDataTypeEnum,
            )
            handles_to_destroy.append(in_token_handle)

            num_recv = num_max_dispatch_tokens_per_rank * num_ranks
            out_dtype = torch.float8_e4m3fn if use_fp8 else dispatch_hidden.dtype
            out_shape = (num_local_experts, num_recv, dispatch_hidden.shape[-1])
            out_hidden = torch.empty(
                out_shape, dtype=out_dtype, device="cuda"
            )
            out_token_handle = _create_ep_tensor(
                lib, ep_group, out_hidden,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
                ncclDataTypeEnum,
            )
            handles_to_destroy.append(out_token_handle)

            out_handles = [out_token_handle]

            out_scale = None
            if use_fp8:
                scale_shape = (num_local_experts, num_recv, hidden // 128)
                out_scale = torch.empty(scale_shape, dtype=torch.float32, device="cuda")
                out_scale_handle = _create_ep_tensor(
                    lib, ep_group, out_scale,
                    ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_SCALES,
                    ncclDataTypeEnum,
                )
                out_handles.append(out_scale_handle)
                handles_to_destroy.append(out_scale_handle)

            recv_expert_counter = torch.zeros(
                num_local_experts, dtype=torch.int32, device="cuda"
            )
            recv_counter_handle = _create_ep_tensor(
                lib, ep_group, recv_expert_counter,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                ncclDataTypeEnum,
            )
            handles_to_destroy.append(recv_counter_handle)

            in_ptr, n_in = _make_handle_array([in_token_handle])
            out_ptr, n_out = _make_handle_array(out_handles)
            local_ptr, n_local = _make_handle_array([recv_counter_handle])

            dispatch_config = ncclEpDispatchConfig_t()
            dispatch_config.round_scales = 0

            lib.ncclEpDispatch(
                handle, in_ptr, n_in, out_ptr, n_out, local_ptr, n_local,
                0, dispatch_config, stream_ptr,
            )
            torch.cuda.synchronize()
        finally:
            _destroy_tensor_handles(lib, ep_group, handles_to_destroy)

        all_topk_idx = torch.empty(
            (num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device="cuda"
        )
        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

        for i in range(num_local_experts):
            expert_id = rank * num_local_experts + i
            expected_count = (all_topk_idx == expert_id).sum().item()
            actual_count = recv_expert_counter[i].item()
            assert actual_count == expected_count, (
                f"LL dispatch expert {expert_id}: expected {expected_count} tokens, "
                f"got {actual_count} (use_fp8={use_fp8})"
            )

        simulated_out = out_hidden.to(torch.bfloat16)
        if use_fp8 and out_scale is not None:
            for ei in range(num_local_experts):
                cnt = recv_expert_counter[ei].item()
                if cnt > 0:
                    simulated_out[ei, :cnt] = per_token_cast_back(
                        out_hidden[ei, :cnt], out_scale[ei, :cnt]
                    )

        combine_handles = [topk_handle]
        try:
            combine_in_handle = _create_ep_tensor(
                lib, ep_group, simulated_out,
                ncclEpTensorTag_t.NCCL_EP_TENSOR_TAG_TOKENS,
                ncclDataTypeEnum,
            )
            combine_handles.append(combine_in_handle)

            combine_out = torch.empty(
                (num_tokens, hidden), dtype=simulated_out.dtype, device="cuda"
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

        assert not torch.isnan(combine_out).any(), (
            f"LL combine produced NaN (use_fp8={use_fp8})"
        )

        if rank == 0:
            print(
                f"  [rank {rank}] LL {'FP8' if use_fp8 else 'BF16'} dispatch+combine: PASSED",
                flush=True,
            )


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

    num_tokens, hidden, num_topk, num_experts = 64, 5120, 8, (256 // num_ranks) * num_ranks
    num_max_dispatch_tokens_per_rank = 128

    nccl_ep_so = os.environ.get("NCCL_EP_SO", None)
    lib = NCCLLibrary(nccl_ep_so)
    assert lib.ep_available, "NCCL EP symbols not found in library"

    comm = get_nccl_comm_from_group(group, lib)

    if rank == 0:
        print("=" * 60)
        print("Testing NCCL-EP Low-Latency (LL) mode ...")
        print("=" * 60)

    token_size_bytes = hidden * 2
    ll_config = ncclEpGroupConfig_t()
    ll_config.version = 1
    ll_config.algorithm = ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY
    ll_config.num_experts = num_experts
    ll_config.max_tokens_per_rank = num_max_dispatch_tokens_per_rank
    ll_config.token_size_bytes = token_size_bytes
    ll_config.rdma_buffer_size = 0
    ll_config.num_qp_per_rank = 0
    ll_config.num_channels = 0

    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    ll_ep_group = lib.ncclEpCreateGroup(comm, ll_config, stream_ptr)

    test_low_latency(
        rank, num_ranks, group, lib, ll_ep_group,
        num_tokens, hidden, num_experts, num_topk,
        num_max_dispatch_tokens_per_rank, nccl_types, seed=42,
    )

    lib.ncclEpGroupDestroy(ll_ep_group, stream_ptr)
    dist.barrier(group)

    if rank == 0:
        print("")
        print("=" * 60)
        print("All NCCL-EP operator-level tests PASSED!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    num_processes = int(os.environ.get("NCCL_EP_TEST_NPROCS", "8"))
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)
