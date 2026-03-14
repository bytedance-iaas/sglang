import torch
import triton
from sglang.srt.layers.moe.ep_moe.kernels import ep_scatter

def test_ep_scatter():
    hidden_size = 4096
    all_tokens = 3
    num_experts = 4
    
    recv_x = torch.randn(all_tokens, hidden_size, dtype=torch.float8_e4m3fn, device="cuda")
    recv_x_scale = torch.randn(all_tokens, hidden_size // 128, dtype=torch.float32, device="cuda")
    
    recv_topk = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32, device="cuda")
    num_recv_tokens_per_expert = torch.tensor([3, 3, 3, 3], dtype=torch.int32, device="cuda")
    
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)
    
    input_tensor = torch.empty((12, hidden_size), dtype=torch.bfloat16, device="cuda")
    input_tensor_scale = torch.empty((12, hidden_size // 128), dtype=torch.float32, device="cuda")
    
    m_indices = torch.empty(12, dtype=torch.int32, device="cuda")
    output_index = torch.empty_like(recv_topk)
    
    ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk,
        num_recv_tokens_per_expert,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=False,
    )
    
    print("input_tensor[0][:5]:", input_tensor[0][:5])
    print("input_tensor[0][1]:", input_tensor[0][1].item())
    print("input_tensor[1][0]:", input_tensor[1][0].item())
    print("input_tensor[1][1]:", input_tensor[1][1].item())

if __name__ == "__main__":
    test_ep_scatter()
