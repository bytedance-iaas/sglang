"""Small C-ABI helpers shared by the migration planner."""

import ctypes as C

import torch


def ptr(tensor):
    return C.c_void_p(tensor.data_ptr())


def stream_ptr():
    return C.c_void_p(torch.cuda.current_stream().cuda_stream)


def check(code):
    if code:
        raise RuntimeError(f"IAAS migration planner CUDA error {code}")
