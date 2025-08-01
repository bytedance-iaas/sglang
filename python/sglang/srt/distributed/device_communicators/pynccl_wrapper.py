# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/pynccl.py

# This file is a pure Python wrapper for the NCCL library.
# The main purpose is to use NCCL combined with CUDA graph.
# Before writing this script, we tried the following approach:
# 1. We tried to use `cupy`, it calls NCCL correctly, but `cupy` itself
#  often gets stuck when initializing the NCCL communicator.
# 2. We tried to use `torch.distributed`, but `torch.distributed.all_reduce`
#  contains many other potential cuda APIs, that are not allowed during
#  capturing the CUDA graph. For further details, please check
# https://discuss.pytorch.org/t/pytorch-cudagraph-with-nccl-operation-failed/ .
#
# Another rejected idea is to write a C/C++ binding for NCCL. It is usually
# doable, but we often encounter issues related with nccl versions, and need
# to switch between different versions of NCCL. See
# https://github.com/NVIDIA/nccl/issues/1234 for more details.
# A C/C++ binding is not flexible enough to handle this. It requires
# recompilation of the code every time we want to switch between different
# versions. This current implementation, with a **pure** Python wrapper, is
# more flexible. We can easily switch between different versions of NCCL by
# changing the environment variable `SGLANG_NCCL_SO_PATH`, or the `so_file`
# variable in the code.

import ctypes
import logging
import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)


def find_nccl_library() -> str:
    """
    We either use the library file specified by the `SGLANG_NCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libnccl.so.2` or `librccl.so.1` can be
    found by `ctypes` automatically.
    """

    # so_file can be set to None in sglang
    so_file = os.environ.get("SGLANG_NCCL_SO_PATH", None)

    # manually load the nccl library
    if so_file:
        logger.info(
            "Found nccl from environment variable SGLANG_NCCL_SO_PATH=%s", so_file
        )
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.debug("Found nccl from library %s", so_file)
    return so_file


# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p
ncclWindow_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

ncclDataType_t = ctypes.c_int


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


ncclRedOp_t = ctypes.c_int


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.ncclSum
        if op == ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == ReduceOp.MAX:
            return cls.ncclMax
        if op == ReduceOp.MIN:
            return cls.ncclMin
        if op == ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class NCCLLibrary:
    exported_functions = [
        # const char* ncclGetErrorString(ncclResult_t result)
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        # ncclResult_t  ncclGetVersion(int *version);
        Function("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        Function("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        # ncclResult_t  ncclCommInitRank(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        # note that ncclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function(
            "ncclCommInitRank",
            ncclResult_t,
            [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int],
        ),
        # ncclResult_t  ncclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ncclAllReduce",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclAllGather(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ncclAllGather",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclReduceScatter(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ncclReduceScatter",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclSend(
        #   const void* sendbuff, size_t count, ncclDataType_t datatype,
        #   int dest, ncclComm_t comm, cudaStream_t stream);
        Function(
            "ncclSend",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t  ncclRecv(
        #   void* recvbuff, size_t count, ncclDataType_t datatype,
        #   int src, ncclComm_t comm, cudaStream_t stream);
        Function(
            "ncclRecv",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t ncclBroadcast(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, int root, ncclComm_t comm,
        #   cudaStream_t stream);
        Function(
            "ncclBroadcast",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # ncclResult_t  ncclCommDestroy(ncclComm_t comm);
        Function("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
    ]

    exported_functions_symm_mem = [
        # ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);
        Function(
            "ncclCommWindowRegister",
            ncclResult_t,
            [
                ncclComm_t,
                buffer_type,
                ctypes.c_size_t,
                ctypes.POINTER(ncclWindow_t),
                ctypes.c_int,
            ],
        ),
        # ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);
        Function("ncclCommWindowDeregister", ncclResult_t, [ncclComm_t, ncclWindow_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        so_file = so_file or find_nccl_library()

        try:
            if so_file not in NCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                NCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = NCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load NCCL library from %s ."
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise, the nccl library might not exist, be corrupted "
                "or it does not support the current platform %s."
                "If you already have the library, please set the "
                "environment variable SGLANG_NCCL_SO_PATH"
                " to point to the correct nccl library path.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            exported_functions = NCCLLibrary.exported_functions
            if hasattr(self.lib, "ncclCommWindowRegister"):
                exported_functions.extend(NCCLLibrary.exported_functions_symm_mem)
            for func in exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            NCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["ncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetRawVersion(self) -> int:
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["ncclGetVersion"](ctypes.byref(version)))
        # something like 21903
        return version.value

    def ncclGetVersion(self) -> str:
        version_str = str(self.ncclGetRawVersion())
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def ncclCommInitRank(
        self, world_size: int, unique_id: ncclUniqueId, rank: int
    ) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(
            self._funcs["ncclCommInitRank"](
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def ncclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["ncclAllReduce"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["ncclReduceScatter"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(
            self._funcs["ncclAllGather"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def ncclSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclSend"](sendbuff, count, datatype, dest, comm, stream)
        )

    def ncclRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclRecv"](recvbuff, count, datatype, src, comm, stream)
        )

    def ncclBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclBroadcast"](
                sendbuff, recvbuff, count, datatype, root, comm, stream
            )
        )

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclCommDestroy"](comm))

    def ncclCommWindowRegister(
        self, comm: ncclComm_t, buff: buffer_type, size: int, win_flags: int
    ) -> ncclWindow_t:
        window = ncclWindow_t()
        self.NCCL_CHECK(
            self._funcs["ncclCommWindowRegister"](
                comm, buff, size, ctypes.byref(window), win_flags
            )
        )
        return window

    def ncclCommWindowDeregister(self, comm: ncclComm_t, window: ncclWindow_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclCommWindowDeregister"](comm, window))


__all__ = [
    "NCCLLibrary",
    "ncclDataTypeEnum",
    "ncclRedOpTypeEnum",
    "ncclUniqueId",
    "ncclComm_t",
    "cudaStream_t",
    "buffer_type",
]
