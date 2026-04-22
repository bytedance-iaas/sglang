from typing import Union

import msgspec
from zmq import Socket
from zmq.asyncio import Socket as AsyncSocket

from sglang.srt.environ import envs

from .pickle_struct import (
    AbortReq,
    ActiveRanksOutput,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BackupDramReq,
    BaseBatchReq,
    BaseReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    BlockReqType,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DisaggregationMetrics,
    DumperControlReqInput,
    DumperControlReqOutput,
    EmbeddingReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    Function,
    GenerateReqInput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    LazyDumpTensorsReqInput,
    LazyDumpTensorsReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    LoRAMetrics,
    LoRAUpdateOutput,
    MemoryMetrics,
    MultimodalDataInputFormat,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ParseFunctionCallReq,
    PauseGenerationReqInput,
    ProfileReq,
    ProfileReqInput,
    ProfileReqOutput,
    ProfileReqType,
    QueueMetrics,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SeparateReasoningReqInput,
    SessionParams,
    SetInjectDumpMetadataReqInput,
    SetInjectDumpMetadataReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    SpeculativeDecodingMetricsMixin,
    SpeculativeMetrics,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    Tool,
    UnloadLoRAAdapterReqInput,
    UpdateExpertBackupReq,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    UpdateWeightVersionReqInput,
    VertexGenerateReqInput,
    WatchLoadUpdateReq,
)

LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput


PICKLE_MAGIC_NUMBER = b"0xSG01"
MSGPACK_MAGIC_NUMBER = b"0xSG02"

if envs.SGLANG_IPC_USE_MSGPACK:
    from .msgpack_struct import FreezeGCReq
else:
    from .pickle_struct import FreezeGCReq


def sock_send(
    socket: Socket, obj: Union[BaseReq, BaseBatchReq, msgspec.Struct], flags=0
):
    # if the msgpack magic number is not used, fallback to pickle
    if not envs.SGLANG_IPC_USE_MSGPACK:
        socket.send_pyobj(obj, flags=flags)
        return

    if isinstance(obj, msgspec.Struct):
        from .msgpack_struct import serialize

        socket.send_multipart([MSGPACK_MAGIC_NUMBER, serialize(obj)], flags=flags)
    else:
        from .pickle_struct import serialize

        socket.send_multipart([PICKLE_MAGIC_NUMBER, serialize(obj)], flags=flags)


def sock_recv(socket: Socket, flags=0) -> Union[BaseReq, BaseBatchReq, msgspec.Struct]:
    if not envs.SGLANG_IPC_USE_MSGPACK:
        return socket.recv_pyobj(flags=flags)

    magic_number, data = socket.recv_multipart(flags=flags)
    if magic_number == MSGPACK_MAGIC_NUMBER:
        from .msgpack_struct import deserialize

        return deserialize(data)
    else:
        from .pickle_struct import deserialize

        return deserialize(data)


async def sock_send_async(
    socket: AsyncSocket, obj: Union[BaseReq, BaseBatchReq, msgspec.Struct], flags=0
):
    if not envs.SGLANG_IPC_USE_MSGPACK:
        await socket.send_pyobj(obj, flags=flags)
        return

    if isinstance(obj, msgspec.Struct):
        from .msgpack_struct import serialize

        await socket.send_multipart([MSGPACK_MAGIC_NUMBER, serialize(obj)], flags=flags)
    else:
        from .pickle_struct import serialize

        await socket.send_multipart([PICKLE_MAGIC_NUMBER, serialize(obj)], flags=flags)


async def sock_recv_async(
    socket: AsyncSocket, flags=0
) -> Union[BaseReq, BaseBatchReq, msgspec.Struct]:
    if not envs.SGLANG_IPC_USE_MSGPACK:
        return await socket.recv_pyobj(flags=flags)

    magic_number, data = await socket.recv_multipart(flags=flags)
    if magic_number == MSGPACK_MAGIC_NUMBER:
        from .msgpack_struct import deserialize

        return deserialize(data)
    else:
        from .pickle_struct import deserialize

        return deserialize(data)


__all__ = [
    "BaseReq",
    "BaseBatchReq",
    "SpeculativeDecodingMetricsMixin",
    "SessionParams",
    "GenerateReqInput",
    "TokenizedGenerateReqInput",
    "BatchTokenizedGenerateReqInput",
    "EmbeddingReqInput",
    "TokenizedEmbeddingReqInput",
    "BatchTokenizedEmbeddingReqInput",
    "BatchTokenIDOutput",
    "BatchStrOutput",
    "BatchEmbeddingOutput",
    "ClearHiCacheReqInput",
    "ClearHiCacheReqOutput",
    "FlushCacheReqInput",
    "FlushCacheReqOutput",
    "AddExternalCorpusReqInput",
    "AddExternalCorpusReqOutput",
    "RemoveExternalCorpusReqInput",
    "RemoveExternalCorpusReqOutput",
    "ListExternalCorporaReqInput",
    "ListExternalCorporaReqOutput",
    "AttachHiCacheStorageReqInput",
    "AttachHiCacheStorageReqOutput",
    "DetachHiCacheStorageReqInput",
    "DetachHiCacheStorageReqOutput",
    "PauseGenerationReqInput",
    "ContinueGenerationReqInput",
    "UpdateWeightFromDiskReqInput",
    "UpdateWeightFromDiskReqOutput",
    "UpdateWeightsFromDistributedReqInput",
    "UpdateWeightsFromDistributedReqOutput",
    "UpdateWeightsFromTensorReqInput",
    "UpdateWeightsFromTensorReqOutput",
    "InitWeightsSendGroupForRemoteInstanceReqInput",
    "UpdateWeightsFromIPCReqInput",
    "UpdateWeightsFromIPCReqOutput",
    "InitWeightsSendGroupForRemoteInstanceReqOutput",
    "SendWeightsToRemoteInstanceReqInput",
    "SendWeightsToRemoteInstanceReqOutput",
    "UpdateExpertBackupReq",
    "BackupDramReq",
    "InitWeightsUpdateGroupReqInput",
    "InitWeightsUpdateGroupReqOutput",
    "DestroyWeightsUpdateGroupReqInput",
    "DestroyWeightsUpdateGroupReqOutput",
    "UpdateWeightVersionReqInput",
    "GetWeightsByNameReqInput",
    "GetWeightsByNameReqOutput",
    "ReleaseMemoryOccupationReqInput",
    "ReleaseMemoryOccupationReqOutput",
    "ResumeMemoryOccupationReqInput",
    "ResumeMemoryOccupationReqOutput",
    "CheckWeightsReqInput",
    "CheckWeightsReqOutput",
    "SlowDownReqInput",
    "SlowDownReqOutput",
    "AbortReq",
    "ActiveRanksOutput",
    "GetInternalStateReq",
    "GetInternalStateReqOutput",
    "SetInternalStateReq",
    "SetInternalStateReqOutput",
    "ProfileReqInput",
    "ProfileReqType",
    "ProfileReq",
    "ProfileReqOutput",
    "FreezeGCReq",
    "ConfigureLoggingReq",
    "OpenSessionReqInput",
    "CloseSessionReqInput",
    "OpenSessionReqOutput",
    "HealthCheckOutput",
    "ExpertDistributionReqType",
    "ExpertDistributionReq",
    "ExpertDistributionReqOutput",
    "Function",
    "Tool",
    "ParseFunctionCallReq",
    "SeparateReasoningReqInput",
    "VertexGenerateReqInput",
    "RpcReqInput",
    "RpcReqOutput",
    "LoadLoRAAdapterReqInput",
    "UnloadLoRAAdapterReqInput",
    "LoadLoRAAdapterFromTensorsReqInput",
    "LoRAUpdateOutput",
    "BlockReqType",
    "BlockReqInput",
    "MemoryMetrics",
    "SpeculativeMetrics",
    "LoRAMetrics",
    "DisaggregationMetrics",
    "QueueMetrics",
    "GetLoadsReqInput",
    "GetLoadsReqOutput",
    "WatchLoadUpdateReq",
    "SetInjectDumpMetadataReqInput",
    "SetInjectDumpMetadataReqOutput",
    "LazyDumpTensorsReqInput",
    "LazyDumpTensorsReqOutput",
    "DumperControlReqInput",
    "DumperControlReqOutput",
    "LoadLoRAAdapterReqOutput",
    "UnloadLoRAAdapterReqOutput",
    "LoadLoRAAdapterFromTensorsReqOutput",
    "MultimodalDataInputFormat",
    "sock_send",
    "sock_recv",
]
