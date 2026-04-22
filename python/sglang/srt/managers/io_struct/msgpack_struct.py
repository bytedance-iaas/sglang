from typing import Optional, Union

import msgspec


class FreezeGCReq(msgspec.Struct, tag=True):
    pass


class FlushCacheReqInput(msgspec.Struct, tag=True):
    timeout_s: Optional[float] = None


class FlushCacheReqOutput(msgspec.Struct, tag=True):
    success: bool
    message: str = ""


_unified_struct = Union[FreezeGCReq, FlushCacheReqInput, FlushCacheReqOutput]

_msgpack_encoder = msgspec.msgpack.Encoder()
_msgpack_decoder = msgspec.msgpack.Decoder(_unified_struct)


def serialize(obj: msgspec.Struct) -> bytes:
    return _msgpack_encoder.encode(obj)


def deserialize(data: bytes) -> msgspec.Struct:
    return _msgpack_decoder.decode(data)
