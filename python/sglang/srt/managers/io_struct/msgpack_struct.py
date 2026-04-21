from typing import Union

import msgspec


class FreezeGCReq(msgspec.Struct, tag=True):
    pass


_unified_struct = Union[FreezeGCReq]

_msgpack_encoder = msgspec.msgpack.Encoder()
_msgpack_decoder = msgspec.msgpack.Decoder(_unified_struct)


def serialize(obj: msgspec.Struct) -> bytes:
    return _msgpack_encoder.encode(obj)


def deserialize(data: bytes) -> msgspec.Struct:
    return _msgpack_decoder.decode(data)
