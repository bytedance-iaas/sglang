from enum import IntEnum, unique


@unique
class P2PTag(IntEnum):
    """Tags reserved for point-to-point communication protocols."""

    DEFAULT = 0
    HIRADIX_PP_SYNC = int.from_bytes(b"PpHi", byteorder="big")
