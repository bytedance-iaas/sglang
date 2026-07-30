from enum import IntEnum, unique


@unique
class P2PTag(IntEnum):
    """Tags reserved for point-to-point communication protocols."""

    DEFAULT = 0
    HIRADIX_PP_SYNC = int.from_bytes(b"PpHi", byteorder="big")
    # EIC load-back admission verdicts (own tag: must never share a FIFO slot
    # with the num_ready stream above).
    HIRADIX_PP_VERDICT = int.from_bytes(b"PpVd", byteorder="big")
