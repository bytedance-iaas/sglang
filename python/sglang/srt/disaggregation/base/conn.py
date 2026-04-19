from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


@dataclasses.dataclass
class KVArgs:
    """Argument bag shared across disaggregation backends.

    Historically this object was a plain class that callers mutated by setting
    attributes after construction. Keeping this as a dataclass preserves that
    usage pattern while making fields explicit (better IDE support + fewer
    `getattr()` fallbacks).
    """

    engine_rank: int = 0

    kv_data_ptrs: List[int] = dataclasses.field(default_factory=list)
    kv_data_lens: List[int] = dataclasses.field(default_factory=list)
    kv_item_lens: List[int] = dataclasses.field(default_factory=list)

    aux_data_ptrs: List[int] = dataclasses.field(default_factory=list)
    aux_data_lens: List[int] = dataclasses.field(default_factory=list)
    aux_item_lens: List[int] = dataclasses.field(default_factory=list)

    state_data_ptrs: List[int] = dataclasses.field(default_factory=list)
    state_data_lens: List[int] = dataclasses.field(default_factory=list)
    state_item_lens: List[int] = dataclasses.field(default_factory=list)
    state_type: str = "none"  # "none", "mamba", "swa", ...

    # For Mamba state different TP slice transfer.
    state_dim_per_tensor: List[int] = dataclasses.field(default_factory=list)

    ib_device: str = ""
    ib_traffic_class: str = ""
    gpu_id: int = 0

    kv_head_num: int = 0
    total_kv_head_num: int = 0
    page_size: int = 0

    # For PP prefill.
    pp_rank: int = 0
    prefill_start_layer: int = 0

    # For system DP.
    system_dp_rank: int = 0


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class BaseKVManager(ABC):
    """Base class for managing transfer states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...

    @abstractmethod
    def register_to_bootstrap(self):
        """Register prefill server info to the bootstrap server."""
        ...


class BaseKVSender(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Set req's index metadata locally or notify the decoder server about the kv indices length and aux index.
        """
        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        """
        Send the kv cache at the given kv indices and the extra cache/state at the given indices to the decoder server.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...


class BaseKVReceiver(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(
        self,
        prefill_dp_rank: int,
    ):
        """
        Resolve bootstrap metadata and mark the receiver ready for transfer metadata.
        """
        ...

    @abstractmethod
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """
        Notify the prefill server about the kv indices, aux index, and state_indices.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...

    def clear(self):
        """
        Clear any internal states.
        """
        pass

    def abort(self):
        """
        Abort the current transfer.
        """
        pass


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, host: str, port: int): ...
