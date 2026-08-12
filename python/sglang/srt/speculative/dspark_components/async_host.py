from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch

_Payload = dict[str, Any]
_TensorOrPayload = Union[torch.Tensor, _Payload]
_DUMMY_KEY = "__tensor__"


@dataclass(slots=True, kw_only=True)
class FutureTensors:
    _data: Optional[_Payload]
    _event: Optional[torch.cuda.Event]
    _retained_device_clones: Optional[dict[str, torch.Tensor]] = None

    @classmethod
    def device_to_host(
        cls, xs_device: _TensorOrPayload, *, d2h_stream: torch.cuda.Stream
    ) -> "FutureTensors":
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DSpark async D2H staging cannot run during capture")
        if not isinstance(xs_device, dict):
            xs_device = {_DUMMY_KEY: xs_device}
        first = next(
            (value for value in xs_device.values() if isinstance(value, torch.Tensor)),
            None,
        )
        if first is None:
            raise ValueError("DSpark async D2H staging requires a tensor")
        device = first.device
        tensors = {
            key: value
            for key, value in xs_device.items()
            if isinstance(value, torch.Tensor)
        }
        metadata = {
            key: value
            for key, value in xs_device.items()
            if not isinstance(value, torch.Tensor)
        }
        clones = {key: value.detach().clone() for key, value in tensors.items()}
        host = {
            key: torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
            for key, value in tensors.items()
        }
        d2h_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(d2h_stream):
            for key, value in clones.items():
                host[key].copy_(value, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        return cls(
            _data=host | metadata,
            _event=event,
            _retained_device_clones=clones,
        )

    def wait(self) -> _TensorOrPayload:
        data, event = self._data, self._event
        retained = self._retained_device_clones
        self._data = None
        self._event = None
        self._retained_device_clones = None
        if data is None or event is None:
            raise RuntimeError("FutureTensors.wait() was called more than once")
        event.synchronize()
        del retained
        return data[_DUMMY_KEY] if _DUMMY_KEY in data else data


@dataclass(slots=True, kw_only=True)
class DelayedDeviceHostHandler:
    d2h_stream: torch.cuda.Stream
    _future: Optional[FutureTensors] = field(default=None)

    def step(
        self,
        *,
        compute_on_device: Callable[[], Optional[_TensorOrPayload]],
        postprocess_on_host: Callable[[_TensorOrPayload], None],
    ) -> None:
        if self._future is not None:
            postprocess_on_host(self._future.wait())
            self._future = None
        device_data = compute_on_device()
        if device_data is not None:
            self._future = FutureTensors.device_to_host(
                device_data, d2h_stream=self.d2h_stream
            )
