"""Bounded reciprocal UCX refresh on the existing per-rank control sockets."""

import json
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sglang.srt.disaggregation.nixl.bootstrap import (
    MAX_ENCODED_METADATA,
    PrefillPeerRegistrar,
    publish_peer_metadata,
)

REQUEST = b"NIXL_UCX_REFRESH_V1"
ACK = b"NIXL_UCX_REFRESH_ACK_V1"
MAX_MESSAGE_BYTES = MAX_ENCODED_METADATA + 2048
REFRESH_TIMEOUT_S = 5.0
logger = logging.getLogger(__name__)


@dataclass
class _Pending:
    peer: str
    sequence: int
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class UCXPeerRecovery:
    def __init__(
        self,
        agent: Any,
        endpoint: str,
        send: Callable[[str, List[bytes]], None],
        registrar: PrefillPeerRegistrar,
        refresh: Callable[[Dict[str, str], str], None],
    ):
        self.agent = agent
        self.endpoint = endpoint
        self.send = send
        self.registrar = registrar
        self.refresh = refresh
        self._lock = threading.Lock()
        self._serve_lock = threading.Lock()
        self._known: Dict[str, str] = {}
        self._pending: Dict[str, _Pending] = {}
        self._sequences: Dict[str, int] = {}
        self._last_received: Dict[str, int] = {}
        self._replies: OrderedDict = OrderedDict()

    def remember_producer(self, metadata: Dict[str, str], endpoint: str) -> None:
        peer = self.registrar.accepted_peer(metadata)
        if peer is None:
            raise ValueError("Cannot remember an unregistered producer")
        with self._serve_lock:
            self._known[peer] = endpoint

    def request(self, peer: str, endpoint: str) -> Dict[str, str]:
        nonce = uuid.uuid4().hex
        with self._lock:
            sequence = self._sequences.get(peer, 0) + 1
            self._sequences[peer] = sequence
            pending = _Pending(peer, sequence)
            self._pending[nonce] = pending
        try:
            metadata = publish_peer_metadata(self.agent, "UCX")
            self.send(
                endpoint,
                [
                    REQUEST,
                    b"1",
                    nonce.encode(),
                    str(sequence).encode(),
                    self.agent.name.encode(),
                    peer.encode(),
                    self.endpoint.encode(),
                    json.dumps(metadata).encode(),
                ],
            )
            if not pending.event.wait(REFRESH_TIMEOUT_S):
                raise TimeoutError("Consumer UCX refresh acknowledgement timed out")
            if pending.error is not None:
                raise RuntimeError("Consumer UCX refresh failed: " + pending.error)
            if pending.result is None:
                raise RuntimeError("Consumer UCX refresh returned no metadata")
            return pending.result
        finally:
            with self._lock:
                self._pending.pop(nonce, None)

    @staticmethod
    def _parse(parts: List[bytes]):
        if len(parts) != 8 or sum(map(len, parts)) > MAX_MESSAGE_BYTES:
            raise ValueError("Refresh message exceeds protocol bounds")
        if parts[1] != b"1":
            raise ValueError("Unsupported refresh version")
        nonce = parts[2].decode("ascii")
        if len(nonce) != 32 or any(c not in "0123456789abcdef" for c in nonce):
            raise ValueError("Invalid refresh nonce")
        sequence = int(parts[3])
        if not 0 < sequence < 2**63:
            raise ValueError("Invalid refresh sequence")
        names = [p.decode("ascii") for p in parts[4:6]]
        if any(not n or len(n) > 256 for n in names):
            raise ValueError("Invalid refresh peer identity")
        if len(parts[6]) > 1024:
            raise ValueError("Invalid refresh route or status")
        payload = json.loads(parts[7])
        if not isinstance(payload, dict):
            raise ValueError("Refresh payload must be an object")
        return nonce, sequence, names[0], names[1], parts[6].decode("ascii"), payload

    def handle(self, parts: List[bytes]) -> bool:
        if not parts or parts[0] not in (REQUEST, ACK):
            return False
        try:
            nonce, sequence, producer, consumer, route, payload = self._parse(parts)
            if parts[0] == ACK:
                if producer != self.agent.name or route not in ("OK", "ERROR"):
                    return True
                with self._lock:
                    pending = self._pending.get(nonce)
                    if (
                        pending is None
                        or pending.peer != consumer
                        or pending.sequence != sequence
                    ):
                        return True
                    if pending.event.is_set():
                        return True
                    if route == "OK":
                        # Validate encoding/version before releasing the sender.
                        self.registrar._decode(payload)
                        pending.result = payload
                    else:
                        error = payload.get("error")
                        if not isinstance(error, str) or len(error) > 256:
                            return True
                        pending.error = error
                    pending.event.set()
                return True
            with self._serve_lock:
                if consumer != self.agent.name or self._known.get(producer) != route:
                    return True
                if self.registrar.accepted_peer(payload) != producer:
                    return True
                key = (producer, nonce, sequence)
                reply = self._replies.get(key)
                if reply is None:
                    if sequence <= self._last_received.get(producer, 0):
                        return True
                    self._last_received[producer] = sequence
                    try:
                        self.refresh(payload, producer)
                        metadata = publish_peer_metadata(self.agent, "UCX")
                        status = b"OK"
                    except Exception as error:
                        metadata = {"error": type(error).__name__}
                        status = b"ERROR"
                    reply = [
                        ACK,
                        b"1",
                        nonce.encode(),
                        str(sequence).encode(),
                        producer.encode(),
                        consumer.encode(),
                        status,
                        json.dumps(metadata).encode(),
                    ]
                    self._replies[key] = reply
                    if len(self._replies) > 128:
                        self._replies.popitem(last=False)
                self.send(route, reply)
            return True
        except (ValueError, TypeError, UnicodeError, KeyError):
            # Invalid traffic must not kill either rank listener or trigger RMA.
            return True
        except Exception as error:
            # A failed ACK send must not kill the consumer rank listener.
            # The producer times out and cannot proceed to another RMA write.
            logger.warning("UCX refresh control failed: %s", type(error).__name__)
            return True
