"""Publish and import UCX peer metadata before starting a PD transfer."""

import base64
import hashlib
import threading
from typing import Any, Dict, Optional

MAX_ENCODED_METADATA = 512 * 1024


def publish_peer_metadata(agent: Any, backend: str) -> Optional[Dict[str, str]]:
    if backend != "UCX":
        return None
    encoded = base64.b64encode(agent.get_agent_metadata()).decode("ascii")
    if not encoded or len(encoded) > MAX_ENCODED_METADATA:
        raise ValueError("NIXL bootstrap metadata exceeds the publication bound")
    return {"version": "1", "backend": backend, "agent_metadata": encoded}


class PrefillPeerRegistrar:
    """Serialize registrations and cache only metadata NIXL accepted.

    The receiver imports the producer before publishing its own metadata to
    that producer. This gives both UCX endpoints a connection before KV writes.
    A cached import is reused only while NIXL still considers that peer valid.
    """

    def __init__(self, agent: Any, backend: str):
        self.agent = agent
        self.backend = backend
        self._lock = threading.Lock()
        self._accepted: Dict[str, str] = {}

    def register(self, metadata: Optional[Dict[str, str]]) -> bool:
        # Older producers and other backends retain their existing protocol.
        if metadata is None or self.backend != "UCX":
            return False
        if not isinstance(metadata, dict):
            raise ValueError("NIXL bootstrap transfer metadata must be an object")
        if metadata.get("version") != "1" or metadata.get("backend") != self.backend:
            raise ValueError(
                "NIXL bootstrap transfer metadata version/backend mismatch"
            )
        encoded = metadata.get("agent_metadata")
        if not isinstance(encoded, str) or not 0 < len(encoded) <= MAX_ENCODED_METADATA:
            raise ValueError("NIXL bootstrap metadata exceeds the import bound")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, UnicodeError) as error:
            raise ValueError("Invalid NIXL bootstrap metadata encoding") from error
        if not raw:
            raise ValueError("Empty NIXL bootstrap metadata")
        digest = hashlib.sha256(raw).hexdigest()
        with self._lock:
            peer = self._accepted.get(digest)
            if peer is not None and self.agent.check_remote_metadata(peer):
                return False
            # loadRemoteMD returns the identity encoded in the NIXL metadata.
            # Do not publish a cache entry before this operation succeeds.
            peer = self.agent.add_remote_agent(raw)
            self._accepted[digest] = peer
            return True
