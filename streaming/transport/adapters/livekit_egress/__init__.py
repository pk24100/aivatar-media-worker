"""LiveKit media egress adapter package.

Facade re-exporting the public API previously in the monolithic
livekit_egress.py. Submodules:
- _net_probe: net probe, egress warming, ICE policy, room diag, env helpers
- _adapter: LiveKitEgressAdapter class

All names that tests patch via monkeypatch.setattr(livekit_egress, ...) are
re-exported here so the package namespace matches the old module namespace.
"""

import socket  # re-export for test patching (livekit_egress.socket)
from livekit import rtc  # re-export for test patching (livekit_egress.rtc)
import numpy as np  # re-export for compatibility

# Re-export publisher classes so tests can patch them on the package namespace
# (monkeypatch.setattr(livekit_egress, "VideoPublisher", ...)).
from streaming.transport.livekit.audio_publisher import AudioPublisher
from streaming.transport.livekit.video_publisher import VideoPublisher

from ._net_probe import (
    _ICE_TRANSPORT_ALL,
    _ICE_TRANSPORT_RELAY,
    _attach_room_diag,
    _NET_PROBE_STUN_REQUEST,
    _env_bool,
    _env_float,
    _is_ice_timeout_error,
    _warm_egress_targets,
    _derive_net_probe_host,
    _derive_net_probe_signaling_host,
    _RECENT_EGRESS_MAX,
    _recent_egress_hosts,
    _recent_egress_lock,
    record_egress_hosts,
    get_recent_egress_hosts,
    warm_egress_for_url,
    _probe_targets,
    _udp_egress_sendable,
    _tcp_egress_connectable,
)
from ._adapter import LiveKitEgressAdapter

__all__ = [
    "LiveKitEgressAdapter",
    "warm_egress_for_url",
    "record_egress_hosts",
    "get_recent_egress_hosts",
    "AudioPublisher",
    "VideoPublisher",
]
