"""LiveKit net probe, egress warming, and ICE policy helpers.

All network probe/warm/STUN/UDP/TCP helpers, env helpers, ICE constants,
and room diag logging live here. Functions that are patched by tests
(_udp_egress_sendable, _warm_egress_targets) are looked up from the
package namespace at call time so monkeypatch.setattr on the package
module takes effect.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import time
from urllib.parse import urlsplit

from livekit import rtc

logger = logging.getLogger("LiveKitEgressAdapter")

# ICE transport policy: attempt 1 defaults to TRANSPORT_ALL (direct, best
# latency), but uses TRANSPORT_RELAY when the response probe finds silent UDP
# and working TCP 443. Deep-blanket (UDP silent plus TCP probe ambiguous)
# also goes RELAY via LIVEKIT_DEEP_BLANKET_RELAY=1 (kill switch =0). Retries
# remain TURN-only. Server ICE preserved (no custom ice_servers).
_ICE_TRANSPORT_ALL = getattr(
    getattr(rtc, "_proto", None), "room_pb2", None
)
if _ICE_TRANSPORT_ALL is not None:
    _ICE_TRANSPORT_ALL = _ICE_TRANSPORT_ALL.IceTransportType.TRANSPORT_ALL
    _ICE_TRANSPORT_RELAY = getattr(
        getattr(rtc, "_proto", None).room_pb2.IceTransportType,
        "TRANSPORT_RELAY",
        0,
    )
else:
    _ICE_TRANSPORT_ALL = 2
    _ICE_TRANSPORT_RELAY = 0


def _attach_room_diag(room, session_id: str) -> None:
    """Attach connection-lifecycle diag logging to an adapter-owned room.

    Observational only: rooms without event support or failing subscriptions
    never block the connect path.
    """
    on_event = getattr(room, "on", None)
    if not callable(on_event):
        return

    def _state(connection_state):
        logger.info("LIVEKIT_ROOM_STATE session=%s state=%s", session_id, connection_state)

    def _disconnected(reason):
        logger.info("LIVEKIT_ROOM_DISCONNECTED session=%s reason=%s", session_id, reason)

    def _reconnecting():
        logger.info("LIVEKIT_ROOM_RECONNECTING session=%s", session_id)

    def _reconnected():
        logger.info("LIVEKIT_ROOM_RECONNECTED session=%s", session_id)

    for event_name, handler in (
        ("connection_state_changed", _state),
        ("disconnected", _disconnected),
        ("reconnecting", _reconnecting),
        ("reconnected", _reconnected),
    ):
        try:
            on_event(event_name, handler)
        except Exception:
            logger.debug("LiveKit diag subscription failed for %s", event_name, exc_info=True)


_NET_PROBE_STUN_REQUEST = struct.pack("!HHI", 0x0001, 0, 0x2112A442) + b"\x00" * 12


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float, minimum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(value, minimum)


def _is_ice_timeout_error(exc: BaseException) -> bool:
    """True when a connect failure looks like an ICE-timeout burn.

    The Rust engine surfaces these as `wait_pc_connection timed out` after
    the hard-coded 15s ICE wait. Fast teardown/backoff is safe here because
    a failed ICE gather holds no publishable state worth draining slowly.
    """
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return ("wait_pc_connection" in text) or ("timed out" in text and "ice" in text) or (
        "timed out" in text and "connect" in text
    )


def _warm_egress_targets(hosts: list[str], port: int, *, packets_per_host: int = 2) -> None:
    """Fire-and-forget UDP egress to prime gVisor NAT/route state.

    No response awaited. Best-effort only, never raises. Uses throwaway
    sockets so engine ICE sockets are unaffected. Helps per-destination
    blackholes where the exact TURN/media IP needs its own NAT entry.
    """
    for host in dict.fromkeys(hosts):
        try:
            infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_DGRAM)
        except OSError:
            continue
        if not infos:
            continue
        infos.sort(key=lambda info: 0 if info[0] == socket.AF_INET else 1)
        for family, socktype, proto, _, address in infos[:2]:
            for _ in range(packets_per_host):
                try:
                    with socket.socket(family, socktype, proto) as sock:
                        sock.settimeout(0.2)
                        sock.sendto(_NET_PROBE_STUN_REQUEST, address)
                except OSError:
                    break


def _derive_net_probe_host(room_url: str) -> str:
    """Derive the UDP probe target from the signaling URL.

    LiveKit Cloud exposes a cluster TURN endpoint at ``<project>.turn.livekit.cloud``
    in the same subnet as the media nodes, so it is a stable stand-in for the UDP
    egress path without needing a room token's ICE servers.
    """
    override = os.environ.get("LIVEKIT_NET_PROBE_HOST")
    if override:
        return override
    hostname = None
    try:
        hostname = urlsplit(room_url).hostname
    except ValueError:
        hostname = None
    if not hostname:
        return room_url
    labels = hostname.split(".")
    if (
        len(labels) >= 3
        and labels[-3] != "turn"
        and labels[-2] == "livekit"
        and labels[-1] == "cloud"
    ):
        return ".".join(labels[:-2] + ["turn"] + labels[-2:])
    return hostname


def _derive_net_probe_signaling_host(room_url: str) -> str | None:
    """Extract the raw signaling hostname from the room URL.

    This is used as a secondary probe target because the signaling host and
    media nodes typically share a subnet in LiveKit Cloud. Probing the
    signaling host catches per-destination blackholes that the TURN probe
    (different host) misses.
    """
    try:
        hostname = urlsplit(room_url).hostname
    except ValueError:
        return None
    return hostname or None


import threading as _threading

_RECENT_EGRESS_MAX = 5
_recent_egress_hosts: list[tuple[str, int]] = []
_recent_egress_lock = _threading.Lock()


def record_egress_hosts(hosts: list[tuple[str, int]]) -> None:
    """Record session TURN/signaling targets for heartbeat priming.

    Thread-safe MRU insert, deduped by (host, port), capped. Never raises.
    No hardcoded customer host: callers pass per-session derived targets.
    """
    try:
        cleaned: list[tuple[str, int]] = []
        for host, port in hosts or []:
            if not isinstance(host, str):
                continue
            host = host.strip().lower()
            if not host:
                continue
            try:
                port = int(port)
            except (TypeError, ValueError):
                continue
            if not 1 <= port <= 65535:
                continue
            if (host, port) not in cleaned:
                cleaned.append((host, port))
        if not cleaned:
            return
        with _recent_egress_lock:
            for item in reversed(cleaned):
                if item in _recent_egress_hosts:
                    _recent_egress_hosts.remove(item)
                _recent_egress_hosts.insert(0, item)
            del _recent_egress_hosts[_RECENT_EGRESS_MAX :]
    except Exception:
        return


def get_recent_egress_hosts() -> list[tuple[str, int]]:
    """Return a copy of recent session egress targets, MRU first."""
    try:
        with _recent_egress_lock:
            return list(_recent_egress_hosts)
    except Exception:
        return []


def warm_egress_for_url(room_url: str | None, *, port: int | None = None) -> None:
    """Best-effort UDP egress warm for a session room URL.

    Sync, blocking-socket, never raises, no FFI use. Invoke via
    asyncio.to_thread fire-and-forget at session start. Derives
    per-customer targets only, never a hardcoded project TURN host.
    """
    # Deferred import: tests patch _warm_egress_targets on the package
    # namespace via monkeypatch.setattr(livekit_egress, ...).
    from streaming.transport.adapters.livekit_egress import _warm_egress_targets

    try:
        if not _env_bool("LIVEKIT_EGRESS_WARM_ENABLED", True):
            return
        if not room_url or not isinstance(room_url, str):
            return
        if port is None:
            try:
                port = int(os.environ.get("LIVEKIT_NET_PROBE_PORT", "3478"))
            except ValueError:
                port = 3478
        hosts: list[str] = []
        for host in (_derive_net_probe_host(room_url), _derive_net_probe_signaling_host(room_url)):
            if host and host not in hosts:
                hosts.append(host)
        try:
            room_host = urlsplit(room_url).hostname
        except ValueError:
            room_host = None
        if room_host and room_host not in hosts:
            hosts.append(room_host)
        if not hosts:
            return
        _warm_egress_targets(hosts, port)
    except Exception:
        return


def _probe_targets(
    targets: list[tuple[str, int]],
) -> tuple[bool, str | None, bool, tuple[str, int] | None]:
    """Probe multiple (host, port) targets in order; return on first success.

    Returns (sendable, error, can_probe, target_that_succeeded).
    can_probe is True only if at least one target was resolvable and attempted.
    """
    # Deferred import: tests patch _udp_egress_sendable on the package
    # namespace via monkeypatch.setattr(livekit_egress, ...).
    from streaming.transport.adapters.livekit_egress import _udp_egress_sendable

    last_error: str | None = None
    any_can_probe = False
    for host, port in targets:
        sendable, error, can_probe = _udp_egress_sendable(host, port)
        if sendable:
            return True, None, True, (host, port)
        if can_probe:
            any_can_probe = True
        if error:
            last_error = error
    return False, last_error or "all probes failed", any_can_probe, None


def _udp_egress_sendable(host: str, port: int) -> tuple[bool, str | None, bool]:
    """Send a STUN binding request and require a matching UDP response."""
    try:
        response_timeout = float(
            os.environ.get("LIVEKIT_NET_PROBE_RESPONSE_TIMEOUT_SECONDS", "0.8")
        )
    except ValueError:
        response_timeout = 0.8
    response_timeout = max(response_timeout, 0.05)

    try:
        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_DGRAM)
    except OSError as exc:
        # Name resolution or address lookup failure is not the same as the UDP
        # egress blackhole the probe is meant to detect.
        return False, f"resolve failed ({exc.__class__.__name__}: {exc})", False

    if not infos:
        return False, "no addresses returned for probe target", False

    # Prefer IPv4 (most LiveKit Cloud media paths today), but fall back to
    # IPv6 if the target is v6-only.
    infos.sort(key=lambda info: 0 if info[0] == socket.AF_INET else 1)

    last_error: str | None = None
    for family, socktype, proto, _, address in infos:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(response_timeout)
                sock.sendto(_NET_PROBE_STUN_REQUEST, address)
                response, _source = sock.recvfrom(2048)
                if (
                    len(response) < 20
                    or response[:2] not in {b"\x01\x01", b"\x01\x11"}
                    or response[4:8] != _NET_PROBE_STUN_REQUEST[4:8]
                    or response[8:20] != _NET_PROBE_STUN_REQUEST[8:20]
                ):
                    last_error = "invalid STUN response"
                    continue
            return True, None, True
        except OSError as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            continue

    return False, last_error or "STUN response failed", True


def _tcp_egress_connectable(
    hosts: list[str], timeout: float
) -> tuple[bool, str | None, int | None, str | None]:
    """Try TCP 443 targets in order and return readiness plus address family."""
    last_error: str | None = None
    for host in hosts:
        try:
            with socket.create_connection((host, 443), timeout=timeout) as sock:
                return True, None, sock.family, host
        except OSError as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
    return False, last_error or "TCP connect failed", None, None
