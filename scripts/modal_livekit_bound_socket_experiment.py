"""Reproduce LiveKit ICE socket routing failures from a Modal gVisor container.

Design notes (why this differs from a naive socket probe):

* Targets are harvested from a live ``JoinResponse`` over the signaling
  WebSocket only. The observed production failures hit per-node endpoints such
  as ``ip-<a>-<b>-<c>-<d>.host.livekit.cloud``, which cannot be derived from
  ``LIVEKIT_URL``. No libwebrtc / FFI is used.
* Sockets are created in a burst and held open, mirroring how libwebrtc
  allocates a whole ICE port set per gathering, instead of one short-lived
  socket at a time.
* Bound and unbound sockets are created inside the same burst so both variants
  see identical conditions.

Deploy once, then drive cold starts with ``scripts/run_livekit_bound_socket_experiment.py``:

    modal deploy scripts/modal_livekit_bound_socket_experiment.py
    python scripts/run_livekit_bound_socket_experiment.py --cold-starts 6

Harvesting joins a throwaway room named ``netdiag-<random>`` in the project that
owns ``livekit-secret``. Tokens are minted inside the container and are never
logged or returned.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import platform
import socket
import struct
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import modal

_APP_NAME = "aivatar-livekit-bound-socket-experiment"
_STUN_COOKIE = b"\x21\x12\xa4\x42"
_SIOCGIFADDR = 0x8915
_ENETUNREACH_CODES = {errno.ENETUNREACH, 101}
_EHOSTUNREACH_CODES = {errno.EHOSTUNREACH, 113}

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.05-py3")
    .pip_install("livekit-api>=1.0.0", "websockets>=12.0")
)
app = modal.App(_APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Pure helpers (unit tested locally)
# ---------------------------------------------------------------------------


def _derive_target_host(room_url: str) -> str:
    hostname = (urlparse(room_url).hostname or "").strip().lower()
    if not hostname:
        raise ValueError("LIVEKIT_URL does not contain a hostname")
    cloud_suffix = ".livekit.cloud"
    turn_suffix = ".turn.livekit.cloud"
    if hostname.endswith(cloud_suffix) and not hostname.endswith(turn_suffix):
        return f"{hostname[: -len(cloud_suffix)]}{turn_suffix}"
    return hostname


def _parse_ice_server_url(url: str) -> dict[str, Any] | None:
    """Turn a single ICE server URL into a concrete probe target."""
    raw = url.strip()
    if not raw:
        return None
    scheme, _, remainder = raw.partition(":")
    scheme = scheme.lower()
    if scheme not in {"stun", "stuns", "turn", "turns"}:
        return None
    address, _, query = remainder.partition("?")
    transport = ""
    for part in query.split("&"):
        key, _, value = part.partition("=")
        if key.strip().lower() == "transport":
            transport = value.strip().lower()
    host, _, port_text = address.rpartition(":")
    if not host:
        host, port_text = address, ""
    try:
        port = int(port_text)
    except ValueError:
        port = 5349 if scheme in {"stuns", "turns"} else 3478
    if not host:
        return None
    if transport in {"tcp", "udp"}:
        protocol = transport
    else:
        protocol = "tcp" if scheme in {"stuns", "turns"} else "udp"
    return {"host": host.strip("[]"), "port": port, "protocol": protocol, "scheme": scheme, "source": "join_response"}


def _dedupe_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, str]] = set()
    unique: list[dict[str, Any]] = []
    for target in targets:
        key = (target["host"], target["port"], target["protocol"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(target)
    return unique


def _classify(record: dict[str, Any]) -> str:
    """Collapse one socket's lifecycle into a single comparable outcome."""
    for phase in ("create_errno", "send_errno", "post_hold_errno"):
        code = record.get(phase)
        if code is None:
            continue
        if code in _ENETUNREACH_CODES:
            return "errno_101_enetunreach"
        if code in _EHOSTUNREACH_CODES:
            return "errno_113_ehostunreach"
        return f"errno_{code}"
    if record["protocol"] == "udp":
        return "sent_response" if record.get("response") else "sent_no_response"
    return "connected"


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        variant = "bound" if record["bound"] else "unbound"
        outcome = _classify(record)
        counts[f"{record['protocol']}:{record['port']}:{variant}"][outcome] += 1
        by_variant[variant][outcome] += 1
    return {
        "by_target": {key: dict(value) for key, value in sorted(counts.items())},
        "by_variant": {key: dict(value) for key, value in sorted(by_variant.items())},
        "enetunreach_total": sum(1 for record in records if _classify(record) == "errno_101_enetunreach"),
        "socket_total": len(records),
    }


def _interface_ipv4_addresses() -> dict[str, str]:
    import fcntl

    addresses: dict[str, str] = {}
    for _index, name in socket.if_nameindex():
        if name == "lo":
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                request = struct.pack("256s", name[:15].encode("ascii"))
                response = fcntl.ioctl(sock.fileno(), _SIOCGIFADDR, request)
            addresses[name] = socket.inet_ntoa(response[20:24])
        except OSError:
            continue
    if not addresses:
        raise RuntimeError("No non-loopback IPv4 interface address found")
    return addresses


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _stun_request() -> tuple[bytes, bytes]:
    transaction_id = os.urandom(12)
    return struct.pack(">HH", 0x0001, 0) + _STUN_COOKIE + transaction_id, transaction_id


# ---------------------------------------------------------------------------
# Signaling-only target harvest
# ---------------------------------------------------------------------------


async def _fetch_join_response(room_url: str, api_key: str, api_secret: str, timeout: float) -> dict[str, Any]:
    from livekit.api import AccessToken, VideoGrants
    from livekit.protocol import rtc as rtc_proto

    room_name = f"netdiag-{uuid.uuid4().hex[:10]}"
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(f"netdiag-{uuid.uuid4().hex[:8]}")
        .with_grants(VideoGrants(room_join=True, room=room_name, can_publish=False, can_subscribe=False))
        .to_jwt()
    )
    parsed = urlparse(room_url)
    scheme = "wss" if parsed.scheme in {"wss", "https"} else "ws"
    base = f"{scheme}://{parsed.netloc}/rtc"
    query = f"access_token={token}&auto_subscribe=0&sdk=python&protocol=15&version=1.0.0"

    try:
        from websockets.asyncio.client import connect
    except ImportError:  # websockets < 14
        from websockets.client import connect  # type: ignore[no-redef]

    async with connect(f"{base}?{query}", open_timeout=timeout, close_timeout=1.0) as websocket:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = await asyncio.wait_for(websocket.recv(), timeout=max(0.1, deadline - time.monotonic()))
            if isinstance(frame, str):
                continue
            response = rtc_proto.SignalResponse()
            response.ParseFromString(frame)
            if response.WhichOneof("message") != "join":
                continue
            join = response.join
            targets: list[dict[str, Any]] = []
            for ice_server in join.ice_servers:
                for url in ice_server.urls:
                    parsed_target = _parse_ice_server_url(url)
                    if parsed_target is not None:
                        targets.append(parsed_target)
            return {
                "room": room_name,
                "server_region": join.server_region,
                "server_version": join.server_version,
                "ice_server_urls": [url for server in join.ice_servers for url in server.urls],
                "targets": _dedupe_targets(targets),
            }
    raise TimeoutError("No JoinResponse received before timeout")


# ---------------------------------------------------------------------------
# Burst socket trial
# ---------------------------------------------------------------------------


def _run_burst(
    resolved: list[dict[str, Any]],
    local_ip: str,
    sockets_per_target: int,
    hold_seconds: float,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Create a whole ICE-like port set at once, hold it, then measure."""
    records: list[dict[str, Any]] = []
    sockets: list[tuple[dict[str, Any], socket.socket | None]] = []
    try:
        # Phase 1: allocate every socket before sending anything.
        for target in resolved:
            for index in range(sockets_per_target):
                for bound in (False, True):
                    record: dict[str, Any] = {
                        "protocol": target["protocol"],
                        "host": target["host"],
                        "remote_ip": target["remote_ip"],
                        "port": target["port"],
                        "scheme": target.get("scheme"),
                        "source": target.get("source"),
                        "bound": bound,
                        "socket_index": index,
                        "create_errno": None,
                        "send_errno": None,
                        "post_hold_errno": None,
                        "response": None,
                        "local": None,
                    }
                    socktype = socket.SOCK_DGRAM if target["protocol"] == "udp" else socket.SOCK_STREAM
                    sock: socket.socket | None = None
                    try:
                        sock = socket.socket(socket.AF_INET, socktype)
                        sock.settimeout(timeout_seconds)
                        if bound:
                            sock.bind((local_ip, 0))
                        record["local"] = list(sock.getsockname())
                    except OSError as exc:
                        record["create_errno"] = exc.errno
                        record["error"] = exc.strerror or str(exc)
                        if sock is not None:
                            sock.close()
                            sock = None
                    records.append(record)
                    sockets.append((record, sock))

        # Phase 2: first send / connect from every live socket.
        pending_udp: list[tuple[dict[str, Any], socket.socket, bytes]] = []
        for record, sock in sockets:
            if sock is None:
                continue
            try:
                if record["protocol"] == "udp":
                    request, transaction_id = _stun_request()
                    sock.sendto(request, (record["remote_ip"], record["port"]))
                    pending_udp.append((record, sock, transaction_id))
                else:
                    sock.connect((record["remote_ip"], record["port"]))
                    record["local"] = list(sock.getsockname())
            except OSError as exc:
                record["send_errno"] = exc.errno
                record["error"] = exc.strerror or str(exc)

        # Phase 3: collect UDP replies without serialising the timeouts.
        for record, sock, transaction_id in pending_udp:
            try:
                payload, source = sock.recvfrom(2048)
                record["response"] = (
                    len(payload) >= 20 and payload[4:8] == _STUN_COOKIE and payload[8:20] == transaction_id
                )
                record["response_from"] = list(source)
            except TimeoutError:
                record["response"] = False
            except OSError as exc:
                record["send_errno"] = record["send_errno"] or exc.errno
                record["error"] = exc.strerror or str(exc)

        # Phase 4: hold the whole set open, then re-send to catch late death.
        if hold_seconds > 0:
            time.sleep(hold_seconds)
        for record, sock in sockets:
            if sock is None or record["send_errno"] is not None:
                continue
            try:
                if record["protocol"] == "udp":
                    request, _ = _stun_request()
                    sock.sendto(request, (record["remote_ip"], record["port"]))
                else:
                    sock.send(b"\x00")
            except OSError as exc:
                record["post_hold_errno"] = exc.errno
                record["error"] = exc.strerror or str(exc)
    finally:
        for _record, sock in sockets:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
    return records


@app.cls(
    gpu="L40S",
    min_containers=0,
    scaledown_window=15,
    timeout=900,
    secrets=[modal.Secret.from_name("livekit-secret")],
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=1)
class SnapshotSocketExperiment:
    @modal.enter(snap=True)
    def load(self) -> None:
        self.snapshot_created_epoch = time.time()

    @modal.enter(snap=False)
    def restore(self) -> None:
        self.restore_epoch = time.time()
        self.restore_monotonic = time.monotonic()
        self.restore_network = {
            "hostname": socket.gethostname(),
            "interfaces": _interface_ipv4_addresses(),
            "proc_net_route": _read_text("/proc/net/route"),
        }

    @modal.method()
    def run(
        self,
        explicit_targets: list[dict[str, Any]],
        bursts: int,
        sockets_per_target: int,
        hold_seconds: float,
        timeout_seconds: float,
        interval_seconds: float,
        harvest_timeout: float,
    ) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc).isoformat()
        room_url = os.environ.get("LIVEKIT_URL", "")
        harvest: dict[str, Any] | None = None
        harvest_error: str | None = None
        targets: list[dict[str, Any]] = list(explicit_targets)
        target_source = "explicit" if targets else "unset"

        if not targets:
            try:
                harvest = asyncio.run(
                    _fetch_join_response(
                        room_url,
                        os.environ["LIVEKIT_API_KEY"],
                        os.environ["LIVEKIT_API_SECRET"],
                        harvest_timeout,
                    )
                )
                targets = harvest["targets"]
                target_source = "join_response"
            except Exception as exc:
                harvest_error = f"{type(exc).__name__}: {exc}"
            if not targets:
                fallback = _derive_target_host(room_url)
                targets = [
                    {"host": fallback, "port": 3478, "protocol": "udp", "scheme": "turn", "source": "derived"},
                    {"host": fallback, "port": 443, "protocol": "tcp", "scheme": "turns", "source": "derived"},
                ]
                target_source = "derived_fallback"

        resolved: list[dict[str, Any]] = []
        resolution_errors: list[dict[str, str]] = []
        for target in _dedupe_targets(targets):
            try:
                infos = socket.getaddrinfo(target["host"], target["port"], socket.AF_INET, socket.SOCK_DGRAM)
            except OSError as exc:
                resolution_errors.append({"host": target["host"], "error": str(exc)})
                continue
            remote_ip = infos[0][4][0]
            resolved.append({**target, "remote_ip": remote_ip})
        if not resolved:
            raise RuntimeError(f"No probe target resolved; errors={resolution_errors}")

        interfaces = _interface_ipv4_addresses()
        interface_name, local_ip = next(iter(interfaces.items()))
        all_records: list[dict[str, Any]] = []
        for burst in range(bursts):
            since_restore_ms = round((time.monotonic() - self.restore_monotonic) * 1000, 3)
            records = _run_burst(resolved, local_ip, sockets_per_target, hold_seconds, timeout_seconds)
            for record in records:
                record["burst"] = burst + 1
                record["since_restore_ms"] = since_restore_ms
            all_records.extend(records)
            print(
                json.dumps(
                    {
                        "burst": burst + 1,
                        "since_restore_ms": since_restore_ms,
                        "sockets": len(records),
                        "enetunreach": sum(1 for item in records if _classify(item) == "errno_101_enetunreach"),
                        "outcomes": dict(Counter(_classify(item) for item in records)),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if burst + 1 < bursts:
                time.sleep(interval_seconds)

        result = {
            "experiment": _APP_NAME,
            "started_at": started_at,
            "target_source": target_source,
            "harvest": harvest,
            "harvest_error": harvest_error,
            "resolution_errors": resolution_errors,
            "resolved_targets": resolved,
            "bursts": bursts,
            "sockets_per_target": sockets_per_target,
            "hold_seconds": hold_seconds,
            "timeout_seconds": timeout_seconds,
            "interval_seconds": interval_seconds,
            "selected_interface": interface_name,
            "local_ipv4": local_ip,
            "snapshot_created_epoch": self.snapshot_created_epoch,
            "restore_epoch": self.restore_epoch,
            "boot_delta_seconds": round(self.restore_epoch - self.snapshot_created_epoch, 1),
            "system": {
                "platform": platform.platform(),
                "at_restore": self.restore_network,
                "after_bursts": {
                    "interfaces": _interface_ipv4_addresses(),
                    "proc_net_route": _read_text("/proc/net/route"),
                },
            },
            "summary": _summarize(all_records),
            "records": all_records,
        }
        print(json.dumps({"final_summary": result["summary"]}, separators=(",", ":")), flush=True)
        return result
