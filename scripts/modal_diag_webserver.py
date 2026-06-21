"""
Modal diagnostic script: @modal.web_server() endpoint.
Runs UDP socket, native livekit.rtc, and aiortc ICE tests inside a
Modal web_server container and returns structured JSON via HTTP GET /.

Usage:
    cd aivatar-media-worker
    modal serve scripts/modal_diag_webserver.py
    # Then curl the printed URL

Or for one-shot deploy + curl:
    modal deploy scripts/modal_diag_webserver.py
    curl <deployed_url>
    modal app stop aivatar-diag-webserver
"""

import json
import os

import modal

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")
    .apt_install("git", "git-lfs", "ffmpeg", "libsndfile1", "wget", "ca-certificates")
    .pip_install("ninja")
    .run_commands("pip install flash-attn --no-build-isolation || true")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App("aivatar-diag-webserver", image=image)

# ---------------------------------------------------------------------------
# Shared diagnostic logic (runs inside container)
# ---------------------------------------------------------------------------

_DIAGNOSTIC_CODE = r'''
import asyncio
import io
import json
import logging
import os
import random
import socket
import struct
import time
import traceback
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 0. Logging setup — capture EVERYTHING into a buffer AND stdout
# ---------------------------------------------------------------------------
_log_buffer = io.StringIO()

_root = logging.getLogger()
_root.setLevel(logging.DEBUG)

# Console handler (goes to Modal container stdout)
_console = logging.StreamHandler()
_console.setLevel(logging.DEBUG)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
_root.addHandler(_console)

# Buffer handler (captured into JSON response)
_buf = logging.StreamHandler(_log_buffer)
_buf.setLevel(logging.DEBUG)
_buf.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
_root.addHandler(_buf)

os.environ["LIVEKIT_RTC_DEBUG"] = "true"

_logger = logging.getLogger("root_cause_diag")

def _log(msg, *args):
    formatted = msg % args if args else msg
    print(formatted, flush=True)
    _logger.info(formatted)

# ---------------------------------------------------------------------------
# 1. UDP socket diagnostic
# ---------------------------------------------------------------------------

async def diag_udp() -> dict:
    """Test whether the container can send/receive UDP packets."""
    results = {
        "test": "udp_socket",
        "stun_host": "stun.l.google.com",
        "stun_port": 19302,
        "steps": [],
        "success": False,
        "duration_ms": 0.0,
    }
    t0 = time.monotonic()

    sock = None
    try:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            results["steps"].append({"step": "socket_create", "ok": True})
        except Exception as exc:
            results["steps"].append({"step": "socket_create", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return results

        try:
            sock.bind(("0.0.0.0", 0))
            local_addr = sock.getsockname()
            results["steps"].append({"step": "socket_bind", "ok": True, "local_addr": local_addr})
        except Exception as exc:
            results["steps"].append({"step": "socket_bind", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return results

        tid = bytes(random.randint(0, 255) for _ in range(12))
        stun_req = struct.pack(">HH", 0x0001, 0x0000) + b"\x21\x12\xA4\x42" + tid
        try:
            sock.sendto(stun_req, (results["stun_host"], results["stun_port"]))
            results["steps"].append({"step": "stun_send", "ok": True, "bytes_sent": len(stun_req)})
        except Exception as exc:
            results["steps"].append({"step": "stun_send", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return results

        try:
            loop = asyncio.get_event_loop()
            data, addr = await asyncio.wait_for(
                loop.sock_recvfrom(sock, 2048),
                timeout=5.0
            )
            if len(data) >= 20:
                msg_type, msg_len, magic = struct.unpack(">HHI", data[:8])
                resp_tid = data[8:20]
                results["steps"].append({
                    "step": "stun_recv",
                    "ok": True,
                    "bytes_recv": len(data),
                    "from": addr,
                    "stun_msg_type": hex(msg_type),
                    "magic_ok": magic == 0x2112A442,
                    "tid_ok": resp_tid == tid,
                })
                results["success"] = True
            else:
                results["steps"].append({"step": "stun_recv", "ok": False, "error": "short packet"})
        except asyncio.TimeoutError:
            results["steps"].append({"step": "stun_recv", "ok": False, "error": "timeout (5s) — no UDP response"})
        except Exception as exc:
            results["steps"].append({"step": "stun_recv", "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    finally:
        if sock:
            sock.close()

    results["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return results


# ---------------------------------------------------------------------------
# 2. Native livekit.rtc diagnostic
# ---------------------------------------------------------------------------

async def diag_native_livekit(url: str, token: str) -> dict:
    """Connect using official livekit.rtc SDK and capture every observable event."""
    from livekit import rtc

    results = {
        "test": "native_livekit_rtc",
        "success": False,
        "connect_exception": None,
        "connect_duration_ms": 0.0,
        "events": [],
        "final_connection_state": None,
    }
    t0 = time.monotonic()

    room = rtc.Room()
    events: List[dict] = results["events"]

    @room.on("connected")
    def on_connected():
        events.append({"event": "connected", "t": round(time.monotonic() - t0, 3)})
        _log("[NATIVE] EVENT: connected")

    @room.on("disconnected")
    def on_disconnected():
        events.append({"event": "disconnected", "t": round(time.monotonic() - t0, 3)})
        _log("[NATIVE] EVENT: disconnected")

    @room.on("connection_quality_changed")
    def on_quality_changed(quality):
        events.append({"event": "quality_changed", "t": round(time.monotonic() - t0, 3), "quality": str(quality)})
        _log("[NATIVE] EVENT: quality_changed=%s", quality)

    @room.on("reconnecting")
    def on_reconnecting():
        events.append({"event": "reconnecting", "t": round(time.monotonic() - t0, 3)})
        _log("[NATIVE] EVENT: reconnecting")

    @room.on("reconnected")
    def on_reconnected():
        events.append({"event": "reconnected", "t": round(time.monotonic() - t0, 3)})
        _log("[NATIVE] EVENT: reconnected")

    @room.on("track_published")
    def on_track_published(pub, participant):
        events.append({"event": "track_published", "t": round(time.monotonic() - t0, 3), "sid": pub.track_sid})
        _log("[NATIVE] EVENT: track_published sid=%s", pub.track_sid)

    @room.on("track_unpublished")
    def on_track_unpublished(pub, participant):
        events.append({"event": "track_unpublished", "t": round(time.monotonic() - t0, 3), "sid": pub.track_sid})

    @room.on("track_subscribed")
    def on_track_subscribed(track, pub, participant):
        events.append({"event": "track_subscribed", "t": round(time.monotonic() - t0, 3), "sid": pub.track_sid, "kind": track.kind})

    @room.on("participant_connected")
    def on_participant_connected(participant):
        events.append({"event": "participant_connected", "t": round(time.monotonic() - t0, 3), "identity": participant.identity})

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        events.append({"event": "participant_disconnected", "t": round(time.monotonic() - t0, 3), "identity": participant.identity})

    _log("[NATIVE] Starting room.connect() ...")
    try:
        await room.connect(
            url,
            token,
            options=rtc.RoomOptions(auto_subscribe=True),
        )
        results["success"] = True
        results["connect_duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _log("[NATIVE] room.connect() SUCCEEDED in %.1f ms", results["connect_duration_ms"])
    except Exception as exc:
        results["connect_exception"] = f"{type(exc).__name__}: {exc}"
        results["connect_duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _log("[NATIVE] room.connect() FAILED after %.1f ms: %s\n%s",
             results["connect_duration_ms"], exc, traceback.format_exc())

    results["final_connection_state"] = str(room.connection_state)
    await asyncio.sleep(3)

    try:
        await room.disconnect()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# 3. aiortc diagnostic (pure-Python WebRTC for comparison)
# ---------------------------------------------------------------------------

async def diag_aiortc_ice() -> dict:
    """Use aiortc to gather ICE candidates — proves pure-Python UDP works."""
    from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

    results = {
        "test": "aiortc_ice_gather",
        "success": False,
        "ice_candidates": [],
        "gather_duration_ms": 0.0,
        "error": None,
    }
    t0 = time.monotonic()

    pc = RTCPeerConnection(configuration=RTCConfiguration(
        iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
    ))

    done = asyncio.Event()

    @pc.on("icecandidate")
    def on_ice(candidate):
        if candidate is None:
            done.set()
        else:
            cand_str = str(candidate)
            results["ice_candidates"].append(cand_str)
            _log("[AIORTC] ICE candidate: %s", cand_str)

    @pc.on("iceconnectionstatechange")
    def on_ice_state():
        _log("[AIORTC] iceConnectionState: %s", pc.iceConnectionState)

    try:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        await asyncio.wait_for(done.wait(), timeout=10.0)
        results["success"] = True
        results["gather_duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _log("[AIORTC] ICE gathering completed in %.1f ms (%d candidates)",
             results["gather_duration_ms"], len(results["ice_candidates"]))
    except asyncio.TimeoutError:
        results["error"] = "ICE gathering timeout (10s)"
        results["gather_duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _log("[AIORTC] ICE gathering TIMEOUT after %.1f ms", results["gather_duration_ms"])
    except Exception as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
        _log("[AIORTC] ICE gathering ERROR: %s\n%s", exc, traceback.format_exc())
    finally:
        await pc.close()

    return results


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------

async def run_all_diagnostics(livekit_url: str, livekit_token: str) -> dict:
    _log("=" * 60)
    _log("ROOT CAUSE DIAGNOSTIC START")
    _log("Container runtime: %s", os.environ.get("MODAL_RUNTIME", "unknown"))
    _log("=" * 60)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diagnostics": [],
    }

    _log("\n--- UDP Socket Test ---")
    report["diagnostics"].append(await diag_udp())

    _log("\n--- Native livekit.rtc Test ---")
    report["diagnostics"].append(await diag_native_livekit(livekit_url, livekit_token))

    _log("\n--- aiortc ICE Test ---")
    report["diagnostics"].append(await diag_aiortc_ice())

    _log("\n" + "=" * 60)
    _log("ROOT CAUSE DIAGNOSTIC COMPLETE")
    _log("=" * 60)

    report["container_logs"] = _log_buffer.getvalue()

    return report
'''


@app.function(
    gpu="L4",
    timeout=300,
    secrets=[modal.Secret.from_name("livekit-secret")],
)
@modal.web_server(port=8080, startup_timeout=30)
def web_server_diagnostics():
    """Run diagnostics inside a Modal @modal.web_server() container."""
    import asyncio
    import json
    import os
    import threading
    from aiohttp import web

    os.environ["MODAL_RUNTIME"] = "@modal.web_server()"

    livekit_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not livekit_url or not api_key or not api_secret:
        error_json = json.dumps({"error": "Missing LiveKit env vars"})

        async def _err_handle(request):
            return web.Response(text=error_json, content_type="application/json")

        def _err_run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _start():
                app = web.Application()
                app.router.add_get("/", _err_handle)
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "0.0.0.0", 8080)
                await site.start()
                await asyncio.Event().wait()

            loop.run_until_complete(_start())

        t = threading.Thread(target=_err_run, daemon=True)
        t.start()
        return

    from livekit.api import AccessToken, VideoGrants
    token = (
        AccessToken(api_key, api_secret)
        .with_identity("modal-diag-worker")
        .with_name("Modal Root-Cause Test")
        .with_grants(
            VideoGrants(
                room_join=True,
                room="modal-diag-room",
                can_publish=True,
                can_subscribe=True,
            )
        )
    ).to_jwt()

    exec_globals = {"__name__": "__main__"}
    exec(_DIAGNOSTIC_CODE, exec_globals)
    run_all_diag = exec_globals["run_all_diagnostics"]

    async def handle(request):
        report = await run_all_diag(livekit_url, token)
        return web.Response(
            text=json.dumps(report, indent=2),
            content_type="application/json",
        )

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _start():
            application = web.Application()
            application.router.add_get("/", handle)
            runner = web.AppRunner(application)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", 8080)
            await site.start()
            await asyncio.Event().wait()

        loop.run_until_complete(_start())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
