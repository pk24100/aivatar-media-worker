"""
Concurrent Modal GPU Worker Test - N sessions with FPS measurement.

Launches N concurrent sessions against a Modal worker, streams audio via
WebSocket to each, and simultaneously connects as a LiveKit viewer to
measure actual video FPS per session. No browser tabs needed.

FPS is measured by subscribing to the worker's video track via the
livekit.rtc SDK and counting incoming VideoFrame events over a time
window. This gives the true end-to-end FPS the worker is delivering.
The first FPS window includes startup effects and is excluded from the
steady-state verdict when later windows are available.

When --modal-app-name is provided, logs are fetched post-test via
`modal app logs` and parsed for VIDEO_PUBLISH_METRICS to extract
generated FPS (liveFrames only, excluding idle/repeated frames).
This is the primary metric for PASS/FAIL verdict.

Configuration:
    Fill in scripts/.env.test (same as modal_direct_test.py).

Usage:
    pip install livekit-api websockets aiohttp

    # Test 3 concurrent sessions (default):
    python scripts/modal_concurrent_test.py --wav test_audio.wav

    # Test with 5 sessions (stress test beyond pool size):
    python scripts/modal_concurrent_test.py --wav test_audio.wav --sessions 5

    # Test with log fetching for generated FPS:
    python scripts/modal_concurrent_test.py --wav test_audio.wav --sessions 5 \
        --gpu-type L40S --modal-app-name aivatar-worker-stress

    # Ramp-up mode (interactive, starts at 3, asks to continue to 4, 5, ...):
    python scripts/modal_concurrent_test.py --wav test_audio.wav --sessions 3 \
        --ramp-up --gpu-type L40S --modal-app-name aivatar-worker-stress

    # Override Modal URL:
    python scripts/modal_concurrent_test.py --modal-url https://other-url.modal.run --wav test_audio.wav
"""

import argparse
import asyncio
import json
import os
import pathlib
import re
import subprocess
import sys
import multiprocessing
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Auto-load scripts/.env.test  (same as modal_direct_test.py)
# ---------------------------------------------------------------------------

def _load_env_file():
    candidates = [
        pathlib.Path(__file__).resolve().parent / ".env.test",
        pathlib.Path.cwd() / "scripts" / ".env.test",
        pathlib.Path.cwd() / ".env.test",
    ]
    for path in candidates:
        if path.is_file():
            loaded = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value
                        loaded += 1
            print(f"[env] Loaded {loaded} vars from {path}")
            return
    print("[env] No .env.test found - using existing env vars / CLI flags")

_load_env_file()

# ---------------------------------------------------------------------------
# Token minting  (same as modal_direct_test.py)
# ---------------------------------------------------------------------------

def mint_livekit_token(api_key: str, api_secret: str, room_name: str,
                       identity: str, *, can_publish: bool = True,
                       ttl_seconds: int = 1800) -> str:
    from livekit.api import AccessToken, VideoGrants
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=can_publish,
            can_subscribe=True,
        ))
    )
    return token.to_jwt()


def mint_ws_token(payload: dict, secret: str) -> str:
    import jwt as pyjwt
    now = int(time.time())
    claims = {
        **payload,
        "iat": now,
        "exp": now + 300,
        "jti": uuid.uuid4().hex,
    }
    return pyjwt.encode(claims, secret, algorithm="HS256")

# ---------------------------------------------------------------------------
# Worker HTTP helpers
# ---------------------------------------------------------------------------

async def wait_for_ready(modal_url: str, timeout: int = 900):
    """One-time /readyz check: wait up to 5 minutes for a ready response.

    Retries the HTTP connection until the server responds with ready=true
    and at least 1 available pipeline, or until the timeout expires.
    Once ready, returns the response data. Does NOT poll continuously after.
    """
    import aiohttp
    url = f"{modal_url}/readyz"
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    data = await resp.json()
                    ready = data.get("ready", False)
                    avail = data.get("availablePipelines", 0)
                    pool_size = data.get("poolSize", 0)
                    active = data.get("activeSessions", 0)
                    print(f"  [{attempt}] ready={ready}  availablePipelines={avail}  poolSize={pool_size}  activeSessions={active}")
                    if ready and avail > 0:
                        return data
                    # Server responded but not ready yet - wait 5 min before retrying
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(300, remaining))
        except Exception as e:
            print(f"  [{attempt}] /readyz connection error: {repr(e)}")
            # Wait 5 min before retrying (matches 'not ready' path)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(300, remaining))
    raise TimeoutError(f"Worker did not become ready within {timeout}s")


async def claim_room(modal_url: str) -> dict:
    import aiohttp
    url = f"{modal_url}/room/claim"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={},
                                timeout=aiohttp.ClientTimeout(total=15)) as resp:
            text = await resp.text()
            if resp.status >= 400:
                return {"roomName": None, "workerToken": None, "clientToken": None}
            return json.loads(text)


async def start_session(modal_url: str, payload: dict) -> dict:
    import aiohttp
    url = f"{modal_url}/sessions/start"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=60)) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"sessions/start failed with status {resp.status}")
            return json.loads(text)


async def end_session(modal_url: str, session_id: str):
    import aiohttp
    url = f"{modal_url}/sessions/{session_id}/end"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                await resp.read()
                print(f"  [end] {session_id}: status={resp.status}")
    except Exception as exc:
        print(f"  [end] {session_id}: failed error={exc.__class__.__name__}")

# ---------------------------------------------------------------------------
# Keyboard watcher (Windows msvcrt, same as modal_direct_test.py)
# ---------------------------------------------------------------------------

async def watch_keyboard_mp(stop_event: multiprocessing.Event):
    """Windows-only non-blocking key watcher. Sets mp stop_event when 'q' is pressed."""
    try:
        import msvcrt
    except ImportError:
        return
    while not stop_event.is_set():
        if msvcrt.kbhit():
            ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
            if ch == "q":
                stop_event.set()
                print("\n  'q' pressed - ending all sessions...")
                break
        await asyncio.sleep(0.1)

# ---------------------------------------------------------------------------
# Per-session result
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    session_id: str = ""
    room_name: str = ""
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    error: str = ""
    # FPS metrics (total frames including idle - from LiveKit viewer)
    frames_received: int = 0
    fps_samples: list = field(default_factory=list)  # FPS measured per window
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    steady_state_fps_samples: list = field(default_factory=list)
    steady_state_avg_fps: float = 0.0
    steady_state_min_fps: float = 0.0
    steady_state_max_fps: float = 0.0
    # Timing
    start_time: float = 0.0
    first_frame_time: float = 0.0
    end_time: float = 0.0

# ---------------------------------------------------------------------------
# LiveKit viewer: subscribes to video track and counts frames
# ---------------------------------------------------------------------------

async def viewer_measure_fps(
    livekit_url: str,
    viewer_token: str,
    session_id: str,
    result: SessionResult,
    stop_event: asyncio.Event,
    duration_seconds: int = 60,
    fps_window: float = 5.0,
):
    """Connect as a LiveKit viewer, subscribe to video track, count frames."""
    from livekit import rtc

    room = rtc.Room()
    frame_count = 0
    window_start = time.monotonic()
    connected = False
    video_track_received = False

    @room.on("track_subscribed")
    def on_track_subscribed(track, pub, participant):
        nonlocal video_track_received
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            video_track_received = True
            asyncio.create_task(_count_video_frames(track))
            pub_sid = getattr(pub, "sid", getattr(pub, "track_sid", "unknown"))
            print(f"  [{session_id}] Video track subscribed: {pub_sid}")

    async def _count_video_frames(track):
        nonlocal frame_count
        try:
            video_stream = rtc.VideoStream(track)
            async for frame in video_stream:
                frame_count += 1
                result.frames_received += 1
                if result.first_frame_time == 0.0:
                    result.first_frame_time = time.monotonic()
                    latency = result.first_frame_time - result.start_time
                    print(f"  [{session_id}] First video frame received (latency={latency:.1f}s)")
        except Exception as exc:
            print(f"  [{session_id}] Video stream error={exc.__class__.__name__}")

    try:
        await room.connect(
            livekit_url,
            viewer_token,
            options=rtc.RoomOptions(auto_subscribe=True),
        )
        connected = True
        result.start_time = time.monotonic()
        print(f"  [{session_id}] Viewer connected, waiting for video...")

        # Wait for video track with timeout
        video_wait_deadline = time.monotonic() + 90
        while not video_track_received and time.monotonic() < video_wait_deadline:
            await asyncio.sleep(0.5)

        if not video_track_received:
            print(f"  [{session_id}] WARNING: No video track received within 90s")
            result.status = "FAILED"
            result.error = "No video track received"
            return

        # Measure FPS over windows (exit early if stop_event is set)
        while time.monotonic() - result.start_time < duration_seconds:
            if stop_event.is_set():
                break
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=fps_window)
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # window elapsed, measure FPS
            elapsed = time.monotonic() - window_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            result.fps_samples.append(round(fps, 1))
            print(f"  [{session_id}] FPS: {fps:.1f} (frames={frame_count}, window={elapsed:.1f}s)")
            frame_count = 0
            window_start = time.monotonic()

    except Exception as exc:
        print(f"  [{session_id}] Viewer error={exc.__class__.__name__}")
        result.status = "FAILED"
        result.error = f"Viewer failed ({exc.__class__.__name__})"
    finally:
        if connected:
            try:
                await room.disconnect()
            except Exception:
                pass

    result.end_time = time.monotonic()
    if result.fps_samples:
        result.avg_fps = round(sum(result.fps_samples) / len(result.fps_samples), 1)
        result.min_fps = min(result.fps_samples)
        result.max_fps = max(result.fps_samples)
        if len(result.fps_samples) > 1:
            result.steady_state_fps_samples = result.fps_samples[1:]
        else:
            result.steady_state_fps_samples = list(result.fps_samples)
        if result.steady_state_fps_samples:
            result.steady_state_avg_fps = round(
                sum(result.steady_state_fps_samples) / len(result.steady_state_fps_samples), 1
            )
            result.steady_state_min_fps = min(result.steady_state_fps_samples)
            result.steady_state_max_fps = max(result.steady_state_fps_samples)
        if result.frames_received <= 0:
            result.status = "FAILED"
            result.error = "Video track subscribed but no frames received"
        elif result.status != "FAILED":
            result.status = "COMPLETED"

# ---------------------------------------------------------------------------
# Audio streamer (same as modal_direct_test.py but async task)
# ---------------------------------------------------------------------------

async def stream_audio_ws(
    ws_url: str,
    wav_path: str,
    ws_token: str,
    session_id: str,
    stop_event: asyncio.Event,
    loop_count: int = -1,
    send_interval: float = 0.02,
    idle_gap_seconds: float = 5.0,
):
    """Stream a WAV file over WebSocket to the worker.

    send_interval controls how fast audio is delivered relative to real-time.
    - 0.1 = real-time (100ms chunk every 100ms) -> engine starves, repeats frames
    - 0.02 = 5x real-time (100ms chunk every 20ms) -> builds buffer like E2E sample site

    After each WAV loop iteration, sends {type: 'end_utterance'} control message
    and waits idle_gap_seconds before next iteration, mimicking the E2E sample
    site's Gnani TTS -> end_utterance -> idle -> speak again flow.
    """
    import websockets
    import wave

    print(f"  [{session_id}] Streaming audio: {wav_path} (send_interval={send_interval}s, idle_gap={idle_gap_seconds}s)")
    try:
        async with websockets.connect(ws_url, subprotocols=[f"facemode.{ws_token}"]) as ws:
            with wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                chunk_frames = sr // 10  # 100ms chunks
                iteration = 0

                while not stop_event.is_set():
                    iteration += 1
                    utterance_id = f"utt-{session_id}-{iteration}"
                    wf.rewind()
                    while True:
                        data = wf.readframes(chunk_frames)
                        if not data:
                            break
                        if stop_event.is_set():
                            break
                        await ws.send(data)
                        await asyncio.sleep(send_interval)

                    # Send end_utterance control message (matches E2E sample site behavior)
                    if not stop_event.is_set():
                        control = json.dumps({"type": "end_utterance", "utteranceId": utterance_id})
                        await ws.send(control)
                        print(f"  [{session_id}] Sent end_utterance (iteration {iteration})")

                    if loop_count >= 0 and iteration >= loop_count:
                        break

                    # Idle gap between utterances (mimics user typing/thinking pause)
                    if not stop_event.is_set():
                        await asyncio.sleep(idle_gap_seconds)

        print(f"  [{session_id}] Audio streaming ended ({iteration} iterations)")
    except Exception as exc:
        print(f"  [{session_id}] Audio stream error={exc.__class__.__name__}")

# ---------------------------------------------------------------------------
# Single session runner
# ---------------------------------------------------------------------------

async def run_single_session(
    modal_url: str,
    livekit_url: str,
    livekit_key: str,
    livekit_secret: str,
    worker_auth_secret: str,
    source_image: str,
    wav_path: str,
    session_idx: int,
    duration_seconds: int,
    stop_event: asyncio.Event,
    audio_send_interval: float = 0.02,
    idle_gap_seconds: float = 5.0,
) -> SessionResult:
    """Run a single session: claim room, start, stream audio, measure FPS."""
    result = SessionResult()
    result.session_id = f"conc-test-{session_idx}-{uuid.uuid4().hex[:6]}"

    try:
        # Step 1: Claim a room (network-safe: falls back to local mint on error)
        try:
            claimed = await claim_room(modal_url)
        except Exception as claim_err:
            print(
                f"  [{result.session_id}] Room claim failed "
                f"(error={claim_err.__class__.__name__}), falling back to local mint"
            )
            claimed = None
        room_name = claimed.get("roomName") if claimed else None
        worker_token = claimed.get("workerToken") if claimed else None
        viewer_token = claimed.get("clientToken") if claimed else None

        if not room_name:
            room_name = f"conc-test-{uuid.uuid4().hex[:8]}"
            worker_token = mint_livekit_token(
                livekit_key, livekit_secret,
                room_name, identity=f"facemode-worker-{result.session_id}",
                can_publish=True, ttl_seconds=1800,
            )

        if not viewer_token:
            viewer_token = mint_livekit_token(
                livekit_key, livekit_secret,
                room_name, identity=f"viewer-manual-{result.session_id}",
                can_publish=False, ttl_seconds=1800,
            )

        # Mint a separate token for the script's FPS-measurement viewer so that
        # joining via meet.livekit.io with viewer_token does not kick the script
        # viewer (LiveKit rejects duplicate identities by kicking the first).
        script_viewer_token = mint_livekit_token(
            livekit_key, livekit_secret,
            room_name, identity=f"viewer-script-{result.session_id}",
            can_publish=False, ttl_seconds=1800,
        )

        result.room_name = room_name
        print(f"  [{result.session_id}] Room: {room_name}")
        print(f"  [{result.session_id}] LiveKit URL: {livekit_url}")
        print(f"  [{result.session_id}] Viewer token minted for the in-process FPS viewer; value not printed")

        # Step 2: Start session on worker
        ingestion_token = f"test-{uuid.uuid4().hex[:8]}"
        payload = {
            "roomName": room_name,
            "livekitToken": worker_token,
            "customLivekitUrl": livekit_url,
            "sourceImage": source_image,
            "streaming": True,
            "sessionId": result.session_id,
            "ingestionToken": ingestion_token,
        }
        start_res = await start_session(modal_url, payload)
        print(f"  [{result.session_id}] Session started: {start_res.get('status', 'unknown')}")
        result.status = "RUNNING"

        # Step 3: Build WS URL and token for audio streaming
        ws_token = mint_ws_token(payload, worker_auth_secret)
        ws_base = modal_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_base}/ws/{result.session_id}"

        # Step 4: Launch viewer FPS measurement + audio streaming concurrently
        viewer_task = asyncio.create_task(
            viewer_measure_fps(
                livekit_url, script_viewer_token, result.session_id,
                result, stop_event, duration_seconds=duration_seconds,
            )
        )

        # Step 4b: Wait for media readiness (first video frame) before streaming audio.
        # The E2E sample site blocks speech until mediaReady=true. Without this gate,
        # audio buffers during avatar preparation and creates an artificial burst.
        print(f"  [{result.session_id}] Waiting for first video frame before audio streaming...")
        media_ready_deadline = time.monotonic() + 90
        while result.first_frame_time == 0.0 and time.monotonic() < media_ready_deadline:
            if stop_event.is_set():
                break
            await asyncio.sleep(0.5)

        if result.first_frame_time == 0.0:
            print(f"  [{result.session_id}] WARNING: No first frame within 90s, starting audio anyway")
        else:
            latency = result.first_frame_time - result.start_time
            print(f"  [{result.session_id}] Media ready (first frame latency={latency:.1f}s), starting audio")

        audio_task = asyncio.create_task(
            stream_audio_ws(
                ws_url, wav_path, ws_token, result.session_id,
                stop_event, loop_count=-1, send_interval=audio_send_interval,
                idle_gap_seconds=idle_gap_seconds,
            )
        )

        # Wait for viewer to finish (duration timeout or stop_event via 'q')
        await viewer_task

        # Stop audio streaming
        audio_task.cancel()
        try:
            await audio_task
        except asyncio.CancelledError:
            pass

        # Step 5: End session
        await end_session(modal_url, result.session_id)

    except Exception as exc:
        print(f"  [{result.session_id}] ERROR: {exc.__class__.__name__}")
        result.status = "FAILED"
        result.error = f"Session failed ({exc.__class__.__name__})"

    return result

# ---------------------------------------------------------------------------
# Log parsing: VIDEO_PUBLISH_METRICS and batched_engine cycle logs
# ---------------------------------------------------------------------------

# Regex for VIDEO_PUBLISH_METRICS lines (with session tag):
# VIDEO_PUBLISH_METRICS session=conc-test-1-abc windowMs=5000 frames=125 effectiveFps=25.0 liveFrames=100 repeatedLiveFrames=5 idleFrames=20 desyncPct=20.0 maxGapMs=40.0 maxCaptureMs=2.1
_VPM_PATTERN = re.compile(
    r"VIDEO_PUBLISH_METRICS\s+"
    r"session=(\S+)\s+"
    r"windowMs=(\d+)\s+"
    r"frames=(\d+)\s+"
    r"effectiveFps=([\d.]+)\s+"
    r"liveFrames=(\d+)\s+"
    r"repeatedLiveFrames=(\d+)\s+"
    r"idleFrames=(\d+)\s+"
    r"desyncPct=([\d.]+)\s+"
    r"maxGapMs=([\d.]+)\s+"
    r"maxCaptureMs=([\d.]+)"
)

# Fallback regex for old-format VPM lines without session tag
_VPM_PATTERN_LEGACY = re.compile(
    r"VIDEO_PUBLISH_METRICS\s+"
    r"windowMs=(\d+)\s+"
    r"frames=(\d+)\s+"
    r"effectiveFps=([\d.]+)\s+"
    r"liveFrames=(\d+)\s+"
    r"repeatedLiveFrames=(\d+)\s+"
    r"idleFrames=(\d+)\s+"
    r"desyncPct=([\d.]+)\s+"
    r"maxGapMs=([\d.]+)\s+"
    r"maxCaptureMs=([\d.]+)"
)

# Regex for batched_engine cycle logs:
# cycle=30 batch=3 sessions=['s1','s2','s3'] embed=45.2ms infer=620.1ms xfer=12.3ms total=705.6ms
_CYCLE_PATTERN = re.compile(
    r"cycle=(\d+)\s+batch=(\d+)\s+sessions=\[.*?\]\s+"
    r"embed=([\d.]+)ms\s+infer=([\d.]+)ms\s+"
    r"xfer=([\d.]+)ms\s+total=([\d.]+)ms"
)


def parse_modal_logs(log_file_path: str) -> dict:
    """Parse a saved Modal log file for VIDEO_PUBLISH_METRICS and cycle logs.

    Returns dict with:
        - per_session_generated_fps: {session_id: {avg, min, max, samples, idle_pct}}
        - generated_fps_samples: list of floats (all sessions combined, for backward compat)
        - generated_fps_avg, generated_fps_min, generated_fps_max
        - per_session_inference_ms: list of floats (total_ms / batch_size per cycle)
        - inference_ms_samples: list of floats (raw total cycle ms, for backward compat)
        - inference_ms_avg, inference_ms_p95, inference_ms_max (per-session, i.e. total/batch_size)
        - batch_inference_ms_avg, batch_inference_ms_p95, batch_inference_ms_max (raw batch totals)
        - batch_size_avg
        - effective_fps_samples: list of floats (total frames / windowSecs)
        - idle_frames_pct_avg: average percentage of idle+repeated frames
    """
    per_session_vpm = {}  # session_id -> list of (gen_fps, idle_pct)
    generated_fps_samples = []
    effective_fps_samples = []
    idle_pcts = []
    inference_ms_samples = []  # raw batch total ms (backward compat)
    per_session_inference_ms = []  # total_ms / batch_size (the real per-session metric)
    batch_sizes = []

    try:
        with open(log_file_path, encoding="utf-8") as f:
            for line in f:
                vpm_match = _VPM_PATTERN.search(line)
                if vpm_match:
                    session_id = vpm_match.group(1)
                    window_ms = int(vpm_match.group(2))
                    total_frames = int(vpm_match.group(3))
                    eff_fps = float(vpm_match.group(4))
                    live_frames = int(vpm_match.group(5))
                    repeated = int(vpm_match.group(6))
                    idle = int(vpm_match.group(7))

                    if window_ms > 0:
                        gen_fps = live_frames / (window_ms / 1000.0)
                        gen_fps_r = round(gen_fps, 1)
                        generated_fps_samples.append(gen_fps_r)
                        effective_fps_samples.append(eff_fps)

                        idle_pct = 0.0
                        if total_frames > 0:
                            idle_pct = ((repeated + idle) / total_frames) * 100.0
                            idle_pcts.append(round(idle_pct, 1))

                        if session_id not in per_session_vpm:
                            per_session_vpm[session_id] = []
                        per_session_vpm[session_id].append((gen_fps_r, round(idle_pct, 1)))
                else:
                    vpm_legacy = _VPM_PATTERN_LEGACY.search(line)
                    if vpm_legacy:
                        window_ms = int(vpm_legacy.group(1))
                        live_frames = int(vpm_legacy.group(4))
                        repeated = int(vpm_legacy.group(5))
                        idle = int(vpm_legacy.group(6))
                        eff_fps = float(vpm_legacy.group(3))

                        if window_ms > 0:
                            gen_fps = live_frames / (window_ms / 1000.0)
                            generated_fps_samples.append(round(gen_fps, 1))
                            effective_fps_samples.append(eff_fps)

                            total_frames = int(vpm_legacy.group(2))
                            if total_frames > 0:
                                idle_pct = ((repeated + idle) / total_frames) * 100.0
                                idle_pcts.append(round(idle_pct, 1))

                cycle_match = _CYCLE_PATTERN.search(line)
                if cycle_match:
                    batch_size = int(cycle_match.group(2))
                    total_ms = float(cycle_match.group(6))
                    inference_ms_samples.append(round(total_ms, 1))
                    batch_sizes.append(batch_size)
                    if batch_size > 0:
                        per_session_inference_ms.append(round(total_ms / batch_size, 1))
    except Exception as e:
        print(f"  [log-parse] Error parsing log file: {e}")

    # Build per-session generated FPS summary
    per_session_generated_fps = {}
    for sid, samples in per_session_vpm.items():
        fps_vals = [s[0] for s in samples]
        idle_vals = [s[1] for s in samples]
        per_session_generated_fps[sid] = {
            "avg": round(sum(fps_vals) / len(fps_vals), 1),
            "min": min(fps_vals),
            "max": max(fps_vals),
            "samples": fps_vals,
            "idle_pct_avg": round(sum(idle_vals) / len(idle_vals), 1) if idle_vals else 0.0,
        }

    result = {
        "per_session_generated_fps": per_session_generated_fps,
        "generated_fps_samples": generated_fps_samples,
        "effective_fps_samples": effective_fps_samples,
        "idle_frames_pct_avg": round(sum(idle_pcts) / len(idle_pcts), 1) if idle_pcts else 0.0,
        "inference_ms_samples": inference_ms_samples,
        "per_session_inference_ms": per_session_inference_ms,
        "batch_size_avg": round(sum(batch_sizes) / len(batch_sizes), 1) if batch_sizes else 0.0,
    }

    if generated_fps_samples:
        result["generated_fps_avg"] = round(sum(generated_fps_samples) / len(generated_fps_samples), 1)
        result["generated_fps_min"] = min(generated_fps_samples)
        result["generated_fps_max"] = max(generated_fps_samples)
    else:
        result["generated_fps_avg"] = 0.0
        result["generated_fps_min"] = 0.0
        result["generated_fps_max"] = 0.0

    # Per-session inference (total_ms / batch_size) - the real budget metric
    if per_session_inference_ms:
        result["inference_ms_avg"] = round(sum(per_session_inference_ms) / len(per_session_inference_ms), 1)
        sorted_ms = sorted(per_session_inference_ms)
        p95_idx = int(len(sorted_ms) * 0.95)
        result["inference_ms_p95"] = sorted_ms[min(p95_idx, len(sorted_ms) - 1)]
        result["inference_ms_max"] = max(per_session_inference_ms)
    else:
        result["inference_ms_avg"] = 0.0
        result["inference_ms_p95"] = 0.0
        result["inference_ms_max"] = 0.0

    # Raw batch totals (for backward compat / debugging)
    if inference_ms_samples:
        result["batch_inference_ms_avg"] = round(sum(inference_ms_samples) / len(inference_ms_samples), 1)
        sorted_batch = sorted(inference_ms_samples)
        p95_idx = int(len(sorted_batch) * 0.95)
        result["batch_inference_ms_p95"] = sorted_batch[min(p95_idx, len(sorted_batch) - 1)]
        result["batch_inference_ms_max"] = max(inference_ms_samples)
    else:
        result["batch_inference_ms_avg"] = 0.0
        result["batch_inference_ms_p95"] = 0.0
        result["batch_inference_ms_max"] = 0.0

    return result


def discover_container_ids(app_name: str = None, timeout: int = 15) -> list:
    """Discover running Modal container IDs via `modal container list --json`.

    If app_name is provided, filters containers for that app.
    Returns a list of container ID strings (e.g. ['ta-01KZ64KERQBVYNZHC5ATXHRPXR']).
    """
    cmd_parts = ["modal", "container", "list", "--json"]
    print(f"  [container-discover] Running: {' '.join(cmd_parts)}")
    try:
        proc = subprocess.run(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        if proc.returncode != 0:
            print(f"  [container-discover] modal container list failed (code={proc.returncode}): {proc.stderr[:200]}")
            return []
        containers = json.loads(proc.stdout)
        if not isinstance(containers, list):
            containers = [containers]
        ids = []
        for c in containers:
            cid = c.get("container_id") or c.get("id") or c.get("Id") or ""
            app = c.get("app_name") or c.get("app") or c.get("App") or ""
            if app_name and app_name not in str(app):
                continue
            if cid:
                ids.append(cid)
        print(f"  [container-discover] Found {len(ids)} container(s): {ids}")
        return ids
    except subprocess.TimeoutExpired:
        print(f"  [container-discover] Timed out after {timeout}s")
    except Exception as e:
        print(f"  [container-discover] Failed: {e}")
    return []


def fetch_container_logs(modal_app_name: str, container_id: str, log_file_path: str, timeout: int = 60):
    """One-shot fetch of logs for a specific container.

    Uses: modal app logs <app_name> --timestamps --container <container_id> --tail 10000
    --tail 4000 fetches up to 4000 most recent log lines (works on stopped containers).
    Saves raw output to log_file_path. Returns True on success.
    """
    cmd_parts = ["modal", "app", "logs", modal_app_name, "--timestamps", "--container", container_id, "--tail", "10000"]
    cmd_str = " ".join(cmd_parts)
    print(f"  [log-fetch] Fetching: {cmd_str}")
    try:
        proc = subprocess.run(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        log_content = proc.stdout
        with open(log_file_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(log_content)
        line_count = log_content.count("\n")
        print(f"  [log-fetch] Saved {len(log_content)} bytes ({line_count} lines) to {log_file_path}")
        return True
    except subprocess.TimeoutExpired:
        print(f"  [log-fetch] Timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"  [log-fetch] Failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Multi-process session worker (child process entry point)
# ---------------------------------------------------------------------------

def _session_worker(args: dict, mp_stop_event: multiprocessing.Event,
                    result_queue: multiprocessing.Queue):
    """Child process entry point. Runs one session in its own event loop.

    This function is top-level so it is picklable for Windows spawn method.
    """
    session_idx = args["session_idx"]
    session_id = f"conc-test-{session_idx}-{uuid.uuid4().hex[:6]}"

    async def _run():
        async_stop = asyncio.Event()

        async def _mp_stop_bridge():
            while not mp_stop_event.is_set():
                await asyncio.sleep(0.5)
            async_stop.set()

        bridge_task = asyncio.create_task(_mp_stop_bridge())
        try:
            res = await run_single_session(
                modal_url=args["modal_url"],
                livekit_url=args["livekit_url"],
                livekit_key=args["livekit_key"],
                livekit_secret=args["livekit_secret"],
                worker_auth_secret=args["worker_auth_secret"],
                source_image=args["source_image"],
                wav_path=args["wav_path"],
                session_idx=session_idx,
                duration_seconds=args["duration_seconds"],
                stop_event=async_stop,
                audio_send_interval=args["audio_send_interval"],
                idle_gap_seconds=args["idle_gap_seconds"],
            )
            result_queue.put({
                "session_id": res.session_id,
                "room_name": res.room_name,
                "status": res.status,
                "error": res.error,
                "frames_received": res.frames_received,
                "fps_samples": res.fps_samples,
                "avg_fps": res.avg_fps,
                "min_fps": res.min_fps,
                "max_fps": res.max_fps,
                "steady_state_fps_samples": res.steady_state_fps_samples,
                "steady_state_avg_fps": res.steady_state_avg_fps,
                "steady_state_min_fps": res.steady_state_min_fps,
                "steady_state_max_fps": res.steady_state_max_fps,
                "start_time": res.start_time,
                "first_frame_time": res.first_frame_time,
                "end_time": res.end_time,
            })
        except Exception as exc:
            result_queue.put({
                "session_id": session_id,
                "room_name": "",
                "status": "FAILED",
                "error": f"Session process failed ({exc.__class__.__name__})",
                "frames_received": 0,
                "fps_samples": [],
                "avg_fps": 0.0,
                "min_fps": 0.0,
                "max_fps": 0.0,
                "steady_state_fps_samples": [],
                "steady_state_avg_fps": 0.0,
                "steady_state_min_fps": 0.0,
                "steady_state_max_fps": 0.0,
                "start_time": 0.0,
                "first_frame_time": 0.0,
                "end_time": 0.0,
            })
        finally:
            bridge_task.cancel()

    asyncio.run(_run())


def _dict_to_session_result(d: dict) -> SessionResult:
    """Convert a dict from the multiprocessing queue back to SessionResult."""
    r = SessionResult()
    r.session_id = d.get("session_id", "")
    r.room_name = d.get("room_name", "")
    r.status = d.get("status", "FAILED")
    r.error = d.get("error", "")
    r.frames_received = d.get("frames_received", 0)
    r.fps_samples = d.get("fps_samples", [])
    r.avg_fps = d.get("avg_fps", 0.0)
    r.min_fps = d.get("min_fps", 0.0)
    r.max_fps = d.get("max_fps", 0.0)
    r.steady_state_fps_samples = d.get("steady_state_fps_samples", [])
    r.steady_state_avg_fps = d.get("steady_state_avg_fps", 0.0)
    r.steady_state_min_fps = d.get("steady_state_min_fps", 0.0)
    r.steady_state_max_fps = d.get("steady_state_max_fps", 0.0)
    r.start_time = d.get("start_time", 0.0)
    r.first_frame_time = d.get("first_frame_time", 0.0)
    r.end_time = d.get("end_time", 0.0)
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_test_session(
    modal_url: str,
    livekit_url: str,
    livekit_key: str,
    livekit_secret: str,
    worker_auth_secret: str,
    source_image: str,
    wav_path: str,
    num_sessions: int,
    duration: int,
    stagger: float,
    skip_readyz: bool,
    monitor_readyz: bool,
    audio_send_interval: float,
    idle_gap: float,
) -> list:
    """Run a single test with N concurrent sessions. Returns list of SessionResult."""

    # ---- Step 1: Wait for worker readiness ----
    if not skip_readyz:
        print("--- Step 1: Waiting for worker readiness ---")
        ready_data = await wait_for_ready(modal_url)
        pool_size = ready_data.get("poolSize", "?")
        avail = ready_data.get("availablePipelines", "?")
        print(f"  Worker ready: poolSize={pool_size}, availablePipelines={avail}")
        if isinstance(avail, int) and isinstance(pool_size, int):
            if avail < num_sessions:
                print(f"  WARNING: Requesting {num_sessions} sessions but only {avail} pipelines available.")
                print(f"           Sessions beyond pool size will queue/block.")
        print()
    else:
        print("--- Step 1: Skipped (--skip-readyz) ---\n")

    # ---- Step 2: Launch concurrent sessions as separate processes ----
    print(f"--- Step 2: Launching {num_sessions} concurrent sessions (multi-process) ---\n")

    mp_stop_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()
    processes = []

    for i in range(num_sessions):
        print(f"  Launching session {i+1}/{num_sessions}...")
        worker_args = {
            "modal_url": modal_url,
            "livekit_url": livekit_url,
            "livekit_key": livekit_key,
            "livekit_secret": livekit_secret,
            "worker_auth_secret": worker_auth_secret,
            "source_image": source_image,
            "wav_path": wav_path,
            "session_idx": i + 1,
            "duration_seconds": duration,
            "audio_send_interval": audio_send_interval,
            "idle_gap_seconds": idle_gap,
        }
        proc = multiprocessing.Process(
            target=_session_worker,
            args=(worker_args, mp_stop_event, result_queue),
        )
        proc.start()
        processes.append(proc)
        if i < num_sessions - 1:
            await asyncio.sleep(stagger)

    print(f"\n  All {num_sessions} sessions launched. Measuring FPS for {duration}s...")
    print(f"  Press 'q' + Enter to stop all sessions early.\n")

    # ---- Step 3: Monitor readyz during test ----
    async def monitor_readyz_task():
        import aiohttp
        monitor_url = f"{modal_url}/readyz"
        check_interval = 10
        elapsed = 0
        while elapsed < duration + 30:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(monitor_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        data = await resp.json()
                        print(f"  [monitor] availablePipelines={data.get('availablePipelines', '?')}  "
                              f"activeSessions={data.get('activeSessions', '?')}  "
                              f"prewarmRooms={data.get('prewarmRoomsAvailable', '?')}")
            except Exception as e:
                print(f"  [monitor] readyz error: {e}")
            await asyncio.sleep(check_interval)
            elapsed += check_interval

    monitor_task = None
    if monitor_readyz:
        monitor_task = asyncio.create_task(monitor_readyz_task())
    keyboard_task = asyncio.create_task(watch_keyboard_mp(mp_stop_event))

    # Wait for all child processes to finish (in a thread to not block the event loop)
    def _join_all():
        for proc in processes:
            proc.join()

    await asyncio.to_thread(_join_all)

    # Collect results from the queue
    raw_results = []
    while not result_queue.empty():
        raw_results.append(result_queue.get())

    # Handle any processes that crashed without putting a result on the queue
    for i, proc in enumerate(processes):
        if proc.exitcode is not None and proc.exitcode != 0:
            already_has = any(
                r.get("session_id", "").startswith(f"conc-test-{i+1}-")
                for r in raw_results
            )
            if not already_has:
                raw_results.append({
                    "session_id": f"conc-test-{i+1}-CRASHED",
                    "room_name": "",
                    "status": "FAILED",
                    "error": f"Child process exited with code {proc.exitcode}",
                    "frames_received": 0,
                    "fps_samples": [],
                    "avg_fps": 0.0,
                    "min_fps": 0.0,
                    "max_fps": 0.0,
                    "steady_state_fps_samples": [],
                    "steady_state_avg_fps": 0.0,
                    "steady_state_min_fps": 0.0,
                    "steady_state_max_fps": 0.0,
                    "start_time": 0.0,
                    "first_frame_time": 0.0,
                    "end_time": 0.0,
                })

    # Convert dicts to SessionResult objects
    results = [_dict_to_session_result(d) for d in raw_results]

    # Sort by session_idx to maintain order
    def _sort_key(r):
        try:
            return int(r.session_id.split("-")[2])
        except (IndexError, ValueError):
            return 999
    results.sort(key=_sort_key)

    # Cleanup processes
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        proc.close()

    # Stop monitor and keyboard watcher
    if monitor_task is not None:
        monitor_task.cancel()
    keyboard_task.cancel()
    try:
        if monitor_task is not None:
            await monitor_task
    except asyncio.CancelledError:
        pass
    try:
        await keyboard_task
    except asyncio.CancelledError:
        pass

    return results


def print_results(results: list, num_sessions: int, log_metrics: dict = None):
    """Print session results and summary verdict.

    Args:
        results: List of SessionResult objects from run_test_session
        num_sessions: Number of sessions that were launched
        log_metrics: Optional dict from parse_modal_logs() with generated FPS and inference timing
    """
    print("\n" + "=" * 70)
    print("  RESULTS - Concurrent Session FPS Report")
    print("=" * 70)
    print()

    successful = 0
    failed = 0
    all_fps_samples = []
    steady_state_fps_samples = []

    for i, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"  Session {i+1}: FAILED - {res}")
            failed += 1
            continue

        if res.status == "COMPLETED":
            successful += 1
            all_fps_samples.extend(res.fps_samples)
            steady_state_fps_samples.extend(res.steady_state_fps_samples)
            time_to_first = res.first_frame_time - res.start_time if res.first_frame_time > 0 else 0
            print(f"  Session {i+1} ({res.session_id}):")
            print(f"    Status:           {res.status}")
            print(f"    Room:             {res.room_name}")
            print(f"    Startup latency:  {time_to_first:.1f}s to first frame")
            print(f"    Avg FPS (all):    {res.avg_fps}")
            print(f"    Min FPS (all):    {res.min_fps}")
            print(f"    Max FPS (all):    {res.max_fps}")
            if res.steady_state_fps_samples:
                print(f"    Avg FPS (steady): {res.steady_state_avg_fps}")
                print(f"    Min FPS (steady): {res.steady_state_min_fps}")
                print(f"    Max FPS (steady): {res.steady_state_max_fps}")
            print(f"    FPS samples:      {res.fps_samples}")
            print()
        else:
            failed += 1
            print(f"  Session {i+1} ({res.session_id}):")
            print(f"    Status:           {res.status}")
            print(f"    Error:            {res.error}")
            print()

    # ---- Summary ----
    print("-" * 70)
    print(f"  SUMMARY: {successful}/{num_sessions} sessions completed, {failed} failed")
    print(f"  Target FPS: 25 (from infer_params.yaml tgt_fps)")

    # Viewer-side FPS (total frames including idle)
    viewer_verdict_avg = 0.0
    if all_fps_samples:
        overall_avg = round(sum(all_fps_samples) / len(all_fps_samples), 1)
        overall_min = min(all_fps_samples)
        overall_max = max(all_fps_samples)
        print(f"  Overall Avg FPS (all windows, viewer-side): {overall_avg}")
        print(f"  Overall Min FPS (all windows, viewer-side): {overall_min}")
        print(f"  Overall Max FPS (all windows, viewer-side): {overall_max}")

        viewer_verdict_avg = overall_avg
        if steady_state_fps_samples:
            steady_avg = round(sum(steady_state_fps_samples) / len(steady_state_fps_samples), 1)
            steady_min = min(steady_state_fps_samples)
            steady_max = max(steady_state_fps_samples)
            viewer_verdict_avg = steady_avg
            print(f"  Steady-state Avg FPS (viewer-side): {steady_avg}")
            print(f"  Steady-state Min FPS (viewer-side): {steady_min}")
            print(f"  Steady-state Max FPS (viewer-side): {steady_max}")

    # Per-session generated FPS and per-session inference from Modal logs
    generated_verdict_avg = 0.0
    inference_p95 = 0.0
    inference_budget_ms = 960
    inference_pass = True
    per_session_gen_fps = {}
    if log_metrics:
        print()
        per_session_gen_fps = log_metrics.get("per_session_generated_fps", {})
        if per_session_gen_fps:
            print(f"  Per-session generated FPS (from VPM, excludes idle/repeated):")
            for sid in sorted(per_session_gen_fps.keys()):
                s = per_session_gen_fps[sid]
                print(f"    {sid}: avg={s['avg']}  min={s['min']}  max={s['max']}  idle={s['idle_pct_avg']}%")
            all_gen_fps = [s["avg"] for s in per_session_gen_fps.values()]
            generated_verdict_avg = round(sum(all_gen_fps) / len(all_gen_fps), 1)
            print(f"  Overall generated FPS avg: {generated_verdict_avg}")
            print(f"  Idle frames pct avg: {log_metrics.get('idle_frames_pct_avg', 0.0)}%")
        else:
            gen_fps = log_metrics.get("generated_fps_samples", [])
            if gen_fps:
                generated_verdict_avg = log_metrics["generated_fps_avg"]
                print(f"  Generated FPS (all sessions combined, excludes idle):")
                print(f"    Avg:  {log_metrics['generated_fps_avg']}")
                print(f"    Min:  {log_metrics['generated_fps_min']}")
                print(f"    Max:  {log_metrics['generated_fps_max']}")
                print(f"    Idle frames pct: {log_metrics.get('idle_frames_pct_avg', 0.0)}%")

        per_session_inf = log_metrics.get("per_session_inference_ms", [])
        if per_session_inf:
            inference_p95 = log_metrics["inference_ms_p95"]
            inference_avg = log_metrics["inference_ms_avg"]
            inference_max = log_metrics.get("inference_ms_max", 0.0)
            print(f"  Per-session inference (total_ms / batch_size):")
            print(f"    Avg:  {inference_avg}ms")
            print(f"    P95:  {inference_p95}ms")
            print(f"    Max:  {inference_max}ms")
            print(f"    Budget: {inference_budget_ms}ms")
            inference_pass = inference_p95 < inference_budget_ms
            if inference_pass:
                print(f"    -> PASS (P95 {inference_p95}ms < {inference_budget_ms}ms)")
            else:
                print(f"    -> FAIL (P95 {inference_p95}ms >= {inference_budget_ms}ms)")
            if inference_max >= inference_budget_ms:
                print(f"    -> WARNING: Max {inference_max}ms >= {inference_budget_ms}ms budget (transient spike)")

        batch_p95 = log_metrics.get("batch_inference_ms_p95", 0.0)
        if batch_p95 > 0:
            print(f"  Raw batch cycle (for reference):")
            print(f"    Avg:  {log_metrics.get('batch_inference_ms_avg', 0.0)}ms")
            print(f"    P95:  {batch_p95}ms")
            print(f"    Max:  {log_metrics.get('batch_inference_ms_max', 0.0)}ms")
            print(f"    Batch size avg: {log_metrics.get('batch_size_avg', 0.0)}")

    # ---- Verdict (based on per-session generated FPS + inference budget) ----
    print()
    verdict_avg = generated_verdict_avg if generated_verdict_avg > 0 else viewer_verdict_avg
    verdict_basis = "per-session generated FPS + per-session inference < 960ms" if per_session_gen_fps else ("generated FPS (excludes idle)" if generated_verdict_avg > 0 else "viewer FPS (includes idle)")

    fps_pass = verdict_avg >= 20
    all_sessions_pass = failed == 0

    if fps_pass and inference_pass and all_sessions_pass:
        verdict = "PASS"
        print(f"  VERDICT: PASS - {num_sessions} sessions, generated FPS avg {verdict_avg} >= 20, inference P95 {inference_p95}ms < {inference_budget_ms}ms")
    elif fps_pass and all_sessions_pass and not inference_pass:
        verdict = "MARGINAL"
        print(f"  VERDICT: MARGINAL - FPS ok ({verdict_avg}) but inference P95 {inference_p95}ms >= {inference_budget_ms}ms budget")
    elif verdict_avg >= 15:
        verdict = "MARGINAL"
        print(f"  VERDICT: MARGINAL - FPS degraded ({verdict_avg}). Consider reducing concurrency.")
    else:
        verdict = "FAIL"
        print(f"  VERDICT: FAIL - generated FPS {verdict_avg} too low for {num_sessions} sessions.")
        print(f"           Reduce --sessions or check GPU utilization.")

    if not all_sessions_pass:
        verdict = "FAIL"
        print(f"  VERDICT: FAIL - {failed} session(s) failed.")

    print(f"  Verdict basis: {verdict_basis}")
    print("-" * 70)

    return {
        "verdict": verdict,
        "verdict_avg_fps": verdict_avg,
        "verdict_basis": verdict_basis,
        "successful": successful,
        "failed": failed,
        "generated_fps_avg": generated_verdict_avg,
        "viewer_fps_avg": viewer_verdict_avg,
        "inference_ms_p95": inference_p95,
        "inference_ms_max": log_metrics.get("inference_ms_max", 0.0) if log_metrics else 0.0,
        "inference_budget_ms": inference_budget_ms,
        "inference_pass": inference_pass,
        "per_session_generated_fps": per_session_gen_fps,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Concurrent Modal GPU worker test with FPS measurement")
    parser.add_argument("--modal-url",
                        default=os.getenv("MODAL_WORKER_BASE_URL"),
                        help="Modal worker base URL")
    parser.add_argument("--livekit-url",
                        default=os.getenv("LIVEKIT_URL"),
                        help="LiveKit server URL (wss://...)")
    parser.add_argument("--livekit-key",
                        default=os.getenv("LIVEKIT_API_KEY"),
                        help="LiveKit API key")
    parser.add_argument("--livekit-secret",
                        default=os.getenv("LIVEKIT_API_SECRET"),
                        help="LiveKit API secret")
    parser.add_argument("--worker-auth-secret",
                        default=os.getenv("WORKER_AUTH_SECRET"),
                        help="Worker auth secret used to sign /ws JWTs. Must match Modal secret WORKER_AUTH_SECRET.")
    parser.add_argument("--source-image",
                        default=os.getenv("SOURCE_IMAGE",
                            "https://i.postimg.cc/594x5VRK/Gemini-Generated-Image-e3xw6se3xw6se3xw.png?dl=1"),
                        help="Avatar source image URL")
    parser.add_argument("--wav",
                        required=True,
                        help="Path to WAV file to stream")
    parser.add_argument("--sessions", type=int, default=3,
                        help="Number of concurrent sessions to launch (default: 3). "
                             "This is the key variable - change it to test different pipeline counts.")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds to measure FPS (default: 60)")
    parser.add_argument("--stagger", type=float, default=3.0,
                        help="Seconds to wait between launching sessions (default: 3.0)")
    parser.add_argument("--skip-readyz", action="store_true",
                        help="Skip the readyz polling step")
    parser.add_argument("--monitor-readyz", action="store_true",
                        help="Poll /readyz during the active test run (disabled by default to avoid waking extra containers)")
    parser.add_argument("--audio-send-interval", type=float, default=0.02,
                        help="Seconds between audio chunk sends (default: 0.02 = 5x real-time). "
                             "0.1 = real-time (causes engine starvation). "
                             "0.02 = 5x real-time (matches E2E sample site buffer buildup).")
    parser.add_argument("--idle-gap", type=float, default=5.0,
                        help="Seconds of idle silence between WAV loop iterations (default: 5.0). "
                             "Mimics the user typing/thinking pause between utterances in the E2E sample site. "
                             "Set to 0 to disable (continuous audio, less realistic but more stress).")
    # New flags for GPU concurrency testing
    parser.add_argument("--gpu-type", default="unknown",
                        help="GPU type being tested (e.g. L40S, A100-40GB, H100). Used for labeling results only.")
    parser.add_argument("--ramp-up", action="store_true",
                        help="Ramp-up mode: start at --sessions, after each run ask whether to continue "
                             "to sessions+1. Interactive prompt in terminal.")
    parser.add_argument("--modal-app-name", default="aivatar-worker-stress",
                        help="Modal app name for fetching logs post-test (default: aivatar-worker-stress). "
                             "When set, logs are fetched and parsed for VIDEO_PUBLISH_METRICS.")
    parser.add_argument("--modal-container-id", default=None,
                        help="Optional Modal container ID (ta-*) to filter logs for a specific container.")
    args = parser.parse_args()

    # Validate required args
    missing = []
    if not args.modal_url:
        missing.append("--modal-url / MODAL_WORKER_BASE_URL")
    if not args.livekit_url:
        missing.append("--livekit-url / LIVEKIT_URL")
    if not args.livekit_key:
        missing.append("--livekit-key / LIVEKIT_API_KEY")
    if not args.livekit_secret:
        missing.append("--livekit-secret / LIVEKIT_API_SECRET")
    if not args.worker_auth_secret:
        missing.append("--worker-auth-secret / WORKER_AUTH_SECRET")
    if missing:
        print("ERROR: Missing required arguments:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    modal_url = args.modal_url.rstrip("/")

    # Prepare log directory
    log_dir = pathlib.Path(__file__).resolve().parent / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.ramp_up:
        await _run_ramp_up(args, modal_url, log_dir)
    else:
        await _run_single_test(args, modal_url, log_dir, args.sessions)


async def _run_single_test(args, modal_url: str, log_dir: pathlib.Path, num_sessions: int):
    """Run a single test with N sessions, fetch logs, parse, print results, save JSON."""

    print("=" * 70)
    print("  AiVatar - Concurrent Modal Worker Test with FPS Measurement")
    print("=" * 70)
    print(f"  Modal URL:      {modal_url}")
    print(f"  LiveKit URL:    {args.livekit_url}")
    print(f"  GPU type:       {args.gpu_type}")
    print(f"  Sessions:       {num_sessions}")
    print(f"  Duration:       {args.duration}s per session")
    print(f"  Stagger:        {args.stagger}s between launches")
    print(f"  Audio interval: {args.audio_send_interval}s ({0.1 / args.audio_send_interval:.1f}x real-time)")
    print(f"  Idle gap:       {args.idle_gap}s between utterances")
    print(f"  Source Image:   {args.source_image[:50]}...")
    print(f"  WAV file:       {args.wav}")
    if args.modal_app_name:
        print(f"  Modal app name: {args.modal_app_name} (logs fetched post-test per container)")
    print()

    results = await run_test_session(
        modal_url=modal_url,
        livekit_url=args.livekit_url,
        livekit_key=args.livekit_key,
        livekit_secret=args.livekit_secret,
        worker_auth_secret=args.worker_auth_secret,
        source_image=args.source_image,
        wav_path=args.wav,
        num_sessions=num_sessions,
        duration=args.duration,
        stagger=args.stagger,
        skip_readyz=args.skip_readyz,
        monitor_readyz=args.monitor_readyz,
        audio_send_interval=args.audio_send_interval,
        idle_gap=args.idle_gap,
    )

    # Post-test: discover container IDs and fetch logs per container.
    # This captures the complete log history including restore() startup
    # messages (e.g. SAGEATTN_VERIFY output) that a --follow stream would miss.
    log_metrics = None
    raw_log_files = []
    parsed_file = None
    if args.modal_app_name:
        timestamp_fetch = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu_safe = args.gpu_type.replace(" ", "_").replace("-", "_")

        # Discover container IDs (use explicit --modal-container-id if provided,
        # otherwise auto-discover via `modal container list --json`)
        if args.modal_container_id:
            container_ids = [args.modal_container_id]
        else:
            container_ids = discover_container_ids(args.modal_app_name)

        if not container_ids:
            print(f"  [log-fetch] No containers found - skipping log fetch")
        else:
            all_raw_content = []
            for cid in container_ids:
                raw_log_file = log_dir / f"raw_{gpu_safe}_{num_sessions}S_{cid[:16]}_{timestamp_fetch}.log"
                fetch_ok = fetch_container_logs(
                    args.modal_app_name, cid, str(raw_log_file),
                )
                if fetch_ok and raw_log_file.exists():
                    raw_log_files.append(raw_log_file)
                    content = raw_log_file.read_text(encoding="utf-8")
                    all_raw_content.append(content)

            # Combine all container logs into one raw file and parse
            if all_raw_content:
                combined_raw = log_dir / f"raw_{gpu_safe}_{num_sessions}S_{timestamp_fetch}.log"
                combined_raw.write_text("\n".join(all_raw_content), encoding="utf-8")
                print(f"  [log-fetch] Combined raw log: {combined_raw}")

                log_metrics = parse_modal_logs(str(combined_raw))
                parsed_file = log_dir / f"parsed_{gpu_safe}_{num_sessions}S_{timestamp_fetch}.json"
                parsed_file.write_text(
                    json.dumps(log_metrics, indent=2), encoding="utf-8",
                )
                print(f"  [log-fetch] Parsed metrics saved to: {parsed_file}")

                if log_metrics["generated_fps_samples"]:
                    per_sess = log_metrics.get("per_session_generated_fps", {})
                    print(f"\n  [log-parse] Found {len(log_metrics['generated_fps_samples'])} VIDEO_PUBLISH_METRICS entries ({len(per_sess)} sessions)")
                else:
                    print(f"\n  [log-parse] No VIDEO_PUBLISH_METRICS entries found in logs")
                if log_metrics["inference_ms_samples"]:
                    per_inf = log_metrics.get("per_session_inference_ms", [])
                    print(f"  [log-parse] Found {len(log_metrics['inference_ms_samples'])} cycle log entries ({len(per_inf)} per-session inference samples)")

                # Check for SageAttention verification results
                try:
                    sage_lines = [l.strip() for l in all_raw_content if "SAGEATTN_VERIFY" in l]
                    if sage_lines:
                        print(f"\n  [SAGEATTN_VERIFY] Found {len(sage_lines)} result(s):")
                        for line in sage_lines:
                            print(f"    {line}")
                except Exception:
                    pass

    # Print results and get verdict
    summary = print_results(results, num_sessions, log_metrics)

    # Save JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_safe = args.gpu_type.replace(" ", "_").replace("-", "_")
    results_file = log_dir / f"results_{gpu_safe}_{num_sessions}S_{timestamp}.json"

    session_results = []
    for res in results:
        if isinstance(res, Exception):
            session_results.append({"status": "FAILED", "error": str(res)})
        else:
            session_results.append({
                "session_id": res.session_id,
                "room_name": res.room_name,
                "status": res.status,
                "error": res.error,
                "frames_received": res.frames_received,
                "fps_samples": res.fps_samples,
                "avg_fps": res.avg_fps,
                "steady_state_avg_fps": res.steady_state_avg_fps,
                "first_frame_latency_s": round(res.first_frame_time - res.start_time, 1) if res.first_frame_time > 0 else 0,
            })

    json_output = {
        "gpu_type": args.gpu_type,
        "num_sessions": num_sessions,
        "duration_s": args.duration,
        "timestamp": timestamp,
        "summary": summary,
        "log_metrics": log_metrics,
        "session_results": session_results,
        "raw_log_files": [str(f) for f in raw_log_files] if raw_log_files else None,
        "parsed_log_file": str(parsed_file) if parsed_file else None,
    }
    results_file.write_text(json.dumps(json_output, indent=2), encoding="utf-8")
    print(f"\n  Results saved to: {results_file}")
    if raw_log_files:
        for f in raw_log_files:
            print(f"  Raw logs saved to: {f}")
    if parsed_file:
        print(f"  Parsed logs saved to: {parsed_file}")

    return summary


async def _run_ramp_up(args, modal_url: str, log_dir: pathlib.Path):
    """Ramp-up mode: start at --sessions, after each run ask to continue to N+1."""

    current_sessions = args.sessions
    all_summaries = []

    print("=" * 70)
    print("  AiVatar - GPU Concurrency Ramp-Up Test")
    print("=" * 70)
    print(f"  GPU type:       {args.gpu_type}")
    print(f"  Starting sessions: {current_sessions}")
    print(f"  Duration:       {args.duration}s per session")
    print(f"  Mode:           RAMP-UP (interactive continue/stop)")
    print()

    while True:
        print(f"\n{'#' * 70}")
        print(f"  RAMP-UP: Testing {current_sessions} concurrent sessions")
        print(f"{'#' * 70}\n")

        summary = await _run_single_test(args, modal_url, log_dir, current_sessions)
        all_summaries.append({
            "sessions": current_sessions,
            **summary,
        })

        # Check verdict - stop automatically on FAIL
        if summary["verdict"] == "FAIL":
            print(f"\n  AUTO-STOP: Verdict=FAIL for {current_sessions} sessions.")
            print(f"  Max viable concurrency: {current_sessions - 1}")
            break

        # Interactive prompt: continue to next session count?
        print(f"\n  Current: {current_sessions} sessions -> {summary['verdict']} "
              f"(avg FPS: {summary['verdict_avg_fps']})")

        try:
            response = input(f"\n  Continue to {current_sessions + 1} sessions? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "n"

        if response in ("y", "yes"):
            current_sessions += 1
            print(f"\n  Proceeding to {current_sessions} sessions...")
        else:
            print(f"\n  Stopping ramp-up. Max viable concurrency: {current_sessions}")
            break

    # Print ramp-up summary
    print(f"\n\n{'=' * 70}")
    print("  RAMP-UP SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Sessions':>9} | {'Verdict':>10} | {'Avg FPS':>10} | {'Gen FPS':>10} | {'Inf P95':>10} | {'Inf Max':>10}")
    print("  " + "-" * 75)
    for s in all_summaries:
        print(f"  {s['sessions']:>9} | {s['verdict']:>10} | {s['verdict_avg_fps']:>10.1f} | "
              f"{s.get('generated_fps_avg', 0):>10.1f} | {s.get('inference_ms_p95', 0):>9.0f}ms | {s.get('inference_ms_max', 0):>9.0f}ms")

    max_viable = max((s["sessions"] for s in all_summaries if s["verdict"] == "PASS"), default=0)
    print(f"\n  Max viable concurrency: {max_viable}")
    print(f"  GPU: {args.gpu_type}")
    print(f"{'=' * 70}")

    # Save ramp-up summary JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_safe = args.gpu_type.replace(" ", "_").replace("-", "_")
    summary_file = log_dir / f"rampup_{gpu_safe}_{timestamp}.json"
    summary_file.write_text(json.dumps({
        "gpu_type": args.gpu_type,
        "timestamp": timestamp,
        "max_viable_concurrency": max_viable,
        "all_runs": all_summaries,
    }, indent=2), encoding="utf-8")
    print(f"\n  Ramp-up summary saved to: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
