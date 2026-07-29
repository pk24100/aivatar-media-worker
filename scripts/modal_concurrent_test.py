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

Configuration:
    Fill in scripts/.env.test (same as modal_direct_test.py).

Usage:
    pip install livekit-api websockets aiohttp

    # Test 3 concurrent sessions (default):
    python scripts/modal_concurrent_test.py --wav test_audio.wav

    # Test with 2 pipelines:
    python scripts/modal_concurrent_test.py --wav test_audio.wav --sessions 2

    # Test with 5 sessions (stress test beyond pool size):
    python scripts/modal_concurrent_test.py --wav test_audio.wav --sessions 5

    # Override Modal URL:
    python scripts/modal_concurrent_test.py --modal-url https://other-url.modal.run --wav test_audio.wav
"""

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
import uuid
from dataclasses import dataclass, field

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

async def wait_for_ready(modal_url: str, timeout: int = 180):
    import aiohttp
    url = f"{modal_url}/readyz"
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    ready = data.get("ready", False)
                    avail = data.get("availablePipelines", 0)
                    pool_size = data.get("poolSize", 0)
                    active = data.get("activeSessions", 0)
                    print(f"  [{attempt}] ready={ready}  availablePipelines={avail}  poolSize={pool_size}  activeSessions={active}")
                    if ready and avail > 0:
                        return data
        except Exception as e:
            print(f"  [{attempt}] readyz poll error: {e}")
        await asyncio.sleep(10)
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
                raise RuntimeError(f"sessions/start failed ({resp.status}): {text}")
            return json.loads(text)


async def end_session(modal_url: str, session_id: str):
    import aiohttp
    url = f"{modal_url}/sessions/{session_id}/end"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                text = await resp.text()
                print(f"  [end] {session_id}: {resp.status} {text[:100]}")
    except Exception as e:
        print(f"  [end] {session_id}: failed - {e}")

# ---------------------------------------------------------------------------
# Keyboard watcher (Windows msvcrt, same as modal_direct_test.py)
# ---------------------------------------------------------------------------

async def watch_keyboard(stop_event: asyncio.Event):
    """Windows-only non-blocking key watcher. Sets stop_event when 'q' is pressed."""
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
    # FPS metrics
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
        except Exception as e:
            print(f"  [{session_id}] Video stream error: {e}")

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
        video_wait_deadline = time.monotonic() + 75
        while not video_track_received and time.monotonic() < video_wait_deadline:
            await asyncio.sleep(0.5)

        if not video_track_received:
            print(f"  [{session_id}] WARNING: No video track received within 75s")
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

    except Exception as e:
        print(f"  [{session_id}] Viewer error: {e}")
        result.status = "FAILED"
        result.error = str(e)
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
        async with websockets.connect(ws_url, subprotocols=[f"aivatar.{ws_token}"]) as ws:
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
    except Exception as e:
        print(f"  [{session_id}] Audio stream error: {e}")

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
        # Step 1: Claim a room
        claimed = await claim_room(modal_url)
        room_name = claimed.get("roomName") if claimed else None
        worker_token = claimed.get("workerToken") if claimed else None
        viewer_token = claimed.get("clientToken") if claimed else None

        if not room_name:
            room_name = f"conc-test-{uuid.uuid4().hex[:8]}"
            worker_token = mint_livekit_token(
                livekit_key, livekit_secret,
                room_name, identity=f"aivatar-worker-{result.session_id}",
                can_publish=True, ttl_seconds=1800,
            )

        if not viewer_token:
            viewer_token = mint_livekit_token(
                livekit_key, livekit_secret,
                room_name, identity=f"viewer-{result.session_id}",
                can_publish=False, ttl_seconds=1800,
            )

        result.room_name = room_name
        print(f"  [{result.session_id}] Room: {room_name}")
        print(f"  [{result.session_id}] LiveKit URL: {livekit_url}")
        print(f"  [{result.session_id}] Viewer token: {viewer_token}")
        print(f"  [{result.session_id}] >> Paste token at https://meet.livekit.io to watch <<")

        # Step 2: Start session on worker
        ingestion_token = f"test-{uuid.uuid4().hex[:8]}"
        payload = {
            "roomName": room_name,
            "livekitToken": worker_token,
            "customLivekitUrl": livekit_url,
            "sourceImage": source_image,
            "streaming": True,
            "ingestionMethod": "websocket",
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
                livekit_url, viewer_token, result.session_id,
                result, stop_event, duration_seconds=duration_seconds,
            )
        )

        # Step 4b: Wait for media readiness (first video frame) before streaming audio.
        # The E2E sample site blocks speech until mediaReady=true. Without this gate,
        # audio buffers during avatar preparation and creates an artificial burst.
        print(f"  [{result.session_id}] Waiting for first video frame before audio streaming...")
        media_ready_deadline = time.monotonic() + 75
        while result.first_frame_time == 0.0 and time.monotonic() < media_ready_deadline:
            if stop_event.is_set():
                break
            await asyncio.sleep(0.5)

        if result.first_frame_time == 0.0:
            print(f"  [{result.session_id}] WARNING: No first frame within 75s, starting audio anyway")
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

    except Exception as e:
        print(f"  [{result.session_id}] ERROR: {e}")
        result.status = "FAILED"
        result.error = str(e)

    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    print("=" * 70)
    print("  AiVatar - Concurrent Modal Worker Test with FPS Measurement")
    print("=" * 70)
    print(f"  Modal URL:      {modal_url}")
    print(f"  LiveKit URL:    {args.livekit_url}")
    print(f"  Sessions:       {args.sessions}")
    print(f"  Duration:       {args.duration}s per session")
    print(f"  Stagger:        {args.stagger}s between launches")
    print(f"  Audio interval: {args.audio_send_interval}s ({0.1 / args.audio_send_interval:.1f}x real-time)")
    print(f"  Idle gap:       {args.idle_gap}s between utterances")
    print(f"  Source Image:   {args.source_image[:50]}...")
    print(f"  WAV file:       {args.wav}")
    print()

    # ---- Step 1: Wait for worker readiness ----
    if not args.skip_readyz:
        print("--- Step 1: Waiting for worker readiness ---")
        ready_data = await wait_for_ready(modal_url)
        pool_size = ready_data.get("poolSize", "?")
        avail = ready_data.get("availablePipelines", "?")
        print(f"  Worker ready: poolSize={pool_size}, availablePipelines={avail}")
        if isinstance(avail, int) and isinstance(pool_size, int):
            if avail < args.sessions:
                print(f"  WARNING: Requesting {args.sessions} sessions but only {avail} pipelines available.")
                print(f"           Sessions beyond pool size will queue/block.")
        print()
    else:
        print("--- Step 1: Skipped (--skip-readyz) ---\n")

    # ---- Step 2: Launch concurrent sessions ----
    print(f"--- Step 2: Launching {args.sessions} concurrent sessions ---\n")

    stop_event = asyncio.Event()
    results = []
    tasks = []

    for i in range(args.sessions):
        print(f"  Launching session {i+1}/{args.sessions}...")
        task = asyncio.create_task(
            run_single_session(
                modal_url=modal_url,
                livekit_url=args.livekit_url,
                livekit_key=args.livekit_key,
                livekit_secret=args.livekit_secret,
                worker_auth_secret=args.worker_auth_secret,
                source_image=args.source_image,
                wav_path=args.wav,
                session_idx=i + 1,
                duration_seconds=args.duration,
                stop_event=stop_event,
                audio_send_interval=args.audio_send_interval,
                idle_gap_seconds=args.idle_gap,
            )
        )
        tasks.append(task)
        if i < args.sessions - 1:
            await asyncio.sleep(args.stagger)

    print(f"\n  All {args.sessions} sessions launched. Measuring FPS for {args.duration}s...")
    print(f"  Press 'q' + Enter to stop all sessions early.\n")

    # ---- Step 3: Monitor readyz during test ----
    async def monitor_readyz():
        import aiohttp
        monitor_url = f"{modal_url}/readyz"
        check_interval = 10
        elapsed = 0
        while elapsed < args.duration + 30:
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
    if args.monitor_readyz:
        monitor_task = asyncio.create_task(monitor_readyz())
    keyboard_task = asyncio.create_task(watch_keyboard(stop_event))

    # Wait for all sessions to complete (or 'q' to be pressed)
    results = await asyncio.gather(*tasks, return_exceptions=True)

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

    # ---- Step 4: Print results ----
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
    print(f"  SUMMARY: {successful}/{args.sessions} sessions completed, {failed} failed")
    print(f"  Target FPS: 25 (from infer_params.yaml tgt_fps)")
    if all_fps_samples:
        overall_avg = round(sum(all_fps_samples) / len(all_fps_samples), 1)
        overall_min = min(all_fps_samples)
        overall_max = max(all_fps_samples)
        print(f"  Overall Avg FPS (all windows): {overall_avg}")
        print(f"  Overall Min FPS (all windows): {overall_min}")
        print(f"  Overall Max FPS (all windows): {overall_max}")

        verdict_avg = overall_avg
        if steady_state_fps_samples:
            steady_avg = round(sum(steady_state_fps_samples) / len(steady_state_fps_samples), 1)
            steady_min = min(steady_state_fps_samples)
            steady_max = max(steady_state_fps_samples)
            verdict_avg = steady_avg
            print(f"  Steady-state Avg FPS: {steady_avg}")
            print(f"  Steady-state Min FPS: {steady_min}")
            print(f"  Steady-state Max FPS: {steady_max}")
            print(f"  Verdict basis: steady-state FPS excluding the first window per session")

        # Verdict
        if verdict_avg >= 23:
            print(f"\n  VERDICT: PASS - {args.sessions} concurrent sessions sustain ~25fps")
        elif verdict_avg >= 18:
            print(f"\n  VERDICT: MARGINAL - FPS degraded but usable. Consider reducing concurrency.")
        else:
            print(f"\n  VERDICT: FAIL - FPS too low for {args.sessions} concurrent sessions.")
            print(f"           Reduce --sessions or check GPU utilization.")
    print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())
