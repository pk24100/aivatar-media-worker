"""
Direct Modal GPU Worker Test (no backend required).

Bypasses the backend entirely and talks straight to the Modal worker.
Mints LiveKit tokens locally using livekit-api, starts a session on the
worker via /sessions/start, streams a WAV file over WebSocket, and prints
a viewer token for meet.livekit.io.

Configuration:
    Fill in your values in  scripts/.env.test  (gitignored).
    The script auto-loads that file — no need to export env vars manually.

    You can still override any value via CLI flags:
        python scripts/modal_direct_test.py --wav test_audio.wav

Usage:
    pip install livekit-api websockets aiohttp

    # Just run it (reads from .env.test):
    python scripts/modal_direct_test.py

    # With audio streaming:
    python scripts/modal_direct_test.py --wav test_audio.wav

    # Override a single value:
    python scripts/modal_direct_test.py --modal-url https://other-url.modal.run
"""

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
import uuid
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Auto-load scripts/.env.test
# ---------------------------------------------------------------------------

def _load_env_file():
    """Load key=value pairs from .env.test into os.environ (won't overwrite)."""
    # Look next to this script first, then CWD
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
    print("[env] No .env.test found — using existing env vars / CLI flags")

_load_env_file()

# ---------------------------------------------------------------------------
# Token minting  (livekit-api >= 1.0)
# ---------------------------------------------------------------------------

def mint_livekit_token(api_key: str, api_secret: str, room_name: str,
                       identity: str, *, can_publish: bool = True,
                       can_reconnect: bool = False,
                       ttl_seconds: int = 1800) -> str:
    """Mint a LiveKit JWT locally using livekit-api."""
    from livekit.api import AccessToken, VideoGrants

    @dataclass
    class ReconnectVideoGrants(VideoGrants):
        can_reconnect: bool = False

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants((ReconnectVideoGrants if can_reconnect else VideoGrants)(
            room_join=True,
            room=room_name,
            can_publish=can_publish,
            can_subscribe=True,
            **({"can_reconnect": True} if can_reconnect else {}),
        ))
    )
    return token.to_jwt()


def mint_ws_token(payload: dict, secret: str) -> str:
    """Mint a signed JWT for WebSocket auth via subprotocol."""
    import jwt as pyjwt
    return pyjwt.encode(payload, secret, algorithm="HS256")


# ---------------------------------------------------------------------------
# Worker interaction
# ---------------------------------------------------------------------------

async def wait_for_ready(modal_url: str, timeout: int = 180):
    """Poll /readyz until the worker reports ready."""
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
                    print(f"  [{attempt}] ready={ready}  availablePipelines={avail}")
                    if ready and avail > 0:
                        return data
        except Exception as e:
            print(f"  [{attempt}] readyz poll error: {e}")
        await asyncio.sleep(10)

    raise TimeoutError(f"Worker did not become ready within {timeout}s")


async def claim_room(modal_url: str) -> dict:
    """POST /room/claim on the worker to get a pre-warmed room."""
    import aiohttp

    url = f"{modal_url}/room/claim"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={},
                                timeout=aiohttp.ClientTimeout(total=15)) as resp:
            text = await resp.text()
            if resp.status >= 400:
                print(f"  /room/claim failed ({resp.status}): {text}")
                return {"roomName": None, "workerToken": None, "clientToken": None}
            return json.loads(text)


async def start_session(modal_url: str, payload: dict) -> dict:
    """POST /sessions/start on the worker."""
    import aiohttp

    url = f"{modal_url}/sessions/start"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=60)) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"sessions/start failed ({resp.status}): {text}")
            return json.loads(text)


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
                print("\n  'q' pressed — stopping stream...")
                break
        await asyncio.sleep(0.1)


async def stream_audio_ws(
    ws_url: str,
    wav_path: str,
    ws_token: str = "",
    loop: int = -1,
    stop_event: asyncio.Event | None = None,
):
    """Stream a WAV file over WebSocket to the worker.  Loop N times (-1 = infinite)."""
    import websockets
    import wave

    print(f"\n=== Streaming audio: {wav_path} ===")
    print(f"    -> {ws_url}\n")

    async with websockets.connect(ws_url, subprotocols=[f"aivatar.{ws_token}"]) as ws:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            duration = n_frames / sr
            print(f"  WAV info: {sr} Hz, {ch} ch, {sw * 8}-bit, {duration:.1f}s")

            chunk_frames = sr // 10  # 100ms chunks
            total_sent = 0
            iteration = 0

            while True:
                iteration += 1
                wf.rewind()
                sent = 0
                while True:
                    data = wf.readframes(chunk_frames)
                    if not data:
                        break
                    if stop_event is not None and stop_event.is_set():
                        break
                    await ws.send(data)
                    sent += len(data)
                    await asyncio.sleep(0.1)

                total_sent += sent
                print(f"  Iteration {iteration}: sent {sent:,} bytes ({sent / sr / ch / sw:.1f}s)")

                if loop >= 0 and iteration >= loop:
                    break
                if stop_event is not None and stop_event.is_set():
                    break

            print(f"  Total sent {total_sent:,} bytes ({total_sent / sr / ch / sw:.1f}s, {iteration} iteration(s))")

    print("  WebSocket closed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Direct Modal GPU worker test (no backend)")
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
    parser.add_argument("--source-image",
                        default=os.getenv("SOURCE_IMAGE",
                            "https://i.postimg.cc/594x5VRK/Gemini-Generated-Image-e3xw6se3xw6se3xw.png?dl=1"),
                        help="Avatar source image URL")
    parser.add_argument("--wav",
                        help="Path to WAV file to stream (optional)")
    parser.add_argument("--loop", type=int, default=-1,
                        help="Loop the WAV N times (-1 = infinite, 0 = no loop). Default: -1")
    parser.add_argument("--room-name",
                        default=None,
                        help="LiveKit room name (auto-generated if omitted)")
    parser.add_argument("--skip-readyz", action="store_true",
                        help="Skip the readyz polling step")
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
    if missing:
        print("ERROR: Missing required arguments:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    modal_url = args.modal_url.rstrip("/")
    session_id = None
    room_name = args.room_name
    ingestion_token = f"test-{uuid.uuid4().hex[:8]}"

    print("=" * 60)
    print("  AiVatar — Direct Modal Worker Test (no backend)")
    print("=" * 60)
    print(f"  Modal URL:    {modal_url}")
    print(f"  LiveKit URL:  {args.livekit_url}")
    print(f"  Source Image:  {args.source_image[:60]}...")
    print()

    # ---- Step 1: Wait for worker readiness ----
    if not args.skip_readyz:
        print("--- Step 1: Waiting for worker readiness ---")
        await wait_for_ready(modal_url)
        print()
    else:
        print("--- Step 1: Skipped (--skip-readyz) ---\n")

    # ---- Step 2: Claim a pre-warmed room (or generate locally) ----
    print("--- Step 2: Claiming pre-warmed room ---")

    worker_token = None
    viewer_token = None

    if not room_name:
        claimed = await claim_room(modal_url)
        if claimed and claimed.get("roomName"):
            room_name = claimed["roomName"]
            worker_token = claimed.get("workerToken")
            viewer_token = claimed.get("clientToken")
            print(f"  Claimed pre-warmed room: {room_name}")
        else:
            print("  No pre-warmed rooms available, generating locally")

    if not room_name:
        room_name = args.room_name or f"direct-test-{uuid.uuid4().hex[:8]}"

    session_id = room_name

    print(f"  Room:         {room_name}")
    print(f"  Session:      {session_id}")
    print()

    # Mint tokens locally only if not claimed from the pool
    if not worker_token:
        print("  Minting LiveKit tokens locally...")
        worker_token = mint_livekit_token(
            args.livekit_key, args.livekit_secret,
            room_name, identity=f"aivatar-worker-{session_id}",
            can_publish=True, can_reconnect=True, ttl_seconds=1800,
        )
        print(f"  Worker token minted (identity=aivatar-worker-{session_id})")

    if not viewer_token:
        viewer_token = mint_livekit_token(
            args.livekit_key, args.livekit_secret,
            room_name, identity="viewer-1",
            can_publish=False, ttl_seconds=1800,
        )
        print(f"  Viewer token minted (identity=viewer-1)")
    print()

    # ---- Step 3: Start session on the worker ----
    print("--- Step 3: Starting session on worker ---")

    payload = {
        "roomName": room_name,
        "livekitToken": worker_token,
        "customLivekitUrl": args.livekit_url,
        "sourceImage": args.source_image,
        "streaming": True,
        "ingestionMethod": "websocket",
        "sessionId": session_id,
        "ingestionToken": ingestion_token,
    }
    result = await start_session(modal_url, payload)
    print(f"  Response: {json.dumps(result, indent=2)}")
    print()

    # ---- Step 4: Print viewer join instructions ----
    print("--- Step 4: Join as viewer ---")
    print()
    print("  Open https://meet.livekit.io and paste:")
    print(f"    LiveKit URL:   {args.livekit_url}")
    print(f"    Token:         {viewer_token}")
    print()
    print("  Or use the LiveKit CLI:")
    print(f"    lk room join --url {args.livekit_url} --token {viewer_token}")
    print()

    # ---- Step 5: Build WebSocket URL and print end session instructions ----
    # Mint JWT for WS subprotocol auth
    ws_auth_secret = os.getenv("WORKER_AUTH_SECRET", "test-secret")
    ws_token = mint_ws_token(payload, ws_auth_secret)

    ws_base = modal_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/ws/{session_id}"

    end_url = f"{modal_url}/sessions/{session_id}/end"

    print("--- Step 5: End session ---")
    print(f"  When done, end the session with one of:")
    print(f"    Linux/macOS:   curl -X POST {end_url}")
    print(f"    PowerShell:    Invoke-RestMethod -Uri '{end_url}' -Method Post")
    print(f"    Python:        python -c \"import urllib.request; urllib.request.urlopen('{end_url}')\"")
    print()
    print("=" * 60)
    print("  Test streaming!")
    print("=" * 60)
    print()

    # ---- Step 6: Stream audio (if WAV provided) ----
    streamed = False
    if args.wav:
        if args.loop == 0:
            print("--- Step 6: Skipped (--loop 0) ---")
        else:
            print("--- Step 6: Streaming audio ---")
            if args.loop < 0:
                print("  Looping indefinitely (press 'q' then Enter to stop and end session)")
            else:
                print(f"  Looping {args.loop} time(s)")
            stop_event = asyncio.Event()
            stream_task = asyncio.create_task(
                stream_audio_ws(ws_url, args.wav, ws_token=ws_token, loop=args.loop, stop_event=stop_event)
            )
            keyboard_task = asyncio.create_task(watch_keyboard(stop_event))
            done, pending = await asyncio.wait(
                [stream_task, keyboard_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            streamed = True
    else:
        print("--- Step 6: Stream audio manually ---")
        print()
        print("  No --wav file provided. Stream manually with:")
        print(f"    python step5_stream_audio.py --ws-url \"{ws_url}\"")
        print()
        print("  Or use the WebSocket URL with subprotocol auth:")
        print(f"    {ws_url}")
        print(f"    Subprotocol: aivatar.<jwt>")
        print()

    # ---- Step 7: Auto-end session after streaming ----
    if streamed:
        print(f"  Ending session via {end_url} ...")
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(end_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    text = await resp.text()
                    print(f"  Response ({resp.status}): {text}")
        except Exception as e:
            print(f"  Failed to end session: {e}")
            print(f"  Please end it manually: {end_url}")


if __name__ == "__main__":
    asyncio.run(main())
