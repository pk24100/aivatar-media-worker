import asyncio
import contextlib
import json
import logging
import os
import time
from collections import defaultdict

import aiohttp
import jwt as pyjwt
from aiohttp import web
from huggingface_hub import snapshot_download

from streaming.stream_processor import run_streaming_session
from streaming.websocket_server import ws_server
from utils.model_pool import FlashHeadModelPool

_BATCHED_INFERENCE = os.environ.get("AIVATAR_BATCHED_INFERENCE", "1") == "1"
if _BATCHED_INFERENCE:
    from streaming.batched_stream_processor import run_batched_streaming_session

# Shared BatchedStreamingEngine instance (set by modal_app.py serve())
batched_engine = None


PREWARM_ROOM_TIMEOUT = 60.0

# --- WebSocket Security Constants (Fixes 4, 6, 8, 15) ---
WORKER_AUTH_SECRET = os.environ.get("WORKER_AUTH_SECRET", "")
ALLOWED_WS_ORIGINS = set(o.strip() for o in os.environ.get("ALLOWED_WS_ORIGINS", "").split(",") if o.strip())
MAX_AUDIO_CHUNK_BYTES = 1_048_576   # 1 MB — 5s @ 48kHz 16-bit stereo PCM
WS_INACTIVITY_TIMEOUT = int(os.environ.get("WS_INACTIVITY_TIMEOUT", "45"))
NO_AUDIO_SESSION_TIMEOUT = int(os.environ.get("NO_AUDIO_SESSION_TIMEOUT", "120"))
SESSION_IDLE_CHECK_INTERVAL = int(os.environ.get("SESSION_IDLE_CHECK_INTERVAL", "5"))
BACKEND_INTERNAL_URL = os.environ.get("BACKEND_INTERNAL_URL", "")
WS_AUDIO_RATE_PER_SEC = 60          # max audio messages/sec (sustained)
WS_AUDIO_BURST = 200                # token bucket burst capacity
WS_MAX_BYTES_PER_SEC = 192_000      # 48kHz 16-bit mono real-time ceiling
WS_MAX_CONNECTIONS_PER_IP = 10      # max concurrent WS per IP per 60s window

# --- Fix 13: One-time JWT tracking (jti -> expiry timestamp) ---
_used_jti: dict = {}

# --- Fix 15: IP-based connection throttling ---
_ip_connections: dict = defaultdict(list)  # ip -> [timestamp, ...]


class AudioRateLimiter:
    """Token-bucket rate limiter for audio messages (Fix 8)."""

    def __init__(self, rate_per_sec, burst, max_bytes_per_sec):
        self.rate = float(rate_per_sec)
        self.burst = float(burst)
        self.tokens = float(burst)
        self.last_refill = time.monotonic()
        self.byte_window_start = time.monotonic()
        self.bytes_in_window = 0
        self.max_bytes_per_sec = max_bytes_per_sec

    def allow(self, msg_size: int) -> tuple:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens < 1.0:
            return False, "message rate exceeded"
        self.tokens -= 1.0
        window_elapsed = now - self.byte_window_start
        if window_elapsed >= 1.0:
            self.byte_window_start = now
            self.bytes_in_window = 0
        self.bytes_in_window += msg_size
        if self.bytes_in_window > self.max_bytes_per_sec:
            return False, "byte rate exceeded"
        return True, ""


def _check_ip_limit(ip: str, max_conn: int, window_sec: int = 60) -> bool:
    """Check if an IP has exceeded the connection limit (Fix 15)."""
    now = time.time()
    _ip_connections[ip] = [t for t in _ip_connections[ip] if now - t < window_sec]
    if len(_ip_connections[ip]) >= max_conn:
        return False
    _ip_connections[ip].append(now)
    return True


def _check_jti(claims: dict) -> bool:
    """Track one-time JWT usage via jti claim (Fix 13). Returns True if valid (not used before)."""
    jti = claims.get("jti")
    if not jti:
        return True  # No jti claim — skip one-time enforcement
    exp = claims.get("exp", 0)
    now = time.time()
    for k, v in list(_used_jti.items()):
        if v < now:
            del _used_jti[k]
    if jti in _used_jti:
        return False
    _used_jti[jti] = exp
    return True


async def _end_session_via_backend(session_id: str, reason: str = "no_audio_timeout"):
    """Call backend POST /internal/sessions/:id/end to finalize a stale session.
    Uses WORKER_AUTH_SECRET to mint a JWT for authentication."""
    if not BACKEND_INTERNAL_URL:
        logger.warning("NO_AUDIO_TIMEOUT session=%s reason=%s BACKEND_INTERNAL_URL not set, cannot end session", session_id, reason)
        return

    if not WORKER_AUTH_SECRET:
        logger.warning("NO_AUDIO_TIMEOUT session=%s reason=%s WORKER_AUTH_SECRET not set, cannot end session", session_id, reason)
        return

    token = pyjwt.encode(
        {"sessionId": session_id, "reason": reason, "iat": int(time.time())},
        WORKER_AUTH_SECRET,
        algorithm="HS256",
    )

    url = f"{BACKEND_INTERNAL_URL.rstrip('/')}/internal/sessions/{session_id}/end"
    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={"reason": reason},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.text()
                if resp.status == 200:
                    logger.info("NO_AUDIO_TIMEOUT session=%s backend end success status=%d", session_id, resp.status)
                else:
                    logger.warning("NO_AUDIO_TIMEOUT session=%s backend end failed status=%d body=%s", session_id, resp.status, body[:200])
    except Exception as exc:
        logger.warning("NO_AUDIO_TIMEOUT session=%s backend end error: %s", session_id, exc)


async def _watch_session_idle(session_id: str, session_state: dict) -> None:
    """End only this session after it has been inactive outside a tail drain."""
    while True:
        await asyncio.sleep(SESSION_IDLE_CHECK_INTERVAL)
        if session_state.get("ending"):
            return
        if session_state.get("is_draining"):
            continue

        elapsed = time.monotonic() - session_state.get("last_activity_time", time.monotonic())
        if elapsed < NO_AUDIO_SESSION_TIMEOUT:
            continue

        session_state["ending"] = True
        logger.info(
            "SESSION_IDLE_TIMEOUT session=%s elapsed=%.0fs threshold=%ds",
            session_id,
            elapsed,
            NO_AUDIO_SESSION_TIMEOUT,
        )
        await session_state["queue"].put({"type": "end_session", "reason": "no_audio_timeout"})
        await _end_session_via_backend(session_id, "no_audio_timeout")

        websocket = session_state.get("websocket")
        if websocket is not None:
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "session_ending", "reason": "no_audio_timeout"})
                await websocket.close(code=1000)
        return


class PrewarmRoomPool:
    def __init__(self, size=1):
        self.size = size
        self._queue = asyncio.Queue(maxsize=size)
        self._entries = {}
        self._claimed = {}

    async def add(self, room_name, room, participant_identity, worker_token):
        entry = {
            "roomName": room_name,
            "room": room,
            "participantIdentity": participant_identity,
            "workerToken": worker_token,
            "createdAt": time.monotonic(),
        }
        self._entries[room_name] = entry
        await self._queue.put(entry)

    def claim(self):
        try:
            entry = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        # Move to claimed map so release() can still find the room handle.
        self._entries.pop(entry["roomName"], None)
        self._claimed[entry["roomName"]] = entry
        return entry

    def take_claimed(self, room_name):
        """Remove and return a claimed room without disconnecting it."""
        return self._claimed.pop(room_name, None)

    async def release(self, room_name):
        # First try the claimed map (normal path after claim).
        entry = self._claimed.pop(room_name, None)
        if entry is None:
            # Fallback: unclaimed expired entry.
            entry = self._entries.pop(room_name, None)
        if entry is None:
            return
        # Fire-and-forget: the Rust FFI disconnect() can block the event loop
        # for 30+ seconds, making asyncio.wait_for useless. The pre-warm
        # participant has a different identity and no publish/subscribe grants,
        # so it won't interfere with the real session. LiveKit server will
        # clean it up on idle timeout.
        async def _bg_disconnect():
            try:
                await entry["room"].disconnect()
            except Exception as e:
                logger.warning("[prewarm] Fire-and-forget disconnect failed for %s: %s", room_name, e)
        asyncio.create_task(_bg_disconnect())

    async def cleanup_expired(self):
        now = time.monotonic()
        # Expired entries are only those that have never been claimed.
        expired = [
            name for name, entry in self._entries.items()
            if now - entry["createdAt"] > PREWARM_ROOM_TIMEOUT
        ]
        for name in expired:
            await self.release(name)
        # Also clean up stale claimed entries that were never released.
        stale = [
            name for name, entry in self._claimed.items()
            if now - entry["createdAt"] > PREWARM_ROOM_TIMEOUT
        ]
        for name in stale:
            await self.release(name)

    def available_count(self):
        return self._queue.qsize()

DEFAULT_FLASHHEAD_HF_CACHE_DIR = (
    "/runpod-volume/huggingface-cache/hub"
    if os.path.isdir("/runpod-volume")
    else os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
)
DEFAULT_FLASHHEAD_HF_REPO_ID = "pkam24100/aivatar-flashhead-model"
LEGACY_FLASHHEAD_CKPT_DIR = "/app/models/SoulX-FlashHead-1_3B"
WAV2VEC_DIR = "/app/models/wav2vec2-base-960h"
FLASHHEAD_HF_REPO_ID = DEFAULT_FLASHHEAD_HF_REPO_ID
FLASHHEAD_HF_CACHE_DIR = DEFAULT_FLASHHEAD_HF_CACHE_DIR
FLASHHEAD_HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
WORKER_POOL_SIZE = int(os.getenv("AIVATAR_WORKER_CONCURRENCY", "3"))
if WORKER_POOL_SIZE < 1:
    raise ValueError("AIVATAR_WORKER_CONCURRENCY must be at least 1")
WORKER_HTTP_HOST = os.getenv("AIVATAR_HTTP_HOST", "0.0.0.0")
WORKER_HTTP_PORT = int(os.getenv("AIVATAR_HTTP_PORT", "8000"))

logger = logging.getLogger(__name__)


def _is_valid_flashhead_ckpt_dir(path):
    return (
        bool(path)
        and os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "Model_Lite"))
        and os.path.isdir(os.path.join(path, "VAE_LTX"))
    )


def _resolve_flashhead_ckpt_dir():
    if _is_valid_flashhead_ckpt_dir(LEGACY_FLASHHEAD_CKPT_DIR):
        logger.info("Using baked FlashHead checkpoint directory: %s", LEGACY_FLASHHEAD_CKPT_DIR)
        return LEGACY_FLASHHEAD_CKPT_DIR

    if FLASHHEAD_HF_REPO_ID:
        try:
            resolved_dir = snapshot_download(
                repo_id=FLASHHEAD_HF_REPO_ID,
                cache_dir=FLASHHEAD_HF_CACHE_DIR,
                token=FLASHHEAD_HF_TOKEN,
                local_files_only=(
                    os.path.isdir("/runpod-volume")
                    and os.path.isdir(FLASHHEAD_HF_CACHE_DIR)
                ),
            )
        except Exception as error:
            logger.warning(
                "Failed to resolve FlashHead checkpoint directory from Hugging Face repo %s: %s",
                FLASHHEAD_HF_REPO_ID,
                error,
            )
        else:
            if _is_valid_flashhead_ckpt_dir(resolved_dir):
                logger.info(
                    "Using Hugging Face FlashHead checkpoint snapshot from %s: %s",
                    FLASHHEAD_HF_REPO_ID,
                    resolved_dir,
                )
                return resolved_dir
            logger.warning(
                "Resolved Hugging Face snapshot is missing Model_Lite or VAE_LTX: %s",
                resolved_dir,
            )

    logger.info("Falling back to legacy FlashHead checkpoint directory: %s", LEGACY_FLASHHEAD_CKPT_DIR)
    return LEGACY_FLASHHEAD_CKPT_DIR


FLASHHEAD_CKPT_DIR = _resolve_flashhead_ckpt_dir()

# Modal's CPU snapshot only needs one immediately usable pipeline. The remaining
# capacity is loaded after restore while that first session can already stream.
INITIAL_PIPELINE_COUNT = (
    1
    if os.getenv("FLASHHEAD_LOAD_DEVICE", "").lower() == "cpu"
    else WORKER_POOL_SIZE
)
model_pool = FlashHeadModelPool(
    max_size=WORKER_POOL_SIZE,
    initial_size=INITIAL_PIPELINE_COUNT,
    ckpt_dir=FLASHHEAD_CKPT_DIR,
    wav2vec_dir=WAV2VEC_DIR,
)
prewarm_pool = PrewarmRoomPool(size=WORKER_POOL_SIZE)
_ws_started = False
_ws_lock = asyncio.Lock()
_active_sessions = {}
_model_ready_event = None
_model_init_error = None


def configure_model_readiness(ready: bool):
    """Initialize model readiness on the aiohttp loop that owns sessions."""
    global _model_ready_event, _model_init_error
    _model_ready_event = asyncio.Event()
    _model_init_error = None
    if ready:
        _model_ready_event.set()


def mark_model_ready(error: Exception = None):
    global _model_init_error
    _model_init_error = error
    if _model_ready_event is not None:
        _model_ready_event.set()


async def wait_for_model_ready():
    if _model_ready_event is None:
        return
    await _model_ready_event.wait()
    if _model_init_error is not None:
        raise RuntimeError("FlashHead model initialization failed") from _model_init_error

# Verify FlashHead and Wav2Vec model directories exist.
def _verify_models():
    if not _is_valid_flashhead_ckpt_dir(FLASHHEAD_CKPT_DIR):
        print(f"Warning: FlashHead checkpoint dir not found at {FLASHHEAD_CKPT_DIR}")
    if not os.path.exists(WAV2VEC_DIR):
        print(f"Warning: Wav2Vec dir not found at {WAV2VEC_DIR}")
        
    return FLASHHEAD_CKPT_DIR

DATA_ROOT = _verify_models()


# Ensure the WebSocket ingestion server is running.
async def _ensure_ws_server_started():
    global _ws_started
    if _ws_started and ws_server.is_running:
        return

    async with _ws_lock:
        if _ws_started and ws_server.is_running:
            return

        await ws_server.start()
        _ws_started = True


# Extract or validate the LiveKit server URL from the event.
def _get_livekit_url(event):
    livekit_url = event.get("customLivekitUrl") or os.getenv("LIVEKIT_URL")
    if not livekit_url:
        raise ValueError("LIVEKIT_URL is required for streaming mode")
    return livekit_url


def _mint_livekit_token(room_name, can_publish, can_subscribe, identity, ttl_seconds=1800):
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required to mint tokens")
    from livekit.api import AccessToken, VideoGrants
    return (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
            )
        )
        .to_jwt()
    )


# Acquire a model and run a streaming session for the given event.
async def _execute_streaming_event(event):
    room_name = event.get("roomName")
    livekit_token = event.get("livekitToken")
    source_image = event.get("sourceImage")
    idle_video_url = event.get("idleVideoUrl")
    idle_video_key = event.get("idleVideoKey")
    ingestion_method = event.get("ingestionMethod", "websocket")
    session_id = event.get("sessionId", room_name)
    ingestion_token = event.get("ingestionToken", "")

    if not room_name:
        raise ValueError("roomName is required for streaming mode")
    if not source_image:
        raise ValueError("sourceImage is required for streaming mode")

    if not livekit_token:
        is_byolr = bool(event.get("customLivekitUrl"))
        if is_byolr:
            raise ValueError("livekitToken is required for BYOLR streaming mode")
        livekit_token = _mint_livekit_token(
            room_name, can_publish=True, can_subscribe=True,
            identity=f"aivatar-worker-{session_id}",
        )
        logger.info("[handler] Minted worker token for room %s (no client token provided)", room_name)

    import time as _htime

    preconnected_room = None
    prewarm_entry = prewarm_pool.take_claimed(room_name)
    if prewarm_entry is not None:
        candidate_room = prewarm_entry.get("room")
        try:
            from livekit import rtc
            current_loop = asyncio.get_running_loop()
            room_loop = getattr(candidate_room, "_loop", None)
            if room_loop is not None and room_loop is not current_loop:
                logger.warning(
                    "[handler] Cannot reuse pre-warm room %s: event loop mismatch room_loop=%s current_loop=%s",
                    room_name,
                    id(room_loop),
                    id(current_loop),
                )
            elif candidate_room and candidate_room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
                preconnected_room = candidate_room
                logger.info(
                    "[handler] Reusing pre-warmed LiveKit room %s connection_state=%s identity=%s",
                    room_name,
                    candidate_room.connection_state,
                    candidate_room.local_participant.identity if candidate_room.local_participant else "N/A",
                )
        except Exception as exc:
            logger.warning("[handler] Cannot reuse pre-warm room %s: %s", room_name, exc)

    if preconnected_room is None:
        # Fallback path for BYOLR, pool miss, expired room, loop mismatch, or
        # Modal routing /room/claim and /sessions/start to different containers.
        _release_t0 = _htime.monotonic()
        try:
            await prewarm_pool.release(room_name)
        except Exception as exc:
            logger.warning("[handler] Failed to release pre-warm room %s: %s", room_name, exc)
        _release_ms = round((_htime.monotonic() - _release_t0) * 1000, 1)
        logger.info("[handler] prewarm_pool.release() took %.1fms for room %s", _release_ms, room_name)

    _session_t0 = _htime.monotonic()
    try:
        if _BATCHED_INFERENCE and batched_engine is not None:
            await run_batched_streaming_session(
                room_name=room_name,
                livekit_token=livekit_token,
                livekit_url=_get_livekit_url(event),
                engine=batched_engine,
                source_image=source_image,
                ingestion_method=ingestion_method,
                session_id=session_id,
                ingestion_token=ingestion_token,
                idle_video_url=idle_video_url,
                idle_video_key=idle_video_key,
                preconnected_room=preconnected_room,
            )
            _session_ms = round((_htime.monotonic() - _session_t0) * 1000, 1)
            logger.info("[handler] run_batched_streaming_session() completed in %.1fms for %s", _session_ms, session_id)
        else:
            await run_streaming_session(
                room_name=room_name,
                livekit_token=livekit_token,
                livekit_url=_get_livekit_url(event),
                pipeline=None,
                model_pool=model_pool,
                model_ready_waiter=wait_for_model_ready,
                source_image=source_image,
                ingestion_method=ingestion_method,
                session_id=session_id,
                ingestion_token=ingestion_token,
                idle_video_url=idle_video_url,
                idle_video_key=idle_video_key,
                preconnected_room=preconnected_room,
            )
            _session_ms = round((_htime.monotonic() - _session_t0) * 1000, 1)
            logger.info("[handler] run_streaming_session() completed in %.1fms for %s", _session_ms, session_id)
        return {"status": "ok", "mode": "streaming", "sessionId": session_id}
    except Exception as exc:
        _session_ms = round((_htime.monotonic() - _session_t0) * 1000, 1)
        logger.error("[handler] Streaming session failed for %s after %.1fms: %s", session_id, _session_ms, exc, exc_info=True)
        return {"status": "error", "mode": "streaming", "sessionId": session_id, "error": str(exc)}


# Track an active session task and clean up when it finishes.
def _track_session_task(session_id, task):
    _active_sessions[session_id] = task

    # Remove session from tracking when the task completes.
    def _cleanup(_task):
        _active_sessions.pop(session_id, None)
        session_state = ws_server.active_sessions.get(session_id)
        idle_watcher = session_state.get("idle_watcher") if session_state else None
        if idle_watcher is not None and idle_watcher is not asyncio.current_task():
            idle_watcher.cancel()

    task.add_done_callback(_cleanup)


# Start a new streaming session on the pod and track it.
async def _start_pod_session(event):
    await _ensure_ws_server_started()

    session_id = event.get("sessionId") or event.get("roomName")
    if not session_id:
        raise ValueError("sessionId or roomName is required")
    if session_id in _active_sessions:
        logger.info("Session %s already active; treating start request as idempotent", session_id)
        return {
            "jobId": session_id,
            "status": "ALREADY_STARTED",
            "sessionId": session_id,
        }
    if model_pool.get_available_count() <= 0:
        raise ValueError("No pipeline capacity available")

    if event.get("ingestionMethod", "websocket") == "websocket":
        ingestion_token = event.get("ingestionToken", "")
        if not ingestion_token:
            raise ValueError("ingestionToken is required for websocket ingestion")
        ws_server.register_session(session_id, ingestion_token)

    task = asyncio.create_task(_execute_streaming_event(event), name=f"session-{session_id}")
    _track_session_task(session_id, task)
    return {
        "jobId": session_id,
        "status": "STARTED",
        "sessionId": session_id,
    }


# Cancel an active session by its ID.
async def _cancel_pod_session(session_id):
    task = _active_sessions.get(session_id)
    if task is None:
        return {"status": "NOT_FOUND", "sessionId": session_id}

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    return {"status": "ENDED", "sessionId": session_id}


# Check if the incoming event requests streaming mode.
def _is_streaming(event):
    if event.get("streaming") is True:
        return True
    if str(event.get("mode", "")).lower() == "streaming":
        return True
    env_flag = os.getenv("AIVATAR_STREAMING", "").strip().lower()
    if env_flag:
        return env_flag in {"1", "true", "yes"}
    return True


# HTTP health endpoint: report worker health and capacity.
async def pod_health(_request):
    return web.json_response({
        "ok": True,
        "status": "healthy",
        "runtimeMode": "load_balancer",
        "activeSessions": len(_active_sessions),
        "availablePipelines": model_pool.get_available_count(),
    })


# HTTP readiness endpoint: report whether the worker is ready.
async def pod_ready(_request):
    server_ready = ws_server.is_running
    models_ready = _model_ready_event is None or (_model_ready_event.is_set() and _model_init_error is None)
    prewarm_rooms_available = prewarm_pool.available_count()
    # A listening HTTP server is not session-ready until it can hand the
    # backend a connected LiveKit room for the first viewer.
    prewarm_ready = prewarm_rooms_available > 0
    ready = server_ready and models_ready and prewarm_ready
    return web.json_response({
        "ready": ready,
        "status": "READY" if ready else "STARTING",
        "serverReady": server_ready,
        "modelsReady": models_ready,
        "prewarmReady": prewarm_ready,
        "runtimeMode": "load_balancer",
        "poolSize": WORKER_POOL_SIZE,
        "availablePipelines": model_pool.get_available_count(),
        "activeSessions": len(_active_sessions),
        "prewarmRoomsAvailable": prewarm_rooms_available,
    }, status=200 if ready else 503)


async def claim_room(_request):
    entry = prewarm_pool.claim()
    if entry is None:
        return web.json_response({
            "roomName": None,
            "workerToken": None,
            "clientToken": None,
        }, status=200)

    try:
        worker_token = _mint_livekit_token(
            entry["roomName"],
            can_publish=True,
            can_subscribe=True,
            identity=f"aivatar-worker-{entry['roomName']}",
        )
        client_token = _mint_livekit_token(
            entry["roomName"],
            can_publish=False,
            can_subscribe=True,
            identity=f"viewer-{entry['roomName']}",
        )
    except Exception as exc:
        logger.error("[claim_room] Failed to mint tokens: %s", exc)
        await prewarm_pool.release(entry["roomName"])
        return web.json_response({
            "roomName": None,
            "workerToken": None,
            "clientToken": None,
        }, status=200)

    logger.info("[claim_room] Claimed room %s for new session", entry["roomName"])
    return web.json_response({
        "roomName": entry["roomName"],
        "workerToken": worker_token,
        "clientToken": client_token,
    })


# Lightweight HTTP ping endpoint.
async def app_ping(_request):
    if ws_server.is_running:
        return web.json_response({
            "status": "healthy",
            "runtimeMode": "load_balancer",
        })

    return web.Response(status=204)


# HTTP handler to start a new streaming session.
async def pod_session_start(request):
    event = await request.json()
    logger.info(
        "[pod_session_start] Received session=%s room=%s ingestion=%s hasIdleAsset=%s",
        event.get("sessionId"),
        event.get("roomName"),
        event.get("ingestionMethod", "websocket"),
        bool(event.get("idleVideoKey") or event.get("idleVideoUrl")),
    )
    if not _is_streaming(event):
        logger.warning("[pod_session_start] Not streaming mode for session=%s", event.get("sessionId"))
        raise web.HTTPBadRequest(text='Only streaming mode is supported')

    try:
        payload = await _start_pod_session(event)
        return web.json_response(payload, status=202)
    except ValueError as error:
        logger.error("[pod_session_start] ValueError: %s", error)
        raise web.HTTPBadRequest(text=str(error)) from error


# HTTP handler to end an active streaming session.
async def pod_session_end(request):
    session_id = request.match_info.get("session_id")
    payload = await _cancel_pod_session(session_id)
    status = 200 if payload["status"] != "NOT_FOUND" else 404
    return web.json_response(payload, status=status)


# HTTP handler to query the status of a session.
async def pod_session_status(request):
    session_id = request.match_info.get("session_id")
    task = _active_sessions.get(session_id)

    if task is None:
        return web.json_response({
            "status": "NOT_FOUND",
            "sessionId": session_id,
        })

    if task.done():
        if task.cancelled():
            return web.json_response({
                "status": "CANCELLED",
                "sessionId": session_id,
            })

        error = task.exception()
        if error is not None:
            return web.json_response({
                "status": "FAILED",
                "sessionId": session_id,
                "error": str(error),
            })

        result = task.result()
        return web.json_response({
            "status": "COMPLETED",
            "sessionId": session_id,
            "output": result,
        })

    return web.json_response({
        "status": "RUNNING",
        "sessionId": session_id,
    })


# HTTP WebSocket ingestion endpoint: receive audio for a session.
# Auth is via signed JWT passed as Sec-WebSocket-Protocol: aivatar.<jwt>.
# The JWT carries session config (replacing the old base64 cfg blob) and is
# verified with WORKER_AUTH_SECRET. If the session isn't registered locally,
# the JWT claims are used to self-sufficiently start the streaming task
# (resolves Modal session affinity where /sessions/start and /ws/{id} may
# hit different containers).
async def app_websocket_ingest(request):
    request_received_at = time.monotonic()
    session_id = request.match_info.get("session_id")
    client_ip = request.remote
    origin = request.headers.get("Origin", "")
    logger.info("WS_REQUEST_RECEIVED session=%s origin=%s ip=%s", session_id, origin, client_ip)

    # --- Fix 9: Origin header validation ---
    if ALLOWED_WS_ORIGINS:
        if origin not in ALLOWED_WS_ORIGINS:
            logger.warning("WS_ORIGIN_REJECTED session=%s origin=%s ip=%s", session_id, origin, client_ip)
            raise web.HTTPForbidden(text="Origin not allowed")

    # --- Fix 15: IP-based connection throttling ---
    if not _check_ip_limit(client_ip, WS_MAX_CONNECTIONS_PER_IP):
        logger.warning("WS_IP_THROTTLED ip=%s session=%s", client_ip, session_id)
        raise web.HTTPTooManyRequests(text="Too many connections from this IP")

    # --- Fix 3: Extract JWT from Sec-WebSocket-Protocol ---
    protocols = request.headers.get("Sec-WebSocket-Protocol", "")
    auth_token = None
    for proto in protocols.split(","):
        proto = proto.strip()
        if proto.startswith("aivatar."):
            auth_token = proto[len("aivatar."):]
            break

    if not auth_token:
        logger.warning("WS_AUTH_FAIL session=%s reason=missing_token ip=%s", session_id, client_ip)
        raise web.HTTPUnauthorized(text="Missing auth token")

    # --- Fix 1: Verify JWT ---
    if not WORKER_AUTH_SECRET:
        raise web.HTTPInternalServerError(text="WORKER_AUTH_SECRET not configured")

    try:
        claims = pyjwt.decode(auth_token, WORKER_AUTH_SECRET, algorithms=["HS256"])
    except pyjwt.ExpiredSignatureError:
        logger.warning("WS_AUTH_FAIL session=%s reason=expired_token ip=%s", session_id, client_ip)
        raise web.HTTPUnauthorized(text="Token expired")
    except pyjwt.InvalidTokenError as _jwt_err:
        logger.warning("WS_AUTH_FAIL session=%s reason=invalid_token ip=%s detail=%s token_len=%d", session_id, client_ip, _jwt_err, len(auth_token))
        raise web.HTTPUnauthorized(text="Invalid token")

    # Verify session_id in JWT matches URL path
    if claims.get("sessionId") != session_id:
        logger.warning("WS_AUTH_FAIL session=%s reason=session_id_mismatch ip=%s", session_id, client_ip)
        raise web.HTTPUnauthorized(text="Session ID mismatch")

    # --- Fix 13: One-time JWT (jti tracking) ---
    if not _check_jti(claims):
        logger.warning("WS_AUTH_FAIL session=%s reason=token_already_used ip=%s", session_id, client_ip)
        raise web.HTTPUnauthorized(text="Token already used")

    # --- Self-sufficient mode: auto-start if not registered ---
    session_state = ws_server.active_sessions.get(session_id)
    if session_state is None:
        if claims.get("reconnect") is True:
            raise web.HTTPConflict(text="Session is not ready for reconnect")
        # Fix 7: JWT auto-start freshness check
        iat = claims.get("iat", 0)
        if time.time() - iat > 120:
            logger.warning("WS_AUTH_FAIL session=%s reason=token_too_old_for_autostart ip=%s", session_id, client_ip)
            raise web.HTTPUnauthorized(text="Token too old for auto-start")

        ingestion_token = claims.get("ingestionToken", "")
        if not ingestion_token:
            raise web.HTTPBadRequest(text="ingestionToken missing from token claims")

        ws_server.register_session(session_id, ingestion_token)

        try:
            await _start_pod_session(claims)
        except ValueError as exc:
            if "already active" in str(exc).lower():
                logger.info("Session %s already active on this container", session_id)
            else:
                raise web.HTTPBadRequest(text=str(exc)) from exc

        session_state = ws_server.active_sessions.get(session_id)

    if session_state is None:
        raise web.HTTPNotFound(text="Unknown session ID")

    if session_state.get("ending"):
        raise web.HTTPGone(text="Session is ending")

    # --- Fix 2: Require non-empty token (fail closed) ---
    provided_token = claims.get("ingestionToken", "")
    expected_token = session_state.get("token")
    if not expected_token or provided_token != expected_token:
        if provided_token and claims.get("reconnect") is True and not ws_server.has_connection(session_id):
            session_state["token"] = provided_token
            logger.info("WS_RECONNECT_TOKEN_ROTATED session=%s ip=%s", session_id, client_ip)
        else:
            logger.warning("WS_AUTH_FAIL session=%s reason=invalid_ingestion_token ip=%s", session_id, client_ip)
            raise web.HTTPUnauthorized(text="Unauthorized: Invalid Token")

    # --- Fix 5: Single connection per session ---
    if not ws_server.acquire_connection(session_id):
        logger.warning("WS_DUP_CONNECTION session=%s ip=%s", session_id, client_ip)
        raise web.HTTPConflict(text="Session already has an active connection")

    # --- Fix 3: Echo subprotocol for WS upgrade ---
    websocket = web.WebSocketResponse(protocols=[f"aivatar.{auth_token}"])
    await websocket.prepare(request)

    audio_queue = session_state["queue"]
    rate_limiter = AudioRateLimiter(WS_AUDIO_RATE_PER_SEC, WS_AUDIO_BURST, WS_MAX_BYTES_PER_SEC)

    session_state.setdefault("last_activity_time", time.monotonic())
    session_state.setdefault("is_draining", False)
    session_state["connection_lost"] = False
    session_state["websocket"] = websocket
    if session_state.get("idle_watcher") is None or session_state["idle_watcher"].done():
        session_state["idle_watcher"] = asyncio.create_task(
            _watch_session_idle(session_id, session_state),
            name=f"idle-watcher-{session_id}",
        )

    logger.info(
        "WS_CONNECT session=%s origin=%s ip=%s handlerMs=%.1f",
        session_id,
        origin,
        client_ip,
        (time.monotonic() - request_received_at) * 1000,
    )

    try:
        # --- Fix 6: Inactivity timeout ---
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=WS_INACTIVITY_TIMEOUT
                )
            except RuntimeError as exc:
                if "WebSocket connection is closed" in str(exc):
                    logger.info("WS_CLOSED session=%s ip=%s", session_id, client_ip)
                    break
                raise
            except asyncio.TimeoutError:
                logger.info("WS_INACTIVITY_TIMEOUT session=%s ip=%s", session_id, client_ip)
                await websocket.close(code=1000)
                break

            if message.type == web.WSMsgType.BINARY:
                # --- Fix 4: Message size limit ---
                if len(message.data) > MAX_AUDIO_CHUNK_BYTES:
                    logger.warning("WS_OVERSIZED session=%s size=%d max=%d ip=%s",
                                   session_id, len(message.data), MAX_AUDIO_CHUNK_BYTES, client_ip)
                    await websocket.close(code=1009)
                    break

                # --- Fix 8: Audio rate limiting (message count + byte rate) ---
                allowed, reason = rate_limiter.allow(len(message.data))
                if not allowed:
                    logger.warning("WS_RATE_LIMIT session=%s reason=%s ip=%s", session_id, reason, client_ip)
                    await websocket.close(code=1013)
                    break

                session_state["last_activity_time"] = time.monotonic()
                await audio_queue.put(message.data)
            elif message.type == web.WSMsgType.TEXT:
                try:
                    control = json.loads(message.data)
                except json.JSONDecodeError:
                    await websocket.close(code=1003)
                    break

                control_type = control.get("type") if isinstance(control, dict) else None
                if control_type == "keepalive":
                    continue
                if control_type == "end_utterance":
                    session_state["is_draining"] = True
                    session_state["last_activity_time"] = time.monotonic()
                    await audio_queue.put({
                        "type": "end_utterance",
                        "utteranceId": control.get("utteranceId"),
                    })
                    logger.info("UTTERANCE_END_RECEIVED session=%s", session_id)
                    continue
                if control_type == "end_session":
                    session_state["ending"] = True
                    await audio_queue.put({"type": "end_session", "reason": "client_requested"})
                    logger.info("SESSION_END_RECEIVED session=%s", session_id)
                    break

                await websocket.close(code=1003)
                break
            elif message.type == web.WSMsgType.ERROR:
                logger.warning("WS_ERROR session=%s ip=%s", session_id, client_ip)
                break
    finally:
        session_task = _active_sessions.get(session_id)
        ws_server.release_connection(session_id)
        if session_state.get("websocket") is websocket:
            session_state["websocket"] = None

        if session_state.get("ending"):
            if session_task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await session_task
            logger.info("WS_SESSION_ENDED session=%s ip=%s", session_id, client_ip)
        else:
            # Keep the pipeline alive while the relay performs its two reconnect attempts.
            session_state["connection_lost"] = True
            logger.info("WS_CONNECTION_LOST session=%s ip=%s", session_id, client_ip)

    return websocket


# Run the HTTP application for load balancer mode.
async def run_pod_app():
    from app_factory import build_app

    configure_model_readiness(ready=True)
    app = await build_app()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, WORKER_HTTP_HOST, WORKER_HTTP_PORT)
    await site.start()

    logger.info("AiVatar load balancer worker listening on http://%s:%s", WORKER_HTTP_HOST, WORKER_HTTP_PORT)

    # Keep running until cancelled
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(run_pod_app())
