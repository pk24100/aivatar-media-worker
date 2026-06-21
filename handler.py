import asyncio
import contextlib
import logging
import os
import time

from aiohttp import web
from huggingface_hub import snapshot_download

from streaming.stream_processor import run_streaming_session
from streaming.websocket_server import ws_server
from utils.model_pool import FlashHeadModelPool


PREWARM_ROOM_TIMEOUT = 60.0


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
                _t0 = time.monotonic()
                logger.info("[prewarm] Starting fire-and-forget disconnect for %s", room_name)
                await entry["room"].disconnect()
                logger.info("[prewarm] Fire-and-forget disconnect completed for %s in %.1fms", room_name, (time.monotonic() - _t0) * 1000)
            except Exception as e:
                logger.warning("[prewarm] Fire-and-forget disconnect failed for %s: %s", room_name, e)
        asyncio.create_task(_bg_disconnect())
        logger.info("[prewarm] Released room %s (disconnect fire-and-forget)", room_name)

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

# Initialize model pool globally so it happens during FlashBoot
model_pool = FlashHeadModelPool(size=WORKER_POOL_SIZE, ckpt_dir=FLASHHEAD_CKPT_DIR, wav2vec_dir=WAV2VEC_DIR)
prewarm_pool = PrewarmRoomPool(size=WORKER_POOL_SIZE)
_ws_started = False
_ws_lock = asyncio.Lock()
_active_sessions = {}

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

    # Disconnect the pre-warm participant BEFORE connecting the real session.
    # This must be awaited (not fire-and-forget) to avoid DuplicateIdentity.
    import time as _htime
    _release_t0 = _htime.monotonic()
    try:
        await prewarm_pool.release(room_name)
    except Exception as exc:
        logger.warning("[handler] Failed to release pre-warm room %s: %s", room_name, exc)
    _release_ms = round((_htime.monotonic() - _release_t0) * 1000, 1)
    logger.info("[handler] prewarm_pool.release() took %.1fms for room %s", _release_ms, room_name)

    _acquire_t0 = _htime.monotonic()
    model_instance = await model_pool.acquire()
    _acquire_ms = round((_htime.monotonic() - _acquire_t0) * 1000, 1)
    logger.info("[handler] model_pool.acquire() took %.1fms", _acquire_ms)

    _session_t0 = _htime.monotonic()
    try:
        await run_streaming_session(
            room_name=room_name,
            livekit_token=livekit_token,
            livekit_url=_get_livekit_url(event),
            pipeline=model_instance,
            source_image=source_image,
            ingestion_method=ingestion_method,
            session_id=session_id,
            ingestion_token=ingestion_token,
            idle_video_url=idle_video_url,
        )
        _session_ms = round((_htime.monotonic() - _session_t0) * 1000, 1)
        logger.info("[handler] run_streaming_session() completed in %.1fms for %s", _session_ms, session_id)
        return {"status": "ok", "mode": "streaming", "sessionId": session_id}
    except Exception as exc:
        _session_ms = round((_htime.monotonic() - _session_t0) * 1000, 1)
        logger.error("[handler] Streaming session failed for %s after %.1fms: %s", session_id, _session_ms, exc, exc_info=True)
        return {"status": "error", "mode": "streaming", "sessionId": session_id, "error": str(exc)}
    finally:
        model_pool.release(model_instance)


# Track an active session task and clean up when it finishes.
def _track_session_task(session_id, task):
    _active_sessions[session_id] = task

    # Remove session from tracking when the task completes.
    def _cleanup(_task):
        _active_sessions.pop(session_id, None)

    task.add_done_callback(_cleanup)


# Start a new streaming session on the pod and track it.
async def _start_pod_session(event):
    await _ensure_ws_server_started()

    session_id = event.get("sessionId") or event.get("roomName")
    if not session_id:
        raise ValueError("sessionId or roomName is required")
    if session_id in _active_sessions:
        raise ValueError(f"Session {session_id} is already active")
    if model_pool.get_available_count() <= 0:
        raise ValueError("No pipeline capacity available")

    if event.get("ingestionMethod", "websocket") == "websocket":
        ws_server.register_session(session_id, event.get("ingestionToken", ""))

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
    ready = ws_server.is_running
    return web.json_response({
        "ready": ready,
        "status": "READY" if ready else "STARTING",
        "runtimeMode": "load_balancer",
        "poolSize": WORKER_POOL_SIZE,
        "availablePipelines": model_pool.get_available_count(),
        "activeSessions": len(_active_sessions),
        "prewarmRoomsAvailable": prewarm_pool.available_count(),
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
    logger.info("[pod_session_start] Received event: %s", event)
    if not _is_streaming(event):
        logger.warning("[pod_session_start] Not streaming mode: %s", event)
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
# Approach A (self-sufficient): if the session isn't registered locally,
# the backend may embed session config in the ?cfg= query param so the
# WS handler can start the streaming task on first connect. This resolves
# session affinity on Modal where /sessions/start and /ws/{id} may hit
# different containers. Idle video covers the small pipeline-acquisition gap.
async def app_websocket_ingest(request):
    session_id = request.match_info.get("session_id")
    session_state = ws_server.active_sessions.get(session_id)

    if session_state is None:
        encoded_cfg = request.query.get("cfg")
        if encoded_cfg:
            try:
                import json
                import base64
                # URL-safe base64 padding restoration
                padded = encoded_cfg + "=" * (4 - len(encoded_cfg) % 4)
                cfg = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
            except Exception as e:
                logger.warning("Failed to decode WS session config: %s", e)
                raise web.HTTPBadRequest(text="Invalid session config") from e

            # Pre-register the audio queue so the streaming task can find it.
            ws_server.register_session(session_id, cfg.get("ingestionToken", ""))

            # Start the streaming task if not already active on this container.
            try:
                await _start_pod_session(cfg)
            except ValueError as exc:
                if "already active" in str(exc).lower():
                    logger.info("Session %s already active on this container", session_id)
                else:
                    raise web.HTTPBadRequest(text=str(exc)) from exc

            session_state = ws_server.active_sessions.get(session_id)

    if session_state is None:
        raise web.HTTPNotFound(text="Unknown session ID")

    provided_token = request.query.get("token")
    expected_token = session_state.get("token")
    if expected_token and provided_token != expected_token:
        raise web.HTTPUnauthorized(text="Unauthorized: Invalid Token")

    audio_queue = session_state["queue"]
    websocket = web.WebSocketResponse()
    await websocket.prepare(request)

    try:
        async for message in websocket:
            if message.type == web.WSMsgType.BINARY:
                await audio_queue.put(message.data)
            elif message.type == web.WSMsgType.ERROR:
                break
    finally:
        await audio_queue.put(None)

    return websocket


# Run the HTTP application for load balancer mode.
async def run_pod_app():
    from app_factory import build_app

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