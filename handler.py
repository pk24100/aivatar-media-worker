import asyncio
import contextlib
import logging
import os

from aiohttp import web
from huggingface_hub import snapshot_download

from streaming.stream_processor import run_streaming_session
from streaming.websocket_server import ws_server
from utils.model_pool import FlashHeadModelPool

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


# Acquire a model and run a streaming session for the given event.
async def _execute_streaming_event(event):
    room_name = event.get("roomName")
    livekit_token = event.get("livekitToken")
    source_image = event.get("sourceImage")
    idle_video_url = event.get("idleVideoUrl")
    ingestion_method = event.get("ingestionMethod", "livekit")
    session_id = event.get("sessionId", room_name)
    ingestion_token = event.get("ingestionToken", "")

    if not room_name or not livekit_token:
        raise ValueError("roomName and livekitToken are required for streaming mode")
    if not source_image:
        raise ValueError("sourceImage is required for streaming mode")

    model_instance = await model_pool.acquire()
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
        return {"status": "ok", "mode": "streaming", "sessionId": session_id}
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

    task = asyncio.create_task(_execute_streaming_event(event))
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
    }, status=200 if ready else 503)


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
    if not _is_streaming(event):
        raise web.HTTPBadRequest(text='Only streaming mode is supported')

    try:
        payload = await _start_pod_session(event)
        return web.json_response(payload, status=202)
    except ValueError as error:
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
async def app_websocket_ingest(request):
    session_id = request.match_info.get("session_id")
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
    await _ensure_ws_server_started()

    app = web.Application()
    app.router.add_get('/ping', app_ping)
    app.router.add_get('/health', pod_health)
    app.router.add_get('/healthz', pod_health)
    app.router.add_get('/readyz', pod_ready)
    app.router.add_post('/sessions/start', pod_session_start)
    app.router.add_post('/sessions/{session_id}/end', pod_session_end)
    app.router.add_get('/sessions/{session_id}/status', pod_session_status)
    app.router.add_get('/ws/{session_id}', app_websocket_ingest)

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