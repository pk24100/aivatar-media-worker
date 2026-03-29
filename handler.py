import asyncio
import contextlib
import logging
import os

from aiohttp import web
import runpod

from streaming.stream_processor import run_streaming_session
from streaming.websocket_server import ws_server
from utils.model_pool import FlashHeadModelPool

FLASHHEAD_CKPT_DIR = os.getenv("FLASHHEAD_CKPT_DIR", "/app/models/SoulX-FlashHead-1_3B")
WAV2VEC_DIR = os.getenv("WAV2VEC_DIR", "/app/models/wav2vec2-base-960h")
FLASHHEAD_REPO_PATH = os.getenv("FLASHHEAD_REPO_PATH", "/app/SoulX-FlashHead")
WORKER_POOL_SIZE = int(os.getenv("AIVATAR_WORKER_CONCURRENCY", "3"))
WORKER_RUNTIME_MODE = os.getenv("AIVATAR_RUNTIME_MODE", "serverless").strip().lower()
WORKER_HTTP_HOST = os.getenv("AIVATAR_HTTP_HOST", "0.0.0.0")
WORKER_HTTP_PORT = int(os.getenv("AIVATAR_HTTP_PORT", "8000"))

logger = logging.getLogger(__name__)

# Initialize model pool globally so it happens during FlashBoot
model_pool = FlashHeadModelPool(size=WORKER_POOL_SIZE, ckpt_dir=FLASHHEAD_CKPT_DIR, wav2vec_dir=WAV2VEC_DIR)
_ws_started = False
_ws_lock = asyncio.Lock()
_active_sessions = {}

def _verify_models():
    if not os.path.exists(FLASHHEAD_CKPT_DIR):
        print(f"Warning: FlashHead checkpoint dir not found at {FLASHHEAD_CKPT_DIR}")
    if not os.path.exists(WAV2VEC_DIR):
        print(f"Warning: Wav2Vec dir not found at {WAV2VEC_DIR}")
        
    return FLASHHEAD_CKPT_DIR

DATA_ROOT = _verify_models()


async def _ensure_ws_server_started():
    global _ws_started
    if _ws_started and ws_server.is_running:
        return

    async with _ws_lock:
        if _ws_started and ws_server.is_running:
            return

        await ws_server.start()
        _ws_started = True


def _get_livekit_url(event):
    livekit_url = event.get("customLivekitUrl") or os.getenv("LIVEKIT_URL")
    if not livekit_url:
        raise ValueError("LIVEKIT_URL is required for streaming mode")
    return livekit_url


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


def _track_session_task(session_id, task):
    _active_sessions[session_id] = task

    def _cleanup(_task):
        _active_sessions.pop(session_id, None)

    task.add_done_callback(_cleanup)


async def _start_pod_session(event):
    await _ensure_ws_server_started()

    session_id = event.get("sessionId") or event.get("roomName")
    if not session_id:
        raise ValueError("sessionId or roomName is required")
    if session_id in _active_sessions:
        raise ValueError(f"Session {session_id} is already active")
    if model_pool.get_available_count() <= 0:
        raise ValueError("No pipeline capacity available on overflow pod")

    task = asyncio.create_task(_execute_streaming_event(event))
    _track_session_task(session_id, task)
    return {
        "jobId": session_id,
        "status": "STARTED",
        "sessionId": session_id,
    }


async def _cancel_pod_session(session_id):
    task = _active_sessions.get(session_id)
    if task is None:
        return {"status": "NOT_FOUND", "sessionId": session_id}

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    return {"status": "ENDED", "sessionId": session_id}


def _is_streaming(event):
    if event.get("streaming") is True:
        return True
    if str(event.get("mode", "")).lower() == "streaming":
        return True
    env_flag = os.getenv("AIVATAR_STREAMING", "").strip().lower()
    return env_flag in {"1", "true", "yes"}

async def handler_async(job):
    """
    Called per request by Runpod Serverless.
    job["input"] example:
    {
      "roomName": "room_abc",
      "livekitToken": "...",
      "audioPath": "s3://bucket/audio.wav",
      "sourceImage": "s3://bucket/avatar.png",
      "ingestionMethod": "websocket",
      "sessionId": "1234"
    }
    """
    await _ensure_ws_server_started()
    event = job["input"]

    if _is_streaming(event):
        return await _execute_streaming_event(event)

    raise NotImplementedError("Offline inference is currently disabled and stored in reference_offline_code")


async def pod_health(_request):
    return web.json_response({
        "ok": True,
        "status": "healthy",
        "runtimeMode": "pod",
        "activeSessions": len(_active_sessions),
        "availablePipelines": model_pool.get_available_count(),
    })


async def pod_ready(_request):
    ready = ws_server.is_running
    return web.json_response({
        "ready": ready,
        "status": "READY" if ready else "STARTING",
        "runtimeMode": "pod",
        "poolSize": WORKER_POOL_SIZE,
        "availablePipelines": model_pool.get_available_count(),
        "activeSessions": len(_active_sessions),
    }, status=200 if ready else 503)


async def pod_session_start(request):
    event = await request.json()
    if not _is_streaming(event):
        raise web.HTTPBadRequest(text='Only streaming mode is supported')

    try:
        payload = await _start_pod_session(event)
        return web.json_response(payload, status=202)
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error)) from error


async def pod_session_end(request):
    session_id = request.match_info.get("session_id")
    payload = await _cancel_pod_session(session_id)
    status = 200 if payload["status"] != "NOT_FOUND" else 404
    return web.json_response(payload, status=status)


async def run_pod_app():
    await _ensure_ws_server_started()

    app = web.Application()
    app.router.add_get('/healthz', pod_health)
    app.router.add_get('/readyz', pod_ready)
    app.router.add_post('/sessions/start', pod_session_start)
    app.router.add_post('/sessions/{session_id}/end', pod_session_end)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, WORKER_HTTP_HOST, WORKER_HTTP_PORT)
    await site.start()

    logger.info("AiVatar pod runtime listening on http://%s:%s", WORKER_HTTP_HOST, WORKER_HTTP_PORT)

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    if WORKER_RUNTIME_MODE == "pod":
        asyncio.run(run_pod_app())
    else:
        runpod.serverless.start({
            "handler": handler_async,
            "return_aggregate_stream": True,
            "concurrency_modifier": lambda x: WORKER_POOL_SIZE
        })
