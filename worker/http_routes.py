"""aiohttp route handlers for health/readiness and session control.

Shared state and sibling helpers are read via `handler.*` at call time so
handler-level monkey-patching keeps working.
"""

import logging

from aiohttp import web

import handler
from utils.errors import log_exception

logger = logging.getLogger("handler")


# HTTP health endpoint: report worker health and capacity.
async def pod_health(_request):
    return web.json_response({
        "ok": True,
        "status": "healthy",
        "runtimeMode": "load_balancer",
        "activeSessions": len(handler._active_sessions),
        "availablePipelines": handler.model_pool.get_available_count(),
    })


# HTTP readiness endpoint: report whether the worker is ready.
# Narrow gating: ONLY ws_server.is_running + model_ready_event + batched_engine
# thread alive. poolSize/availablePipelines/warmupDone are fields only, never gates.
# Must stay fast (ms, under 5s HEALTH_TIMEOUT): no awaits, no blocking calls.
async def pod_ready(_request):
    try:
        server_ready = bool(handler.ws_server.is_running)
    except Exception:
        server_ready = False
    try:
        _evt = handler._model_ready_event
        _err = handler._model_init_error
        models_ready = bool(_evt is not None and _evt.is_set() and _err is None)
    except Exception:
        models_ready = False
    # Batched engine thread alive (is_alive if exists). No engine -> no gating.
    try:
        _engine = getattr(handler, "batched_engine", None)
        if _engine is None:
            batched_ready = True
        elif bool(getattr(_engine, "_stopped", False)):
            batched_ready = False
        else:
            _thread = getattr(_engine, "_thread", None)
            if _thread is None:
                batched_ready = False
            else:
                batched_ready = bool(_thread.is_alive())
    except Exception:
        batched_ready = False
    ready = bool(server_ready and models_ready and batched_ready)
    # Fields only (never gate on warmup or pool>0).
    try:
        pool_size = int(handler.WORKER_POOL_SIZE)
    except Exception:
        pool_size = 0
    try:
        available = int(handler.model_pool.get_available_count())
    except Exception:
        available = 0
    try:
        _warmup_thread = getattr(handler, "_batched_warmup_thread", None)
        if _warmup_thread is None:
            warmup_done = True
        else:
            warmup_done = not bool(_warmup_thread.is_alive())
    except Exception:
        warmup_done = True
    try:
        active = len(handler._active_sessions)
    except Exception:
        active = 0
    return web.json_response({
        "ready": ready,
        "status": "READY" if ready else "STARTING",
        "serverReady": server_ready,
        "modelsReady": models_ready,
        "batchedReady": batched_ready,
        "runtimeMode": "load_balancer",
        "poolSize": pool_size,
        "availablePipelines": available,
        "warmupDone": warmup_done,
        "activeSessions": active,
    }, status=200 if ready else 503)


# Lightweight HTTP ping endpoint.
async def app_ping(_request):
    if handler.ws_server.is_running:
        return web.json_response({
            "status": "healthy",
            "runtimeMode": "load_balancer",
        })

    return web.Response(status=204)


# HTTP handler to start a new streaming session.
async def pod_session_start(request):
    try:
        event = await request.json()
    except Exception as exc:
        log_exception(logger, exc, "SESSION_START_BAD_JSON", "-")
        raise
    logger.info(
        "[pod_session_start] Received session=%s room=%s hasIdleAsset=%s",
        event.get("sessionId"),
        event.get("roomName"),
        bool(event.get("idleVideoKey") or event.get("idleVideoUrl")),
    )
    if not handler._is_streaming(event):
        logger.warning("[pod_session_start] Not streaming mode for session=%s", event.get("sessionId"))
        raise web.HTTPBadRequest(text='Only streaming mode is supported')

    try:
        payload = await handler._start_pod_session(event)
        return web.json_response(payload, status=202)
    except ValueError as error:
        logger.error("[pod_session_start] ValueError: %s", error)
        raise web.HTTPBadRequest(text=str(error)) from error


# HTTP handler to end an active streaming session.
async def pod_session_end(request):
    session_id = request.match_info.get("session_id")
    payload = await handler._cancel_pod_session(session_id)
    status = 200 if payload["status"] != "NOT_FOUND" else 404
    return web.json_response(payload, status=status)


# HTTP handler to query the status of a session.
async def pod_session_status(request):
    session_id = request.match_info.get("session_id")
    task = handler._active_sessions.get(session_id)

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

        try:
            error = task.exception()
        except Exception as exc:
            log_exception(logger, exc, "SESSION_STATUS_EXCEPTION_READ", session_id)
            raise
        if error is not None:
            return web.json_response({
                "status": "FAILED",
                "sessionId": session_id,
                "error": "Streaming session failed",
            })

        try:
            result = task.result()
        except Exception as exc:
            log_exception(logger, exc, "SESSION_STATUS_RESULT_READ", session_id)
            raise
        return web.json_response({
            "status": "COMPLETED",
            "sessionId": session_id,
            "output": result,
        })

    return web.json_response({
        "status": "RUNNING",
        "sessionId": session_id,
    })
