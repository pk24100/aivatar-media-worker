"""Streaming-session lifecycle: start/cancel, execution dispatch, ws server bootstrap.

Shared state, config and sibling helpers are read via `handler.*` at call
time so handler-level monkey-patching keeps working.
"""

import asyncio
import contextlib
import logging
import os

from aiohttp import web

import handler
from utils.errors import log_exception, make_task_guard
from streaming.lifecycle.worker import (
    LifecycleError,
    WorkerLifecycleClient,
    WorkerLocalLifecycleTransition,
)
from streaming.orchestration.guards import reject_parked_session_config
from streaming.orchestration.stream_processor import run_streaming_session

logger = logging.getLogger("handler")


# Ensure the WebSocket ingestion server is running.
async def _ensure_ws_server_started():
    if handler._ws_started and handler.ws_server.is_running:
        return

    async with handler._ws_lock:
        if handler._ws_started and handler.ws_server.is_running:
            return

        await handler.ws_server.start()
        handler._ws_started = True


# Extract or validate the transport room URL from the event.
def _get_room_url(event):
    room = event.get("room") if isinstance(event.get("room"), dict) else {}
    room_url = event.get("roomUrl") or room.get("url") or event.get("customLivekitUrl")
    if not room_url:
        raise ValueError("room.url or roomUrl is required for streaming mode")
    return room_url


# Kept as an internal alias for Phase 1 helper callers while the worker API is
# transport-neutral.
def _get_livekit_url(event):
    return handler._get_room_url(event)


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
    room = event.get("room") if isinstance(event.get("room"), dict) else {}
    room_name = event.get("roomName") or event.get("sessionId") or ""
    room_token = event.get("roomToken") or room.get("token") or event.get("livekitToken")
    room_url = handler._get_room_url(event)
    egress_type = event.get("egressType") or room.get("type")
    source_image = event.get("sourceImage")
    idle_video_url = event.get("idleVideoUrl")
    idle_video_key = event.get("idleVideoKey")
    session_id = event.get("sessionId") or room_name
    ingestion_token = event.get("ingestionToken", "")

    if not session_id:
        raise ValueError("sessionId is required for streaming mode")
    if not source_image:
        raise ValueError("sourceImage is required for streaming mode")
    if not room_token:
        raise ValueError("room.token or roomToken is required for streaming mode")
    reject_parked_session_config(event)

    import time as _htime

    _session_t0 = _htime.monotonic()
    try:
        if handler._BATCHED_INFERENCE and handler.batched_engine is not None:
            await handler.run_batched_streaming_session(
                room_name=room_name,
                room_token=room_token,
                room_url=room_url,
                egress_type=egress_type,
                room_metadata=room,
                engine=handler.batched_engine,
                source_image=source_image,
                session_id=session_id,
                ingestion_token=ingestion_token,
                idle_video_url=idle_video_url,
                idle_video_key=idle_video_key,
            )
            _session_ms = round((_htime.monotonic() - _session_t0) * 1000, 1)
            logger.info("[handler] run_batched_streaming_session() completed in %.1fms for %s", _session_ms, session_id)
        else:
            await run_streaming_session(
                room_name=room_name,
                room_token=room_token,
                room_url=room_url,
                egress_type=egress_type,
                room_metadata=room,
                pipeline=None,
                model_pool=handler.model_pool,
                model_ready_waiter=handler.wait_for_model_ready,
                source_image=source_image,
                session_id=session_id,
                ingestion_token=ingestion_token,
                idle_video_url=idle_video_url,
                idle_video_key=idle_video_key,
            )
            _session_ms = round((_htime.monotonic() - _session_t0) * 1000, 1)
            logger.info("[handler] run_streaming_session() completed in %.1fms for %s", _session_ms, session_id)
        return {"status": "ok", "mode": "streaming", "sessionId": session_id}
    except Exception as exc:
        _session_ms = round((_htime.monotonic() - _session_t0) * 1000, 1)
        logger.error(
            "[handler] Streaming session failed for %s after %.1fms error=%s",
            session_id,
            _session_ms,
            exc.__class__.__name__,
        )
        return {
            "status": "error",
            "mode": "streaming",
            "sessionId": session_id,
            "error": "Streaming session failed",
        }


# Start a new streaming session on the pod after globally claiming its owner lease.
async def _start_pod_session(
    event,
    *,
    lifecycle: WorkerLifecycleClient | None = None,
    connection_token: str | None = None,
    lifecycle_claimed: bool = False,
):
    await handler._ensure_ws_server_started()

    session_id = event.get("sessionId") or event.get("roomName")
    if not session_id:
        raise ValueError("sessionId or roomName is required")
    if not (handler._BATCHED_INFERENCE and handler.batched_engine is not None) and handler.model_pool.get_available_count() <= 0:
        raise ValueError("No pipeline capacity available")

    reject_parked_session_config(event)
    # #1 early egress warm (BYOLR: room_url only known here, not at restore).
    # Fire-and-forget; never blocks claim or session setup.
    try:
        _warm_url = event.get("roomUrl") or event.get("customLivekitUrl")
        if not _warm_url:
            _warm_room = event.get("room") if isinstance(event.get("room"), dict) else {}
            _warm_url = _warm_room.get("url") if isinstance(_warm_room, dict) else None
    except Exception as exc:
        log_exception(logger, exc, "EGRESS_WARM_URL_EXTRACT", session_id, level="debug")
        _warm_url = None
    if _warm_url:
        try:
            from streaming.transport.adapters.livekit_egress import warm_egress_for_url

            _wt = asyncio.create_task(
                asyncio.to_thread(warm_egress_for_url, _warm_url),
                name=f"egress-warm-{session_id}",
            )
            _wt.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            _wt.add_done_callback(make_task_guard(logger, session_id, "EGRESS_WARM_TASK"))
        except Exception as exc:
            log_exception(logger, exc, "EGRESS_WARM_DISPATCH", session_id, level="debug")
            pass
    lifecycle = lifecycle or handler._lifecycle_from_event(session_id, event)
    if lifecycle is not None and not lifecycle.is_configured:
        raise ValueError("Worker lifecycle requires BACKEND_INTERNAL_URL and WORKER_AUTH_SECRET")

    if lifecycle is not None and not lifecycle_claimed:
        try:
            await lifecycle.claim(connection_token=connection_token)
        except LifecycleError as exc:
            logger.warning(
                "WORKER_OWNER_CLAIM_REJECTED session=%s code=%s status=%s",
                session_id,
                exc.code,
                exc.status,
            )
            raise web.HTTPConflict(text="Session owner claim rejected") from exc

    if session_id in handler._active_sessions:
        logger.info("Session %s already active; treating start request as idempotent", session_id)
        return {
            "jobId": session_id,
            "status": "ALREADY_STARTED",
            "sessionId": session_id,
        }

    ingestion_token = event.get("ingestionToken", "")
    if not ingestion_token:
        raise ValueError("ingestionToken is required for WebSocket ingestion")

    session_state = handler.ws_server.register_session(
        session_id,
        ingestion_token,
        input_provider=event.get("inputProvider"),
        telephony_provider=None,
    )
    # register_session returns the audio queue, so retrieve the state once here.
    session_state = handler.ws_server.active_sessions[session_id]
    session_state.setdefault("lifecycle_transition", WorkerLocalLifecycleTransition())
    session_state.setdefault("finalization_complete", asyncio.Event())
    if lifecycle is not None:
        session_state["lifecycle"] = lifecycle

    try:
        task = asyncio.create_task(handler._execute_streaming_event(event), name=f"session-{session_id}")
        task.add_done_callback(make_task_guard(logger, session_id, "SESSION_TASK"))
        handler._track_session_task(session_id, task, session_state)
        await handler._start_session_heartbeat(session_id, session_state)
        return {
            "jobId": session_id,
            "status": "STARTED",
            "sessionId": session_id,
        }
    except Exception as exc:
        log_exception(logger, exc, "SESSION_START_TASK", session_id)
        handler.ws_server.unregister_session(session_id)
        if lifecycle is not None:
            await lifecycle.stop(release=True)
        raise


async def request_active_session_shutdown(reason: str = "worker_shutdown") -> int:
    """Signal every active processor and close its aiohttp ingestion socket."""
    session_ids = list(handler._active_sessions)
    requests = []
    for session_id in session_ids:
        session_state = handler.ws_server.active_sessions.get(session_id)
        if session_state is None:
            task = handler._active_sessions.get(session_id)
            if task is not None:
                task.cancel()
            continue
        await handler._cancel_reconnect_expiry(session_state)
        requests.append(
            handler._request_owner_termination(
                session_id,
                session_state,
                {"reason": reason},
            )
        )
    if requests:
        signal_tasks = [
            asyncio.create_task(request, name="session-shutdown-signal")
            for request in requests
        ]
        for _st in signal_tasks:
            _st.add_done_callback(make_task_guard(logger, "-", "SESSION_SHUTDOWN_SIGNAL"))
        _, pending = await asyncio.wait(signal_tasks, timeout=5)
        if pending:
            logger.warning("SESSION_SHUTDOWN_SIGNAL_TIMEOUT pending=%d", len(pending))
            for task in pending:
                task.cancel()
        await asyncio.gather(*signal_tasks, return_exceptions=True)
    return len(session_ids)


# End an active session by signaling its processor and ingestion connection.
async def _cancel_pod_session(session_id):
    task = handler._active_sessions.get(session_id)
    if task is None:
        return {"status": "NOT_FOUND", "sessionId": session_id}

    session_state = handler.ws_server.active_sessions.get(session_id)
    if session_state is not None:
        await handler._request_owner_termination(
            session_id,
            session_state,
            {"reason": "provider_requested"},
        )

    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=10)
    except asyncio.TimeoutError:
        logger.warning("SESSION_END_TIMEOUT session=%s", session_id)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    except Exception as exc:
        log_exception(logger, exc, "SESSION_END_WAIT", session_id, level="debug")
        pass

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
