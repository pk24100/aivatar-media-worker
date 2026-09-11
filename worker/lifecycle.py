"""Backend lifecycle reporting: termination requests, heartbeats, idle/reconnect timers.

Shared config and sibling helpers are read via `handler.*` at call time so
handler-level monkey-patching keeps working.
"""

import asyncio
import contextlib
import logging
import time
from datetime import datetime

import aiohttp
import jwt as pyjwt

import handler
from streaming.lifecycle.worker import LifecycleError, WorkerLifecycleClient

logger = logging.getLogger("handler")


async def _end_session_via_backend(session_id: str, reason: str = "no_audio_timeout"):
    """Call backend POST /internal/sessions/:id/end to finalize a stale session.
    Uses WORKER_AUTH_SECRET to mint a JWT for authentication."""
    if not handler.BACKEND_INTERNAL_URL:
        logger.warning("NO_AUDIO_TIMEOUT session=%s reason=%s BACKEND_INTERNAL_URL not set, cannot end session", session_id, reason)
        return

    if not handler.WORKER_AUTH_SECRET:
        logger.warning("NO_AUDIO_TIMEOUT session=%s reason=%s WORKER_AUTH_SECRET not set, cannot end session", session_id, reason)
        return

    token = pyjwt.encode(
        {"sessionId": session_id, "reason": reason, "iat": int(time.time())},
        handler.WORKER_AUTH_SECRET,
        algorithm="HS256",
    )

    url = f"{handler.BACKEND_INTERNAL_URL.rstrip('/')}/internal/sessions/{session_id}/end"
    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={"reason": reason},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    logger.info("NO_AUDIO_TIMEOUT session=%s backend end success status=%d", session_id, resp.status)
                else:
                    logger.warning(
                        "NO_AUDIO_TIMEOUT session=%s backend end failed status=%d",
                        session_id,
                        resp.status,
                    )
    except Exception as exc:
        logger.warning(
            "NO_AUDIO_TIMEOUT session=%s backend end error=%s",
            session_id,
            exc.__class__.__name__,
        )


async def _request_owner_termination(session_id: str, session_state: dict, terminate: dict | None = None) -> None:
    """Apply a backend-delivered termination command to the local pipeline."""
    if session_state.get("ending"):
        return
    session_state["ending"] = True
    session_state["termination_reason"] = (terminate or {}).get("reason", "backend_requested")
    logger.info(
        "WORKER_TERMINATION_REQUESTED session=%s reason=%s",
        session_id,
        session_state["termination_reason"],
    )
    try:
        await session_state["bus"].end_session()
    except (RuntimeError, KeyError):
        queue = session_state.get("queue")
        if queue is not None:
            await queue.put(None)

    websocket = session_state.get("websocket")
    if websocket is not None:
        with contextlib.suppress(Exception):
            await websocket.send_json({
                "type": "session_ending",
                "reason": session_state["termination_reason"],
            })
            await websocket.close(code=1000, message=b"Session termination requested")


def _lifecycle_from_event(session_id: str, event: dict) -> WorkerLifecycleClient | None:
    lease_id = event.get("workerLeaseId")
    generation = event.get("workerGeneration")
    if not lease_id or generation is None:
        return None
    return WorkerLifecycleClient(
        session_id,
        lease_id=lease_id,
        generation=generation,
    )


async def _start_session_heartbeat(session_id: str, session_state: dict) -> None:
    lifecycle = session_state.get("lifecycle")
    if lifecycle is None:
        return

    async def on_terminate(command: dict) -> None:
        await handler._request_owner_termination(session_id, session_state, command)

    lifecycle.start_heartbeat(on_terminate)
    if session_state.get("lifecycle_started"):
        return
    session_state["lifecycle_started"] = True
    try:
        result = await lifecycle.send_event("worker_started")
    except LifecycleError as exc:
        logger.warning(
            "WORKER_LIFECYCLE_START_EVENT_FAILED session=%s code=%s status=%s",
            session_id,
            exc.code,
            exc.status,
        )
        return
    terminate = result.get("terminate") if isinstance(result, dict) else None
    if terminate:
        await on_terminate(terminate)


async def _watch_session_idle(session_id: str, session_state: dict) -> None:
    """End only this session after it has been inactive outside a tail drain."""
    while True:
        await asyncio.sleep(handler.SESSION_IDLE_CHECK_INTERVAL)
        if session_state.get("ending"):
            return
        if session_state.get("is_draining"):
            continue

        elapsed = time.monotonic() - session_state.get("last_activity_time", time.monotonic())
        if elapsed < handler.NO_AUDIO_SESSION_TIMEOUT:
            continue

        logger.info(
            "SESSION_IDLE_TIMEOUT session=%s elapsed=%.0fs threshold=%ds",
            session_id,
            elapsed,
            handler.NO_AUDIO_SESSION_TIMEOUT,
        )
        await handler._request_owner_termination(
            session_id,
            session_state,
            {"reason": "no_audio_timeout"},
        )
        return


async def _terminate_after_reconnect_expiry(
    session_id: str,
    session_state: dict,
    expires_at: str,
) -> None:
    deadline = datetime.fromisoformat(expires_at.replace("Z", "+00:00")).timestamp()
    await asyncio.sleep(max(0.0, deadline - time.time()))
    if (
        session_state.get("reconnect_expiry_task") is asyncio.current_task()
        and session_state.get("connection_lost")
    ):
        await handler._request_owner_termination(
            session_id,
            session_state,
            {"reason": "reconnect_grace_expired"},
        )


def _schedule_reconnect_expiry(
    session_id: str,
    session_state: dict,
    expires_at: str | None,
) -> None:
    previous_task = session_state.get("reconnect_expiry_task")
    if previous_task is not None and not previous_task.done():
        previous_task.cancel()
    if not expires_at:
        session_state.pop("reconnect_expiry_task", None)
        return
    task = asyncio.create_task(
        handler._terminate_after_reconnect_expiry(session_id, session_state, expires_at),
        name=f"reconnect-expiry-{session_id}",
    )
    session_state["reconnect_expiry_task"] = task


async def _cancel_reconnect_expiry(session_state: dict) -> None:
    task = session_state.pop("reconnect_expiry_task", None)
    if task is None:
        return
    if task is asyncio.current_task():
        return
    if task.done():
        if not task.cancelled():
            task.exception()
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
