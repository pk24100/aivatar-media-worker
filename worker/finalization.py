"""Session task tracking and terminal-state finalization.

Shared state and sibling helpers are read via `handler.*` at call time so
handler-level monkey-patching keeps working.
"""

import asyncio
import contextlib
import logging

import handler
from utils.errors import log_exception, make_task_guard
from streaming.protocol.messages import ErrorMessage, serialize_message

logger = logging.getLogger("handler")


async def _finalize_session_task(session_id: str, task, session_state: dict) -> None:
    """Report terminal lifecycle state only after the processor has cleaned up."""
    await handler._cancel_reconnect_expiry(session_state)
    lifecycle = session_state.get("lifecycle")
    failure_code = None
    failure_message = None
    terminal_event = "session_ended"

    if task.cancelled():
        terminal_event = "session_ended"
    else:
        try:
            result = task.result()
        except Exception as exc:
            log_exception(logger, exc, "FINALIZE_TASK_RESULT", session_id)
            terminal_event = "session_failed"
            failure_code = "WORKER_TASK_FAILED"
            failure_message = "Worker task failed"
        else:
            if isinstance(result, dict) and result.get("status") == "error":
                terminal_event = "session_failed"
                failure_code = "WORKER_SESSION_FAILED"
                failure_message = "Streaming session failed"

    if lifecycle is not None:
        await lifecycle.stop()
        await lifecycle.send_terminal(
            terminal_event,
            failure_code=failure_code,
            failure_message=failure_message,
        )
    elif terminal_event == "session_ended":
        await handler._end_session_via_backend(
            session_id,
            session_state.get("termination_reason", "worker_completed"),
        )
    else:
        await handler._end_session_via_backend(session_id, "worker_failed")


async def drain_finalization_tasks(timeout: float = 10.0) -> bool:
    """Boundedly drain terminal lifecycle callbacks before the serve loop closes."""
    await asyncio.sleep(0)
    deadline = asyncio.get_running_loop().time() + timeout

    while handler._finalization_tasks:
        pending = tuple(handler._finalization_tasks)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        _, still_pending = await asyncio.wait(pending, timeout=remaining)
        if not still_pending:
            await asyncio.sleep(0)

    if not handler._finalization_tasks:
        return True

    pending = tuple(handler._finalization_tasks)
    logger.warning("WORKER_FINALIZATION_DRAIN_TIMEOUT pending=%d", len(pending))
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    handler._finalization_tasks.difference_update(pending)
    return False


async def _run_session_finalizer(session_id: str, task, session_state: dict) -> None:
    try:
        if session_state.get("prepare_rollback"):
            await handler._cancel_reconnect_expiry(session_state)
            idle_watcher = session_state.get("idle_watcher")
            if idle_watcher is not None and not idle_watcher.done():
                idle_watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await idle_watcher
            lifecycle = session_state.get("lifecycle")
            if lifecycle is not None:
                await lifecycle.stop(release=True)
            handler.ws_server.unregister_session(session_id)
            return
        await handler._finalize_session_task(session_id, task, session_state)
    finally:
        completed = session_state.get("finalization_complete")
        if completed is not None:
            completed.set()


async def _rollback_self_started_session(session_id: str, session_state: dict) -> None:
    session_state["prepare_rollback"] = True
    task = handler._active_sessions.get(session_id)
    if task is not None and not task.done():
        task.cancel()
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)
    completed = session_state.get("finalization_complete")
    if completed is not None:
        await completed.wait()


def _observe_finalization_task(completed_task: asyncio.Task) -> None:
    handler._finalization_tasks.discard(completed_task)
    if not completed_task.cancelled():
        try:
            exc = completed_task.exception()
        except Exception:
            return
        if exc is not None:
            try:
                log_exception(logger, exc, "FINALIZATION_TASK", "-")
            except Exception:
                pass


def _track_session_task(session_id, task, session_state: dict):
    handler._active_sessions[session_id] = task

    def _cleanup(completed_task):
        handler._active_sessions.pop(session_id, None)
        idle_watcher = session_state.get("idle_watcher")
        if idle_watcher is not None and idle_watcher is not asyncio.current_task():
            idle_watcher.cancel()
        expiry_task = session_state.pop("reconnect_expiry_task", None)
        if expiry_task is not None and not expiry_task.done():
            expiry_task.cancel()

        task_exception = None
        task_result = None
        if not completed_task.cancelled():
            try:
                task_result = completed_task.result()
            except Exception as exc:
                task_exception = exc
                log_exception(logger, exc, "SESSION_TASK_FAILED", session_id)
        result_failed = isinstance(task_result, dict) and task_result.get("status") == "error"
        if session_state.get("websocket") and not session_state.get("ending") and (task_exception or result_failed):
            error_message = serialize_message(ErrorMessage(
                "INTERNAL_ERROR",
                "Streaming session failed",
                fatal=True,
            ))
            websocket = session_state.get("websocket")
            with contextlib.suppress(Exception):
                diagnostic = asyncio.create_task(
                    websocket.send_str(error_message),
                    name=f"failure-diagnostic-{session_id}",
                )
                handler._finalization_tasks.add(diagnostic)
                diagnostic.add_done_callback(handler._finalization_tasks.discard)
                diagnostic.add_done_callback(make_task_guard(logger, session_id, "FAILURE_DIAGNOSTIC"))
                diagnostic.add_done_callback(
                    lambda completed: completed.exception() if not completed.cancelled() else None
                )

        finalizer = asyncio.create_task(
            handler._run_session_finalizer(session_id, completed_task, session_state),
            name=f"lifecycle-finalize-{session_id}",
        )
        handler._finalization_tasks.add(finalizer)
        finalizer.add_done_callback(handler._observe_finalization_task)

    task.add_done_callback(_cleanup)
