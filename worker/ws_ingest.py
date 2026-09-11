"""WebSocket audio ingestion endpoint (Fixes 1-15 auth, throttling, rate limits).

Shared state, limits and sibling helpers are read via `handler.*` at call
time so handler-level monkey-patching keeps working.
"""

import asyncio
import contextlib
import logging
import os
import time

import jwt as pyjwt
from aiohttp import web

import handler
from streaming.lifecycle.worker import LifecycleError, WorkerLocalLifecycleTransition
from streaming.orchestration.guards import reject_parked_session_config
from streaming.protocol.messages import (
    AudioReadyMessage,
    EndedMessage,
    ErrorMessage,
    StartedMessage,
    serialize_message,
)

FACEMODE_WS_AUTOSTART_MAX_TOKEN_AGE_S = int(
    os.environ.get("FACEMODE_WS_AUTOSTART_MAX_TOKEN_AGE_S", "300")
)

logger = logging.getLogger("handler")


# HTTP WebSocket ingestion endpoint: receive audio for a session.
# Auth is via signed JWT passed as Sec-WebSocket-Protocol: facemode.<jwt>.
# The JWT carries session config (replacing the old base64 cfg blob) and is
# verified with WORKER_AUTH_SECRET. If the session isn't registered locally,
# the JWT claims are used to self-sufficiently start the streaming task
# (resolves Modal session affinity where /sessions/start and /ws/{id} may
# hit different containers).
async def _safe_send_str(session_state: dict, websocket: web.WebSocketResponse, text: str) -> bool:
    if websocket.closed:
        return False
    lock = session_state.setdefault("send_lock", asyncio.Lock())
    try:
        async with lock:
            if websocket.closed:
                return False
            await websocket.send_str(text)
            return True
    except Exception as exc:
        logger.warning("WS_SEND_ERROR error=%s", exc.__class__.__name__)
        return False


async def _rollback_unowned_self_started_session(session_id: str, session_state: dict) -> None:
    # A self-started session whose post-claim gate fails must not be left
    # running ownerless, but a competing request for the same session may
    # already own it. Roll back only when nothing owns or is terminating it:
    # no committed websocket, no held connection slot, no reconnect grace and
    # no in-progress termination.
    if (
        session_state.get("websocket") is None
        and not session_state.get("connection_lost")
        and not session_state.get("ending")
        and not handler.ws_server.has_connection(session_id)
    ):
        await handler._rollback_self_started_session(session_id, session_state)


async def app_websocket_ingest(request):
    request_received_at = time.monotonic()
    session_id = request.match_info.get("session_id")
    client_ip = request.remote
    origin = request.headers.get("Origin", "")
    logger.info("WS_REQUEST_RECEIVED session=%s origin=%s ip=%s", session_id, origin, client_ip)

    # --- Fix 9: Origin header validation ---
    if handler.ALLOWED_WS_ORIGINS:
        if origin not in handler.ALLOWED_WS_ORIGINS and not request.query.get("token"):
            logger.warning("WS_ORIGIN_REJECTED session=%s origin=%s ip=%s", session_id, origin, client_ip)
            raise web.HTTPForbidden(text="Origin not allowed")

    # --- Fix 15: IP-based connection throttling ---
    if not handler._check_ip_limit(client_ip, handler.WS_MAX_CONNECTIONS_PER_IP):
        logger.warning("WS_IP_THROTTLED ip=%s session=%s", client_ip, session_id)
        raise web.HTTPTooManyRequests(text="Too many connections from this IP")

    # --- Fix 3: Extract JWT from Sec-WebSocket-Protocol ---
    protocols = request.headers.get("Sec-WebSocket-Protocol", "")
    auth_token = None
    from_subprotocol = False
    for proto in protocols.split(","):
        proto = proto.strip()
        if proto.startswith("facemode."):
            auth_token = proto[len("facemode."):]
            from_subprotocol = True
            break
    if not auth_token:
        # Clients that cannot set a WebSocket subprotocol may pass the signed
        # session JWT in the URL query string as a fallback.
        auth_token = request.query.get("token")

    if not auth_token:
        logger.warning("WS_AUTH_FAIL session=%s reason=missing_token ip=%s", session_id, client_ip)
        raise web.HTTPUnauthorized(text="Missing auth token")

    # --- Fix 1: Verify JWT ---
    if not handler.WORKER_AUTH_SECRET:
        raise web.HTTPInternalServerError(text="WORKER_AUTH_SECRET not configured")

    try:
        claims = pyjwt.decode(auth_token, handler.WORKER_AUTH_SECRET, algorithms=["HS256"])
    except pyjwt.ExpiredSignatureError:
        logger.warning("WS_AUTH_FAIL session=%s reason=expired_token ip=%s", session_id, client_ip)
        raise web.HTTPUnauthorized(text="Token expired")
    except pyjwt.InvalidTokenError:
        logger.warning(
            "WS_AUTH_FAIL session=%s reason=invalid_token ip=%s token_len=%d",
            session_id,
            client_ip,
            len(auth_token),
        )
        raise web.HTTPUnauthorized(text="Invalid token")

    # Verify session_id in JWT matches URL path
    if claims.get("sessionId") != session_id:
        logger.warning("WS_AUTH_FAIL session=%s reason=session_id_mismatch ip=%s", session_id, client_ip)
        raise web.HTTPUnauthorized(text="Session ID mismatch")

    # Check JTI availability first, but do not consume it until the backend owner
    # claim succeeds. A failed claim must not burn a caller's reconnect credential.
    if not handler._jti_is_available(claims):
        logger.warning("WS_AUTH_FAIL session=%s reason=token_already_used ip=%s", session_id, client_ip)
        raise web.HTTPUnauthorized(text="Token already used")

    session_state = handler.ws_server.active_sessions.get(session_id)
    lifecycle = handler._lifecycle_from_event(session_id, claims)
    claimed_lifecycle = False
    claim_result = None
    self_started = False

    if session_state is None and (claims.get("roomUrl") or claims.get("room")):
        is_reconnect = claims.get("reconnect") is True
        should_autostart = not is_reconnect

        # A reconnect may only create a fresh local worker after an owner lease has
        # expired and the backend explicitly accepted the fenced transfer.
        if is_reconnect and lifecycle is not None:
            try:
                claim_result = await lifecycle.claim(connection_token=auth_token)
                claimed_lifecycle = True
            except LifecycleError as exc:
                status = web.HTTPConflict if exc.status == 409 else web.HTTPUnauthorized
                logger.warning(
                    "WS_OWNER_CLAIM_REJECTED session=%s code=%s status=%s ip=%s",
                    session_id,
                    exc.code,
                    exc.status,
                    client_ip,
                )
                raise status(text="Session owner claim rejected") from exc
            should_autostart = bool(claim_result.get("transferred"))

        if not should_autostart:
            raise web.HTTPConflict(text="Session is not ready for reconnect")

        # Fresh WebSocket starts retain the short credential window. Reconnects
        # already passed a backend-fenced transfer claim above.
        if not is_reconnect:
            iat = claims.get("iat", 0)
            if time.time() - iat > FACEMODE_WS_AUTOSTART_MAX_TOKEN_AGE_S:
                logger.warning("WS_AUTH_FAIL session=%s reason=token_too_old_for_autostart ip=%s", session_id, client_ip)
                raise web.HTTPUnauthorized(text="Token too old for auto-start")

        ingestion_token = claims.get("ingestionToken", "")
        if not ingestion_token:
            raise web.HTTPBadRequest(text="ingestionToken missing from token claims")

        try:
            reject_parked_session_config(claims)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

        try:
            start_result = await handler._start_pod_session(
                claims,
                lifecycle=lifecycle,
                connection_token=auth_token,
                lifecycle_claimed=claimed_lifecycle,
            )
            self_started = start_result.get("status") == "STARTED"
        except web.HTTPException:
            raise
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

        session_state = handler.ws_server.active_sessions.get(session_id)

    if session_state is None:
        raise web.HTTPNotFound(text="Unknown session ID")

    claim_existing_reconnect = bool(
        claims.get("reconnect") is True
        and lifecycle is not None
        and not claimed_lifecycle
        and (session_state.get("connection_lost") or handler.ws_server.has_connection(session_id))
    )
    lifecycle_transition = session_state.setdefault(
        "lifecycle_transition",
        WorkerLocalLifecycleTransition(),
    )

    # Existing reconnects claim after WebSocket preparation, so a failed upgrade
    # cannot rotate the persisted owner generation or consume its one-time JTI.
    if not claim_existing_reconnect and not handler._consume_jti(claims):
        logger.warning("WS_AUTH_FAIL session=%s reason=token_already_used_after_claim ip=%s", session_id, client_ip)
        if self_started:
            await _rollback_unowned_self_started_session(session_id, session_state)
        raise web.HTTPUnauthorized(text="Token already used")

    if session_state.get("ending"):
        raise web.HTTPGone(text="Session is ending")

    # --- Fix 2: Require non-empty token (fail closed) ---
    provided_token = claims.get("ingestionToken", "")
    expected_token = session_state.get("token")
    if not expected_token or provided_token != expected_token:
        if provided_token and claims.get("reconnect") is True and not handler.ws_server.has_connection(session_id):
            session_state["token"] = provided_token
            logger.info("WS_RECONNECT_TOKEN_ROTATED session=%s ip=%s", session_id, client_ip)
        else:
            logger.warning("WS_AUTH_FAIL session=%s reason=invalid_ingestion_token ip=%s", session_id, client_ip)
            if self_started:
                await _rollback_unowned_self_started_session(session_id, session_state)
            raise web.HTTPUnauthorized(text="Unauthorized: Invalid Token")

    # --- Fix 5: Single connection per session ---
    if not handler.ws_server.acquire_connection(session_id):
        logger.warning("WS_DUP_CONNECTION session=%s ip=%s", session_id, client_ip)
        raise web.HTTPConflict(text="Session already has an active connection")

    # --- Fix 3: Echo subprotocol for canonical WebSocket clients ---
    websocket = web.WebSocketResponse(protocols=[f"facemode.{auth_token}"] if from_subprotocol else ())
    prepared = False
    try:
        await websocket.prepare(request)
        prepared = True
        if claim_existing_reconnect:
            try:
                await lifecycle.claim(connection_token=auth_token)
            except LifecycleError as exc:
                await websocket.close(code=4001, message=b"Session owner claim rejected")
                logger.warning(
                    "WS_OWNER_CLAIM_REJECTED session=%s code=%s status=%s ip=%s",
                    session_id,
                    exc.code,
                    exc.status,
                    client_ip,
                )
                raise web.HTTPConflict(text="Session owner claim rejected") from exc
            if not handler._consume_jti(claims):
                await websocket.close(code=4001, message=b"Token already used")
                raise web.HTTPUnauthorized(text="Token already used")
            previous_lifecycle = session_state.get("lifecycle")
            if previous_lifecycle is not None and previous_lifecycle is not lifecycle:
                await previous_lifecycle.pause_heartbeat()
            session_state["lifecycle"] = lifecycle
    except Exception:
        if self_started:
            await handler._rollback_self_started_session(session_id, session_state)
        raise
    finally:
        if not prepared:
            handler.ws_server.release_connection(session_id)

    input_adapter = session_state["input_adapter"]
    attach_websocket = getattr(input_adapter, "attach_websocket", None)
    if attach_websocket is not None:
        attach_websocket(websocket)
    rate_limiter = handler.AudioRateLimiter(handler.WS_AUDIO_RATE_PER_SEC, handler.WS_AUDIO_BURST, handler.WS_MAX_BYTES_PER_SEC)

    async def commit_connection() -> None:
        session_state.setdefault("last_activity_time", time.monotonic())
        session_state.setdefault("is_draining", False)
        session_state["connection_lost"] = False
        await handler._cancel_reconnect_expiry(session_state)
        session_state["websocket"] = websocket
        await handler._start_session_heartbeat(session_id, session_state)

    connection_epoch = await lifecycle_transition.commit_connection(commit_connection)
    if session_state.get("idle_watcher") is None or session_state["idle_watcher"].done():
        session_state["idle_watcher"] = asyncio.create_task(
            handler._watch_session_idle(session_id, session_state),
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
                    timeout=handler.WS_INACTIVITY_TIMEOUT
                )
            except RuntimeError as exc:
                if "WebSocket connection is closed" in str(exc):
                    logger.info("WS_CLOSED session=%s ip=%s", session_id, client_ip)
                    break
                raise
            except asyncio.TimeoutError:
                logger.info("WS_INACTIVITY_TIMEOUT session=%s ip=%s", session_id, client_ip)
                await websocket.close(code=1000, message=b"Inactivity timeout")
                break

            if message.type in (web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
                logger.info(
                    "WS_CLIENT_DISCONNECTED session=%s type=%s close_code=%s ip=%s",
                    session_id,
                    message.type,
                    websocket.close_code,
                    client_ip,
                )
                break
            if message.type not in (web.WSMsgType.BINARY, web.WSMsgType.TEXT):
                if message.type == web.WSMsgType.ERROR:
                    ws_error = websocket.exception()
                    logger.warning(
                        "WS_ERROR session=%s error=%s ip=%s",
                        session_id,
                        ws_error.__class__.__name__ if ws_error is not None else "Unknown",
                        client_ip,
                    )
                break

            payload_size = len(message.data) if isinstance(message.data, (bytes, bytearray, str)) else 0
            logger.info("WS_FRAME_RECEIVED session=%s type=%s size=%d ip=%s", session_id, message.type, payload_size, client_ip)

            if payload_size > handler.MAX_AUDIO_CHUNK_BYTES:
                logger.warning("WS_OVERSIZED session=%s size=%d max=%d ip=%s",
                               session_id, payload_size, handler.MAX_AUDIO_CHUNK_BYTES, client_ip)
                await handler._safe_send_str(session_state, websocket, serialize_message(ErrorMessage(
                    "INVALID_FORMAT", "audio message exceeds the maximum payload size", True
                )))
                await websocket.close(code=4000, message=b"Invalid protocol payload")
                break

            allowed, reason = rate_limiter.allow(payload_size)
            if not allowed:
                logger.warning("WS_RATE_LIMIT session=%s reason=%s ip=%s", session_id, reason, client_ip)
                await handler._safe_send_str(session_state, websocket, serialize_message(ErrorMessage("RATE_LIMITED", reason, True)))
                await websocket.close(code=4006, message=b"Rate limited")
                break

            try:
                responses = await input_adapter.handle_message(message.data)
            except Exception as exc:
                logger.error(
                    "WS_MESSAGE_HANDLER_ERROR session=%s error=%s ip=%s",
                    session_id,
                    exc.__class__.__name__,
                    client_ip,
                    exc_info=exc,
                )
                error_msg = serialize_message(ErrorMessage(
                    "INTERNAL_ERROR",
                    "WebSocket message handling failed",
                    fatal=True,
                ))
                await handler._safe_send_str(session_state, websocket, error_msg)
                await websocket.close(code=4000, message=b"Internal handler error")
                break

            session_state["last_activity_time"] = time.monotonic()
            fatal_response = False
            send_failed = False
            for response in responses:
                response_type = getattr(response, "type", "unknown")
                sent = await handler._safe_send_str(session_state, websocket, serialize_message(response))
                if not sent:
                    logger.warning(
                        "WS_SEND_RESPONSE_FAILED session=%s type=%s closed=%s close_code=%s ip=%s",
                        session_id,
                        response_type,
                        websocket.closed,
                        websocket.close_code,
                        client_ip,
                    )
                    send_failed = True
                    break
                logger.info(
                    "WS_SEND_RESPONSE session=%s type=%s ip=%s",
                    session_id,
                    response_type,
                    client_ip,
                )
                if isinstance(response, StartedMessage) and session_state.get("audio_ready"):
                    if not await handler._safe_send_str(
                        session_state, websocket, serialize_message(AudioReadyMessage(0))
                    ):
                        send_failed = True
                        break
                if isinstance(response, ErrorMessage) and response.fatal:
                    fatal_response = True
                if isinstance(response, EndedMessage):
                    session_state["ending"] = True
                    logger.info("SESSION_END_RECEIVED session=%s", session_id)
                    break
            if send_failed:
                break
            if fatal_response:
                close_code = {
                    "SESSION_NOT_FOUND": 4004,
                    "NOT_NEGOTIATED": 4005,
                    "RATE_LIMITED": 4006,
                }.get(next((response.code for response in responses if isinstance(response, ErrorMessage)), ""), 4000)
                await websocket.close(code=close_code, message=b"Fatal protocol error")
                break
            if any(isinstance(response, EndedMessage) for response in responses):
                break
    except asyncio.CancelledError:
        logger.warning(
            "WS_HANDLER_CANCELLED session=%s closed=%s close_code=%s ip=%s",
            session_id,
            websocket.closed,
            websocket.close_code,
            client_ip,
        )
        raise
    except Exception as exc:
        logger.error(
            "WS_HANDLER_ERROR session=%s error=%s closed=%s close_code=%s ip=%s",
            session_id,
            exc.__class__.__name__,
            websocket.closed,
            websocket.close_code,
            client_ip,
            exc_info=exc,
        )
        raise
    finally:
        session_task = handler._active_sessions.get(session_id)
        with contextlib.suppress(Exception):
            await input_adapter.disconnect()
        handler.ws_server.release_connection(session_id)

        async def clear_current_connection() -> None:
            if session_state.get("websocket") is websocket:
                session_state["websocket"] = None

        is_current_connection = await lifecycle_transition.apply_if_current(
            connection_epoch,
            clear_current_connection,
        )

        if session_state.get("ending"):
            if session_task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await session_task
            logger.info("WS_SESSION_ENDED session=%s ip=%s", session_id, client_ip)
        elif is_current_connection:
            # Keep the pipeline alive while the relay performs its reconnect attempts,
            # but pause heartbeat renewal so the persisted lease enforces the grace
            # window and another container can only claim it after expiry.
            lifecycle = session_state.get("lifecycle")
            disconnect_result = {}
            if lifecycle is not None:
                try:
                    disconnect_result = await lifecycle.send_event("transport_disconnected")
                except LifecycleError as exc:
                    logger.warning(
                        "WORKER_DISCONNECT_LIFECYCLE_FAILED session=%s code=%s status=%s",
                        session_id,
                        exc.code,
                        exc.status,
                    )

            async def commit_disconnect() -> None:
                session_state["connection_lost"] = True
                handler._schedule_reconnect_expiry(
                    session_id,
                    session_state,
                    disconnect_result.get("expiresAt"),
                )
                if lifecycle is not None:
                    await lifecycle.pause_heartbeat()

            disconnect_committed = await lifecycle_transition.apply_if_current(
                connection_epoch,
                commit_disconnect,
            )
            if disconnect_committed:
                ws_exception = websocket.exception()
                logger.info(
                    "WS_CONNECTION_LOST session=%s closed=%s close_code=%s exception=%s ip=%s",
                    session_id,
                    websocket.closed,
                    websocket.close_code,
                    ws_exception.__class__.__name__ if ws_exception is not None else "None",
                    client_ip,
                )

    return websocket
