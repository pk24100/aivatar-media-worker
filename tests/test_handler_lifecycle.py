import asyncio
import gc
import importlib.util
import os
import sys
import time
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WORKER_AUTH_SECRET", "test-worker-secret-with-at-least-32-bytes")
os.environ.setdefault("BACKEND_INTERNAL_URL", "https://backend.example.test")
os.environ["AIVATAR_BATCHED_INFERENCE"] = "0"


class _ImportModelPool:
    def __init__(self, *args, **kwargs):
        self.available = 1

    def get_available_count(self):
        return self.available


class _ImportWebSocketServer:
    is_running = True
    active_sessions = {}

    async def start(self):
        self.is_running = True


async def _unused_streaming_session(**_kwargs):
    return None


def _load_handler_module():
    stream_processor = types.ModuleType("streaming.orchestration.stream_processor")
    stream_processor.run_streaming_session = _unused_streaming_session

    websocket_server = types.ModuleType("streaming.transport.websocket_server")
    websocket_server.ws_server = _ImportWebSocketServer()

    model_pool = types.ModuleType("utils.model_pool")
    model_pool.FlashHeadModelPool = _ImportModelPool

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = lambda **_kwargs: ""

    spec = importlib.util.spec_from_file_location(
        "handler_lifecycle_under_test",
        ROOT / "handler.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            # Register the exec'd module as "handler" so `import handler`
            # inside the worker/ package binds this same module instance and
            # handler.* monkey-patching keeps working.
            "handler": module,
            "streaming.orchestration.stream_processor": stream_processor,
            "streaming.transport.websocket_server": websocket_server,
            "utils.model_pool": model_pool,
            "huggingface_hub": huggingface_hub,
        },
    ):
        spec.loader.exec_module(module)
    return module


handler = _load_handler_module()


class _FakeInputAdapter:
    def __init__(self):
        self.disconnect = AsyncMock()

    async def handle_message(self, _data):
        return []


class _FakeWebSocketServer:
    def __init__(self, order=None):
        self.is_running = True
        self.active_sessions = {}
        self.order = order if order is not None else []
        self.register_calls = 0
        self.acquire_calls = 0
        self.release_calls = 0
        self.register_kwargs = None

    async def start(self):
        self.is_running = True

    def register_session(self, session_id, token, **_kwargs):
        self.order.append("register")
        self.register_calls += 1
        self.register_kwargs = _kwargs
        state = {
            "token": token,
            "input_adapter": _FakeInputAdapter(),
            "queue": asyncio.Queue(),
        }
        self.active_sessions[session_id] = state
        return state["queue"]

    def unregister_session(self, session_id):
        self.active_sessions.pop(session_id, None)

    def acquire_connection(self, _session_id):
        self.acquire_calls += 1
        return True

    def release_connection(self, _session_id):
        self.release_calls += 1

    def has_connection(self, _session_id):
        return False


class _FakeWebSocket:
    def __init__(self):
        self.closed = False
        self.close_code = 1000

    async def prepare(self, _request):
        return None

    async def receive(self):
        return SimpleNamespace(
            type=handler.web.WSMsgType.CLOSING,
            data=1000,
            extra="",
        )

    async def send_str(self, _text):
        return None

    async def send_json(self, _payload):
        return None

    async def close(self, code=1000, message=b""):
        self.closed = True
        self.close_code = code

    def exception(self):
        return None


class _FakeRequest:
    def __init__(self, session_id, token, *, prefix="facemode.", query_token=None):
        self.match_info = {"session_id": session_id}
        self.headers = (
            {"Sec-WebSocket-Protocol": f"{prefix}{token}"}
            if prefix is not None
            else {}
        )
        self.query = {"token": query_token} if query_token is not None else {}
        self.remote = "127.0.0.1"


class _FakeLifecycle:
    def __init__(self, claim_result=None, claim_error=None):
        self.is_configured = True
        self.claim_result = claim_result or {}
        self.claim_error = claim_error
        self.claim = AsyncMock(side_effect=self._claim)
        self.send_event = AsyncMock(return_value={})
        self.stop = AsyncMock()
        self.send_terminal = AsyncMock()
        self.pause_heartbeat = AsyncMock()
        self.heartbeat_callback = None

    async def _claim(self, **_kwargs):
        if self.claim_error is not None:
            raise self.claim_error
        return self.claim_result

    def start_heartbeat(self, callback):
        self.heartbeat_callback = callback


class HandlerOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        handler._used_jti.clear()
        handler._active_sessions.clear()
        handler._finalization_tasks.clear()
        handler._ip_connections.clear()
        handler._ws_started = True
        self.ws_server = _FakeWebSocketServer()
        handler.ws_server = self.ws_server
        handler.model_pool = _ImportModelPool()
        handler.WORKER_AUTH_SECRET = os.environ["WORKER_AUTH_SECRET"]

    async def asyncTearDown(self):
        expiry_tasks = []
        for state in self.ws_server.active_sessions.values():
            expiry_task = state.get("reconnect_expiry_task")
            if expiry_task is not None and not expiry_task.done():
                expiry_task.cancel()
                expiry_tasks.append(expiry_task)
        if expiry_tasks:
            await asyncio.gather(*expiry_tasks, return_exceptions=True)

    def _token(
        self,
        session_id,
        *,
        jti,
        reconnect=True,
        issued_at=None,
        include_authority=False,
    ):
        now = int(time.time())
        claims = {
            "sessionId": session_id,
            "roomUrl": "wss://livekit.example.test",
            "roomName": f"room-{session_id}",
            "roomToken": "runtime-room-token",
            "sourceImage": "https://assets.example.test/avatar.png",
            "ingestionToken": "runtime-ingestion-token",
            "workerLeaseId": "lease-123",
            "workerGeneration": 2,
            "reconnect": reconnect,
            "jti": jti,
            "iat": now if issued_at is None else issued_at,
            "exp": now + 900,
        }
        if include_authority:
            claims["workerStartAuthority"] = "websocket"
        return handler.pyjwt.encode(
            claims,
            handler.WORKER_AUTH_SECRET,
            algorithm="HS256",
        )

    async def _connect_fresh(self, session_id, token):
        lifecycle = _FakeLifecycle()

        async def start_session(event, **_kwargs):
            self.ws_server.active_sessions[session_id] = {
                "token": event["ingestionToken"],
                "input_adapter": _FakeInputAdapter(),
            }
            return {"status": "STARTED"}

        websocket = _FakeWebSocket()
        with (
            patch.object(handler, "_lifecycle_from_event", return_value=lifecycle),
            patch.object(
                handler,
                "_start_pod_session",
                new=AsyncMock(side_effect=start_session),
            ) as start,
            patch.object(handler, "_start_session_heartbeat", new=AsyncMock()),
            patch.object(handler.web, "WebSocketResponse", return_value=websocket) as response,
        ):
            result = await handler.app_websocket_ingest(_FakeRequest(session_id, token))
        return result, start, response

    async def test_accepts_and_echoes_only_facemode_subprotocol(self):
        session_id = "session-facemode-prefix"
        token = self._token(session_id, jti="jti-facemode-prefix", reconnect=False)

        result, _start, response = await self._connect_fresh(session_id, token)

        self.assertIsInstance(result, _FakeWebSocket)
        response.assert_called_once_with(protocols=[f"facemode.{token}"])

    async def test_rejects_aivatar_subprotocol(self):
        session_id = "session-old-prefix"
        token = self._token(session_id, jti="jti-old-prefix", reconnect=False)

        with self.assertRaises(handler.web.HTTPUnauthorized) as raised:
            await handler.app_websocket_ingest(
                _FakeRequest(session_id, token, prefix="aivatar.")
            )

        self.assertEqual(raised.exception.text, "Missing auth token")

    async def test_rejects_malformed_facemode_token_without_echoing_it(self):
        malformed_token = "private-malformed-token"

        with self.assertRaises(handler.web.HTTPUnauthorized) as raised:
            await handler.app_websocket_ingest(
                _FakeRequest("session-malformed-token", malformed_token)
            )

        self.assertEqual(raised.exception.text, "Invalid token")
        self.assertNotIn(malformed_token, raised.exception.text)

    async def test_query_token_fallback_remains_supported(self):
        session_id = "session-query-token"
        token = self._token(session_id, jti="jti-query-token", reconnect=False)
        request = _FakeRequest(session_id, token, prefix=None, query_token=token)
        lifecycle = _FakeLifecycle()

        async def start_session(event, **_kwargs):
            self.ws_server.active_sessions[session_id] = {
                "token": event["ingestionToken"],
                "input_adapter": _FakeInputAdapter(),
            }
            return {"status": "STARTED"}

        with (
            patch.object(handler, "_lifecycle_from_event", return_value=lifecycle),
            patch.object(handler, "_start_pod_session", new=AsyncMock(side_effect=start_session)),
            patch.object(handler, "_start_session_heartbeat", new=AsyncMock()),
            patch.object(handler.web, "WebSocketResponse", return_value=_FakeWebSocket()) as response,
        ):
            await handler.app_websocket_ingest(request)

        response.assert_called_once_with(protocols=())

    async def test_fresh_connect_without_authority_field_skips_registration_wait(self):
        session_id = "session-no-authority"
        token = self._token(session_id, jti="jti-no-authority", reconnect=False)

        with patch.object(
            handler.asyncio,
            "sleep",
            new=AsyncMock(side_effect=AssertionError("control registration wait used")),
        ):
            result, start, _response = await self._connect_fresh(session_id, token)

        self.assertIsInstance(result, _FakeWebSocket)
        start.assert_awaited_once()

    async def test_fresh_autostart_accepts_ages_through_300_seconds(self):
        now = int(time.time())
        for age in (240, 300):
            with self.subTest(age=age):
                session_id = f"session-age-{age}"
                token = self._token(
                    session_id,
                    jti=f"jti-age-{age}",
                    reconnect=False,
                    issued_at=now - age,
                )
                with patch.object(handler.time, "time", return_value=now):
                    result, _start, _response = await self._connect_fresh(session_id, token)
                self.assertIsInstance(result, _FakeWebSocket)

    async def test_fresh_autostart_rejects_age_301_seconds(self):
        now = int(time.time())
        session_id = "session-age-301"
        token = self._token(
            session_id,
            jti="jti-age-301",
            reconnect=False,
            issued_at=now - 301,
        )

        with (
            patch.object(handler.time, "time", return_value=now),
            patch.object(handler, "_lifecycle_from_event", return_value=_FakeLifecycle()),
            patch.object(handler, "_start_pod_session", new=AsyncMock()) as start,
        ):
            with self.assertRaises(handler.web.HTTPUnauthorized) as raised:
                await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertEqual(raised.exception.text, "Token too old for auto-start")
        start.assert_not_awaited()

    async def test_claim_occurs_before_local_registration(self):
        order = []
        self.ws_server.order = order
        lifecycle = _FakeLifecycle()

        async def claim(**_kwargs):
            order.append("claim")
            return {}

        lifecycle.claim = AsyncMock(side_effect=claim)
        event = {
            "sessionId": "session-123",
            "ingestionToken": "runtime-ingestion-token",
            "inputProvider": "custom",
        }

        with (
            patch.object(handler, "_execute_streaming_event", new=AsyncMock(return_value={})),
            patch.object(handler, "_track_session_task") as track_task,
            patch.object(handler, "_start_session_heartbeat", new=AsyncMock()),
        ):
            result = await handler._start_pod_session(event, lifecycle=lifecycle)

        session_task = track_task.call_args.args[1]
        await session_task
        self.assertEqual(result["status"], "STARTED")
        self.assertEqual(order, ["claim", "register"])
        self.assertEqual(self.ws_server.register_kwargs["input_provider"], "custom")

    async def test_rejected_claim_does_not_consume_jti(self):
        session_id = "session-claim-rejected"
        jti = "jti-claim-rejected"
        token = self._token(session_id, jti=jti)
        lifecycle = _FakeLifecycle(
            claim_error=handler.LifecycleError(
                "upstream text must stay private",
                status=409,
                code="OWNER_LEASE_ACTIVE",
            )
        )

        with (
            patch.object(handler, "_lifecycle_from_event", return_value=lifecycle),
            patch.object(handler, "_start_pod_session", new=AsyncMock()) as start_session,
        ):
            with self.assertRaises(handler.web.HTTPConflict) as raised:
                await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertEqual(raised.exception.text, "Session owner claim rejected")
        self.assertNotIn("upstream text must stay private", raised.exception.text)
        lifecycle.claim.assert_awaited_once_with(connection_token=token)
        start_session.assert_not_awaited()
        self.assertNotIn(jti, handler._used_jti)
        self.assertTrue(handler._jti_is_available({"jti": jti, "exp": time.time() + 60}))

    async def test_failed_jti_consume_after_self_start_rolls_back_orphan(self):
        session_id = "session-self-start-consume-failure"
        jti = "jti-self-start-consume-failure"
        token = self._token(session_id, jti=jti, reconnect=False)
        lifecycle = _FakeLifecycle()
        processor_started = asyncio.Event()
        processor_cleaned = asyncio.Event()
        tracked = {}

        async def processor(_event):
            processor_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                processor_cleaned.set()

        real_track = handler._track_session_task

        def track_and_record(current_session_id, task, state):
            tracked["task"] = task
            return real_track(current_session_id, task, state)

        async def heartbeat_after_processor_started(_session_id, _state):
            await processor_started.wait()

        with (
            patch.object(handler, "_lifecycle_from_event", return_value=lifecycle),
            patch.object(handler, "_execute_streaming_event", side_effect=processor),
            patch.object(handler, "_track_session_task", side_effect=track_and_record),
            patch.object(
                handler,
                "_start_session_heartbeat",
                new=AsyncMock(side_effect=heartbeat_after_processor_started),
            ),
            patch.object(handler, "_consume_jti", return_value=False),
        ):
            with self.assertRaises(handler.web.HTTPUnauthorized) as raised:
                await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertEqual(raised.exception.text, "Token already used")
        task = tracked["task"]
        try:
            self.assertTrue(processor_started.is_set())
            self.assertTrue(processor_cleaned.is_set())
            self.assertNotIn(session_id, handler._active_sessions)
            self.assertNotIn(session_id, self.ws_server.active_sessions)
            lifecycle.claim.assert_awaited_once_with(connection_token=token)
            lifecycle.stop.assert_awaited_once_with(release=True)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await handler.drain_finalization_tasks(timeout=1)
            self.ws_server.unregister_session(session_id)

    async def test_failed_jti_consume_after_self_start_keeps_owned_session(self):
        session_id = "session-self-start-consume-owned"
        token = self._token(session_id, jti="jti-self-start-consume-owned", reconnect=False)
        lifecycle = _FakeLifecycle()
        processor_release = asyncio.Event()
        processor_cleaned = asyncio.Event()
        tracked = {}

        async def processor(_event):
            try:
                await processor_release.wait()
            finally:
                processor_cleaned.set()

        real_track = handler._track_session_task

        def track_and_record(current_session_id, task, state):
            tracked["task"] = task
            return real_track(current_session_id, task, state)

        self.ws_server.has_connection = lambda _session_id: True
        with (
            patch.object(handler, "_lifecycle_from_event", return_value=lifecycle),
            patch.object(handler, "_execute_streaming_event", side_effect=processor),
            patch.object(handler, "_track_session_task", side_effect=track_and_record),
            patch.object(handler, "_start_session_heartbeat", new=AsyncMock()),
            patch.object(handler, "_consume_jti", return_value=False),
        ):
            with self.assertRaises(handler.web.HTTPUnauthorized) as raised:
                await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertEqual(raised.exception.text, "Token already used")
        task = tracked["task"]
        self.assertIn(session_id, self.ws_server.active_sessions)
        self.assertIn(session_id, handler._active_sessions)
        self.assertFalse(task.done())
        lifecycle.stop.assert_not_awaited()

        processor_release.set()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(processor_cleaned.is_set())
        self.assertTrue(await handler.drain_finalization_tasks(timeout=1))
        self.ws_server.unregister_session(session_id)

    async def test_ingestion_token_mismatch_after_self_start_rolls_back_orphan(self):
        session_id = "session-self-start-token-mismatch"
        jti = "jti-self-start-token-mismatch"
        token = self._token(session_id, jti=jti, reconnect=False)
        lifecycle = _FakeLifecycle()
        processor_started = asyncio.Event()
        processor_cleaned = asyncio.Event()
        tracked = {}

        async def processor(_event):
            processor_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                processor_cleaned.set()

        real_track = handler._track_session_task

        def track_and_record(current_session_id, task, state):
            tracked["task"] = task
            return real_track(current_session_id, task, state)

        async def heartbeat_after_processor_started(_session_id, _state):
            await processor_started.wait()

        real_start = handler._start_pod_session

        async def start_then_rotate_token(*args, **kwargs):
            result = await real_start(*args, **kwargs)
            self.ws_server.active_sessions[session_id]["token"] = "rotated-ingestion-token"
            return result

        with (
            patch.object(handler, "_lifecycle_from_event", return_value=lifecycle),
            patch.object(handler, "_execute_streaming_event", side_effect=processor),
            patch.object(handler, "_track_session_task", side_effect=track_and_record),
            patch.object(
                handler,
                "_start_session_heartbeat",
                new=AsyncMock(side_effect=heartbeat_after_processor_started),
            ),
            patch.object(
                handler,
                "_start_pod_session",
                new=AsyncMock(side_effect=start_then_rotate_token),
            ),
        ):
            with self.assertRaises(handler.web.HTTPUnauthorized) as raised:
                await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertEqual(raised.exception.text, "Unauthorized: Invalid Token")
        self.assertIn(jti, handler._used_jti)
        task = tracked["task"]
        try:
            self.assertTrue(processor_started.is_set())
            self.assertTrue(processor_cleaned.is_set())
            self.assertNotIn(session_id, handler._active_sessions)
            self.assertNotIn(session_id, self.ws_server.active_sessions)
            lifecycle.stop.assert_awaited_once_with(release=True)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await handler.drain_finalization_tasks(timeout=1)
            self.ws_server.unregister_session(session_id)

    async def test_duplicate_control_start_does_not_register_twice(self):
        session_id = "session-existing"
        handler._active_sessions[session_id] = object()

        result = await handler._start_pod_session({"sessionId": session_id})

        self.assertEqual(result["status"], "ALREADY_STARTED")
        self.assertEqual(self.ws_server.register_calls, 0)

    async def test_existing_local_reconnect_reuses_registration(self):
        session_id = "session-local-reconnect"
        jti = "jti-local-reconnect"
        token = self._token(session_id, jti=jti)
        previous_lifecycle = _FakeLifecycle()
        reconnect_lifecycle = _FakeLifecycle(
            claim_result={"leaseId": "lease-123", "generation": 3, "resumed": True}
        )
        self.ws_server.active_sessions[session_id] = {
            "token": "runtime-ingestion-token",
            "input_adapter": _FakeInputAdapter(),
            "lifecycle": previous_lifecycle,
            "connection_lost": True,
        }
        websocket = _FakeWebSocket()

        with (
            patch.object(handler, "_start_pod_session", new=AsyncMock()) as start_session,
            patch.object(handler, "_start_session_heartbeat", new=AsyncMock()),
            patch.object(handler, "_lifecycle_from_event", return_value=reconnect_lifecycle),
            patch.object(handler.web, "WebSocketResponse", return_value=websocket),
        ):
            result = await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertIs(result, websocket)
        start_session.assert_not_awaited()
        self.assertEqual(self.ws_server.register_calls, 0)
        self.assertEqual(self.ws_server.acquire_calls, 1)
        self.assertIn(jti, handler._used_jti)
        reconnect_lifecycle.claim.assert_awaited_once_with(connection_token=token)
        previous_lifecycle.pause_heartbeat.assert_awaited_once()
        self.assertIs(
            self.ws_server.active_sessions[session_id]["lifecycle"],
            reconnect_lifecycle,
        )

    async def test_disconnect_expiry_requests_owner_termination_while_disconnected(self):
        session_id = "session-disconnect-expiry"
        token = self._token(session_id, jti="jti-disconnect-expiry")
        lifecycle = _FakeLifecycle()
        expires_at = datetime.now(timezone.utc) + timedelta(milliseconds=30)
        lifecycle.send_event.side_effect = [
            {},
            {"expiresAt": expires_at.isoformat().replace("+00:00", "Z")},
        ]
        state = {
            "token": "runtime-ingestion-token",
            "input_adapter": _FakeInputAdapter(),
            "lifecycle": lifecycle,
        }
        self.ws_server.active_sessions[session_id] = state
        termination_requested = asyncio.Event()

        async def request_termination(*_args):
            termination_requested.set()

        with (
            patch.object(handler.web, "WebSocketResponse", return_value=_FakeWebSocket()),
            patch.object(
                handler,
                "_request_owner_termination",
                new=AsyncMock(side_effect=request_termination),
            ) as terminate,
        ):
            await handler.app_websocket_ingest(_FakeRequest(session_id, token))
            await asyncio.wait_for(termination_requested.wait(), timeout=1)

        self.assertTrue(state["connection_lost"])
        terminate.assert_awaited_once_with(
            session_id,
            state,
            {"reason": "reconnect_grace_expired"},
        )

    async def test_prepared_reconnect_cancels_stale_expiry_and_later_disconnect_reschedules(self):
        session_id = "session-reconnect-expiry"
        first_token = self._token(session_id, jti="jti-reconnect-expiry-first")
        second_token = self._token(session_id, jti="jti-reconnect-expiry-second")
        lifecycle = _FakeLifecycle()
        disconnect_count = 0
        expiry_started = [asyncio.Event(), asyncio.Event()]
        expiry_releases = [asyncio.Event(), asyncio.Event()]
        expiry_count = 0

        async def send_event(event):
            nonlocal disconnect_count
            if event == "worker_started":
                return {}
            disconnect_count += 1
            return {"expiresAt": f"2030-01-01T00:00:0{disconnect_count}.000Z"}

        async def terminate_after_expiry(current_session_id, current_state, _expires_at):
            nonlocal expiry_count
            expiry_index = expiry_count
            expiry_count += 1
            expiry_started[expiry_index].set()
            await expiry_releases[expiry_index].wait()
            if (
                current_state.get("reconnect_expiry_task") is asyncio.current_task()
                and current_state.get("connection_lost")
            ):
                await handler._request_owner_termination(
                    current_session_id,
                    current_state,
                    {"reason": "reconnect_grace_expired"},
                )

        lifecycle.send_event.side_effect = send_event
        state = {
            "token": "runtime-ingestion-token",
            "input_adapter": _FakeInputAdapter(),
            "lifecycle": lifecycle,
        }
        self.ws_server.active_sessions[session_id] = state
        first_websocket = _FakeWebSocket()
        reconnect_prepared = asyncio.Event()
        reconnect_active = asyncio.Event()
        reconnect_closed = asyncio.Event()

        class ReconnectWebSocket(_FakeWebSocket):
            async def prepare(self, request):
                await super().prepare(request)
                reconnect_prepared.set()

            async def receive(self):
                reconnect_active.set()
                await reconnect_closed.wait()
                return await super().receive()

        second_websocket = ReconnectWebSocket()
        termination_requested = asyncio.Event()

        async def request_termination(*_args):
            termination_requested.set()

        with (
            patch.object(
                handler.web,
                "WebSocketResponse",
                side_effect=[first_websocket, second_websocket],
            ),
            patch.object(handler, "_lifecycle_from_event", return_value=lifecycle),
            patch.object(
                handler,
                "_request_owner_termination",
                new=AsyncMock(side_effect=request_termination),
            ) as terminate,
            patch.object(
                handler,
                "_terminate_after_reconnect_expiry",
                side_effect=terminate_after_expiry,
            ),
        ):
            await handler.app_websocket_ingest(_FakeRequest(session_id, first_token))
            first_expiry_task = state["reconnect_expiry_task"]
            await asyncio.wait_for(expiry_started[0].wait(), timeout=1)
            reconnect = asyncio.create_task(
                handler.app_websocket_ingest(_FakeRequest(session_id, second_token))
            )
            await asyncio.wait_for(reconnect_prepared.wait(), timeout=1)
            await asyncio.wait_for(reconnect_active.wait(), timeout=1)

            self.assertTrue(first_expiry_task.cancelled())
            self.assertFalse(state["connection_lost"])
            terminate.assert_not_awaited()

            reconnect_closed.set()
            await reconnect
            second_expiry_task = state["reconnect_expiry_task"]
            self.assertIsNot(second_expiry_task, first_expiry_task)
            await asyncio.wait_for(expiry_started[1].wait(), timeout=1)
            expiry_releases[1].set()
            await asyncio.wait_for(termination_requested.wait(), timeout=1)

        terminate.assert_awaited_once_with(
            session_id,
            state,
            {"reason": "reconnect_grace_expired"},
        )

    async def test_stale_disconnect_cannot_pause_prepared_reconnect(self):
        session_id = "session-overlapping-reconnect"
        first_token = self._token(session_id, jti="jti-overlap-first")
        second_token = self._token(session_id, jti="jti-overlap-second")
        disconnect_started = asyncio.Event()
        release_disconnect = asyncio.Event()
        reconnect_prepared = asyncio.Event()
        close_reconnect = asyncio.Event()
        lifecycle = _FakeLifecycle()

        async def send_event(event):
            if event == "transport_disconnected":
                disconnect_started.set()
                await release_disconnect.wait()
                return {"expiresAt": "2030-01-01T00:00:00.000Z"}
            return {}

        lifecycle.send_event.side_effect = send_event
        state = {
            "token": "runtime-ingestion-token",
            "input_adapter": _FakeInputAdapter(),
            "lifecycle": lifecycle,
            "lifecycle_started": True,
        }
        self.ws_server.active_sessions[session_id] = state

        class ReconnectWebSocket(_FakeWebSocket):
            async def prepare(self, request):
                await super().prepare(request)
                reconnect_prepared.set()

            async def receive(self):
                await close_reconnect.wait()
                return await super().receive()

        with patch.object(
            handler.web,
            "WebSocketResponse",
            side_effect=[_FakeWebSocket(), ReconnectWebSocket()],
        ), patch.object(handler, "_lifecycle_from_event", return_value=lifecycle):
            old_connection = asyncio.create_task(
                handler.app_websocket_ingest(_FakeRequest(session_id, first_token))
            )
            await asyncio.wait_for(disconnect_started.wait(), timeout=1)
            reconnect = asyncio.create_task(
                handler.app_websocket_ingest(_FakeRequest(session_id, second_token))
            )
            await asyncio.wait_for(reconnect_prepared.wait(), timeout=1)
            release_disconnect.set()
            await old_connection

            self.assertFalse(state["connection_lost"])
            self.assertNotIn("reconnect_expiry_task", state)
            lifecycle.pause_heartbeat.assert_not_awaited()

            state["ending"] = True
            close_reconnect.set()
            await reconnect

    async def test_latest_disconnect_schedules_exact_backend_expiry(self):
        session_id = "session-current-disconnect"
        token = self._token(session_id, jti="jti-current-disconnect")
        expires_at = "2030-05-06T07:08:09.123Z"
        lifecycle = _FakeLifecycle()
        lifecycle.send_event.side_effect = lambda event: (
            {"expiresAt": expires_at} if event == "transport_disconnected" else {}
        )
        state = {
            "token": "runtime-ingestion-token",
            "input_adapter": _FakeInputAdapter(),
            "lifecycle": lifecycle,
            "lifecycle_started": True,
        }
        self.ws_server.active_sessions[session_id] = state

        with (
            patch.object(handler.web, "WebSocketResponse", return_value=_FakeWebSocket()),
            patch.object(handler, "_schedule_reconnect_expiry") as schedule_expiry,
        ):
            await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        schedule_expiry.assert_called_once_with(session_id, state, expires_at)
        lifecycle.pause_heartbeat.assert_awaited_once()
        self.assertTrue(state["connection_lost"])

    async def test_websocket_prepare_failure_releases_acquired_slot_once(self):
        session_id = "session-prepare-failure"
        token = self._token(session_id, jti="jti-prepare-failure")
        self.ws_server.active_sessions[session_id] = {
            "token": "runtime-ingestion-token",
            "input_adapter": _FakeInputAdapter(),
        }
        websocket = _FakeWebSocket()
        websocket.prepare = AsyncMock(side_effect=RuntimeError("prepare failed"))

        with patch.object(handler.web, "WebSocketResponse", return_value=websocket):
            with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertEqual(self.ws_server.acquire_calls, 1)
        self.assertEqual(self.ws_server.release_calls, 1)

    async def test_self_started_prepare_failure_rolls_back_all_local_ownership(self):
        session_id = "session-self-start-prepare-failure"
        token = self._token(
            session_id,
            jti="jti-self-start-prepare-failure",
            reconnect=False,
        )
        lifecycle = _FakeLifecycle()
        processor_started = asyncio.Event()
        processor_cleaned = asyncio.Event()

        async def processor(_event):
            processor_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                processor_cleaned.set()

        websocket = _FakeWebSocket()

        async def fail_prepare(_request):
            await processor_started.wait()
            raise RuntimeError("prepare failed")

        websocket.prepare = AsyncMock(side_effect=fail_prepare)
        with (
            patch.object(handler, "_lifecycle_from_event", return_value=lifecycle),
            patch.object(handler, "_execute_streaming_event", side_effect=processor),
            patch.object(handler.web, "WebSocketResponse", return_value=websocket),
        ):
            with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertTrue(processor_started.is_set())
        self.assertTrue(processor_cleaned.is_set())
        self.assertNotIn(session_id, handler._active_sessions)
        self.assertNotIn(session_id, self.ws_server.active_sessions)
        lifecycle.stop.assert_awaited_once_with(release=True)
        self.assertEqual(self.ws_server.acquire_calls, 1)
        self.assertEqual(self.ws_server.release_calls, 1)

    async def test_control_started_prepare_failure_preserves_existing_session(self):
        session_id = "session-control-start-prepare-failure"
        token = self._token(session_id, jti="jti-control-prepare-failure")
        lifecycle = _FakeLifecycle()
        processor = asyncio.create_task(asyncio.Event().wait())
        state = {
            "token": "runtime-ingestion-token",
            "input_adapter": _FakeInputAdapter(),
            "lifecycle": lifecycle,
        }
        self.ws_server.active_sessions[session_id] = state
        handler._active_sessions[session_id] = processor
        websocket = _FakeWebSocket()
        websocket.prepare = AsyncMock(side_effect=RuntimeError("prepare failed"))

        with patch.object(handler.web, "WebSocketResponse", return_value=websocket):
            with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertIs(handler._active_sessions[session_id], processor)
        self.assertIs(self.ws_server.active_sessions[session_id], state)
        lifecycle.stop.assert_not_awaited()
        processor.cancel()
        await asyncio.gather(processor, return_exceptions=True)

    async def test_fresh_owner_transfer_starts_with_claimed_fence(self):
        session_id = "session-transferred"
        jti = "jti-transferred"
        token = self._token(session_id, jti=jti)
        lifecycle = _FakeLifecycle(claim_result={"transferred": True})
        websocket = _FakeWebSocket()

        async def start_session(event, **_kwargs):
            self.ws_server.active_sessions[session_id] = {
                "token": event["ingestionToken"],
                "input_adapter": _FakeInputAdapter(),
            }
            return {"status": "STARTED"}

        with (
            patch.object(handler, "_lifecycle_from_event", return_value=lifecycle),
            patch.object(handler, "_start_pod_session", new=AsyncMock(side_effect=start_session)) as start,
            patch.object(handler, "_start_session_heartbeat", new=AsyncMock()),
            patch.object(handler.web, "WebSocketResponse", return_value=websocket),
        ):
            result = await handler.app_websocket_ingest(_FakeRequest(session_id, token))

        self.assertIs(result, websocket)
        lifecycle.claim.assert_awaited_once_with(connection_token=token)
        start.assert_awaited_once()
        _, start_kwargs = start.await_args
        self.assertIs(start_kwargs["lifecycle"], lifecycle)
        self.assertTrue(start_kwargs["lifecycle_claimed"])
        self.assertEqual(start_kwargs["connection_token"], token)
        self.assertIn(jti, handler._used_jti)

    async def test_heartbeat_termination_callback_reaches_handler(self):
        lifecycle = _FakeLifecycle()
        state = {"lifecycle": lifecycle}
        command = {
            "reason": "public_end",
            "deadlineAt": "2026-08-28T02:01:00.000Z",
        }

        with patch.object(
            handler,
            "_request_owner_termination",
            new=AsyncMock(),
        ) as request_termination:
            await handler._start_session_heartbeat("session-123", state)
            await lifecycle.heartbeat_callback(command)

        lifecycle.send_event.assert_awaited_once_with("worker_started")
        request_termination.assert_awaited_once_with("session-123", state, command)

    async def test_failed_task_reports_terminal_only_after_processor_cleanup(self):
        order = []
        processor_release = asyncio.Event()
        terminal_sent = asyncio.Event()
        terminal_call = {}

        class OrderedLifecycle:
            async def stop(self):
                order.append("lifecycle_stop")

            async def send_terminal(self, event, **kwargs):
                order.append(event)
                terminal_call.update({"event": event, **kwargs})
                terminal_sent.set()

        async def processor():
            await processor_release.wait()
            order.append("processor_cleanup")
            return {"status": "error"}

        state = {"lifecycle": OrderedLifecycle(), "websocket": None}
        task = asyncio.create_task(processor())
        handler._track_session_task("session-failed", task, state)

        await asyncio.sleep(0)
        self.assertFalse(terminal_sent.is_set())
        processor_release.set()
        await task
        await asyncio.wait_for(terminal_sent.wait(), timeout=1)

        self.assertEqual(
            order,
            ["processor_cleanup", "lifecycle_stop", "session_failed"],
        )
        self.assertEqual(terminal_call["failure_code"], "WORKER_SESSION_FAILED")
        self.assertEqual(terminal_call["failure_message"], "Streaming session failed")
        self.assertNotIn("session-failed", handler._active_sessions)

    async def test_session_task_completion_cancels_pending_reconnect_expiry(self):
        class SilentLifecycle:
            async def stop(self):
                pass

            async def send_terminal(self, _event, **_kwargs):
                pass

        state = {"lifecycle": SilentLifecycle(), "websocket": None}
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        handler._schedule_reconnect_expiry("session-expiry", state, expires_at)
        expiry_task = state["reconnect_expiry_task"]
        self.assertFalse(expiry_task.done())

        async def processor():
            return {"status": "ok"}

        task = asyncio.create_task(processor())
        handler._track_session_task("session-expiry", task, state)
        await task

        for _ in range(50):
            if expiry_task.done():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(expiry_task.done())
        self.assertTrue(expiry_task.cancelled())
        self.assertNotIn("reconnect_expiry_task", state)
        self.assertNotIn("session-expiry", handler._active_sessions)

    async def test_idle_watcher_requests_termination_once(self):
        state = {
            "last_activity_time": time.monotonic() - handler.NO_AUDIO_SESSION_TIMEOUT - 1,
            "is_draining": False,
        }

        with (
            patch.object(handler, "SESSION_IDLE_CHECK_INTERVAL", 0),
            patch.object(handler, "_request_owner_termination", new=AsyncMock()) as terminate,
        ):
            await handler._watch_session_idle("session-idle", state)

        terminate.assert_awaited_once_with(
            "session-idle",
            state,
            {"reason": "no_audio_timeout"},
        )
        self.assertNotIn("ending", state)

    async def test_provider_end_signals_session_before_waiting_for_processor(self):
        processor_release = asyncio.Event()
        websocket = _FakeWebSocket()
        bus = SimpleNamespace(end_session=AsyncMock(side_effect=processor_release.set))
        state = {
            "bus": bus,
            "queue": asyncio.Queue(),
            "websocket": websocket,
        }

        async def processor():
            await processor_release.wait()
            return {"status": "ok"}

        task = asyncio.create_task(processor())
        handler._active_sessions["session-end"] = task
        self.ws_server.active_sessions["session-end"] = state

        result = await handler._cancel_pod_session("session-end")

        self.assertEqual(result["status"], "ENDED")
        self.assertTrue(state["ending"])
        self.assertEqual(state["termination_reason"], "provider_requested")
        bus.end_session.assert_awaited_once()
        self.assertTrue(websocket.closed)
        self.assertFalse(task.cancelled())
        self.assertEqual(task.result(), {"status": "ok"})

    async def test_worker_shutdown_closes_all_registered_aiohttp_websockets(self):
        tasks = []
        states = []
        for index in range(2):
            websocket = _FakeWebSocket()
            state = {
                "bus": SimpleNamespace(end_session=AsyncMock()),
                "queue": asyncio.Queue(),
                "websocket": websocket,
            }
            session_id = f"session-shutdown-{index}"
            task = asyncio.create_task(asyncio.Event().wait())
            tasks.append(task)
            states.append(state)
            handler._active_sessions[session_id] = task
            self.ws_server.active_sessions[session_id] = state

        count = await handler.request_active_session_shutdown()

        self.assertEqual(count, 2)
        for state in states:
            self.assertTrue(state["ending"])
            self.assertTrue(state["websocket"].closed)
            state["bus"].end_session.assert_awaited_once()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_worker_shutdown_cancels_and_awaits_reconnect_expiry(self):
        session_id = "session-shutdown-expiry"
        expiry_started = asyncio.Event()
        expiry_cancelled = asyncio.Event()

        async def expiry():
            expiry_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                expiry_cancelled.set()

        expiry_task = asyncio.create_task(expiry())
        processor = asyncio.create_task(asyncio.Event().wait())
        state = {
            "bus": SimpleNamespace(end_session=AsyncMock()),
            "queue": asyncio.Queue(),
            "websocket": _FakeWebSocket(),
            "reconnect_expiry_task": expiry_task,
        }
        handler._active_sessions[session_id] = processor
        self.ws_server.active_sessions[session_id] = state
        await asyncio.wait_for(expiry_started.wait(), timeout=1)

        await handler.request_active_session_shutdown()

        self.assertTrue(expiry_task.done())
        self.assertTrue(expiry_cancelled.is_set())
        self.assertNotIn("reconnect_expiry_task", state)
        processor.cancel()
        await asyncio.gather(processor, return_exceptions=True)

    async def test_terminal_finalizer_cancels_expiry_that_initiated_termination(self):
        session_id = "session-expiry-terminal"
        termination_started = asyncio.Event()
        expiry_finished = asyncio.Event()
        release_processor = asyncio.Event()

        async def request_termination(*_args):
            termination_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                expiry_finished.set()

        async def processor():
            await release_processor.wait()
            return {"status": "ok"}

        state = {
            "lifecycle": _FakeLifecycle(),
            "websocket": None,
            "connection_lost": True,
        }
        with patch.object(
            handler,
            "_request_owner_termination",
            new=AsyncMock(side_effect=request_termination),
        ):
            expiry_task = asyncio.create_task(
                handler._terminate_after_reconnect_expiry(
                    session_id,
                    state,
                    "2000-01-01T00:00:00.000Z",
                )
            )
            state["reconnect_expiry_task"] = expiry_task
            task = asyncio.create_task(processor())
            handler._track_session_task(session_id, task, state)
            await asyncio.wait_for(termination_started.wait(), timeout=1)
            release_processor.set()
            await task
            self.assertTrue(await handler.drain_finalization_tasks(timeout=1))

        self.assertTrue(expiry_task.done())
        self.assertTrue(expiry_finished.is_set())
        self.assertNotIn("reconnect_expiry_task", state)

    async def test_drain_waits_for_terminal_finalizer(self):
        release_terminal = asyncio.Event()
        terminal_started = asyncio.Event()

        class GatedLifecycle:
            async def stop(self):
                return None

            async def send_terminal(self, *_args, **_kwargs):
                terminal_started.set()
                await release_terminal.wait()

        async def processor():
            return {"status": "error"}

        state = {"lifecycle": GatedLifecycle(), "websocket": None}
        task = asyncio.create_task(processor())
        handler._track_session_task("session-finalizer", task, state)
        await task
        await asyncio.wait_for(terminal_started.wait(), timeout=1)

        drain = asyncio.create_task(handler.drain_finalization_tasks(timeout=1))
        await asyncio.sleep(0)
        self.assertFalse(drain.done())
        release_terminal.set()

        self.assertTrue(await drain)
        self.assertFalse(handler._finalization_tasks)

    async def test_drain_waits_for_failed_session_diagnostic_send(self):
        diagnostic_started = asyncio.Event()
        release_diagnostic = asyncio.Event()
        diagnostic_task = None

        class GatedWebSocket(_FakeWebSocket):
            async def send_str(self, _text):
                nonlocal diagnostic_task
                diagnostic_task = asyncio.current_task()
                diagnostic_started.set()
                await release_diagnostic.wait()

        async def processor():
            return {"status": "error"}

        lifecycle = _FakeLifecycle()
        state = {"lifecycle": lifecycle, "websocket": GatedWebSocket()}
        task = asyncio.create_task(processor())
        handler._track_session_task("session-diagnostic-drain", task, state)
        await task
        await asyncio.wait_for(diagnostic_started.wait(), timeout=1)

        self.assertIn(diagnostic_task, handler._finalization_tasks)
        drain = asyncio.create_task(handler.drain_finalization_tasks(timeout=1))
        await asyncio.sleep(0)
        self.assertFalse(drain.done())
        release_diagnostic.set()

        self.assertTrue(await drain)
        self.assertFalse(handler._finalization_tasks)

    async def test_failed_diagnostic_send_exception_is_retrieved(self):
        class FailingWebSocket(_FakeWebSocket):
            async def send_str(self, _text):
                raise RuntimeError("diagnostic send failed")

        async def processor():
            return {"status": "error"}

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        leaked_contexts = []
        loop.set_exception_handler(lambda _loop, context: leaked_contexts.append(context))
        try:
            state = {"lifecycle": _FakeLifecycle(), "websocket": FailingWebSocket()}
            task = asyncio.create_task(processor())
            handler._track_session_task("session-diagnostic-failure", task, state)
            await task
            self.assertTrue(await handler.drain_finalization_tasks(timeout=1))
            gc.collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        self.assertFalse(
            any(
                context.get("message") == "Task exception was never retrieved"
                for context in leaked_contexts
            )
        )

    async def test_lifecycle_finalizer_exception_is_retrieved(self):
        finalizer_started = asyncio.Event()
        release_finalizer = asyncio.Event()
        finalizer_task = None

        class FailingLifecycle:
            async def stop(self):
                nonlocal finalizer_task
                finalizer_task = asyncio.current_task()
                finalizer_started.set()
                await release_finalizer.wait()
                raise RuntimeError("finalizer failed")

        async def processor():
            return {"status": "ok"}

        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        leaked_contexts = []
        loop.set_exception_handler(lambda _loop, context: leaked_contexts.append(context))
        try:
            task = asyncio.create_task(processor())
            handler._track_session_task(
                "session-finalizer-failure",
                task,
                {"lifecycle": FailingLifecycle(), "websocket": None},
            )
            await task
            await asyncio.wait_for(finalizer_started.wait(), timeout=1)
            release_finalizer.set()
            await asyncio.wait({finalizer_task})
            finalizer_task = None
            gc.collect()
        finally:
            loop.set_exception_handler(previous_handler)

        self.assertFalse(
            any(
                context.get("message") == "Task exception was never retrieved"
                for context in leaked_contexts
            )
        )

    async def test_drain_cancels_stuck_finalizer(self):
        never = asyncio.Event()
        stuck = asyncio.create_task(never.wait())
        handler._finalization_tasks.add(stuck)

        self.assertFalse(await handler.drain_finalization_tasks(timeout=0.01))
        self.assertTrue(stuck.done())
        self.assertFalse(handler._finalization_tasks)


if __name__ == "__main__":
    unittest.main()
