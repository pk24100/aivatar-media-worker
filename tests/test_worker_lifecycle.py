import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BACKEND_INTERNAL_URL", "https://backend.example.test")
os.environ.setdefault("WORKER_AUTH_SECRET", "test-worker-secret-with-at-least-32-bytes")

import jwt as pyjwt
from streaming.lifecycle import worker
from streaming.lifecycle.worker import WorkerLifecycleClient


class WorkerLifecycleClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_transition_rejects_stale_epoch_after_await(self):
        transition = worker.WorkerLocalLifecycleTransition()
        first_epoch = await transition.commit_connection(AsyncMock())
        second_epoch = await transition.commit_connection(AsyncMock())
        stale_action = AsyncMock()

        applied = await transition.apply_if_current(first_epoch, stale_action)

        self.assertEqual(second_epoch, first_epoch + 1)
        self.assertFalse(applied)
        stale_action.assert_not_awaited()

    def test_callback_token_contains_only_fence_metadata(self):
        client = WorkerLifecycleClient(
            "session-123",
            lease_id="lease-123",
            generation=7,
            instance_id="worker-123",
        )

        claims = pyjwt.decode(
            client._mint_callback_token(),
            os.environ["WORKER_AUTH_SECRET"],
            algorithms=["HS256"],
        )

        self.assertEqual(claims["sessionId"], "session-123")
        self.assertEqual(claims["workerLeaseId"], "lease-123")
        self.assertEqual(claims["workerGeneration"], 7)
        self.assertIn("jti", claims)
        self.assertNotIn("roomToken", claims)
        self.assertNotIn("ingestionToken", claims)

    async def test_terminal_callback_is_sent_once(self):
        client = WorkerLifecycleClient(
            "session-123",
            lease_id="lease-123",
            generation=1,
            instance_id="worker-123",
        )
        client.send_event = AsyncMock(return_value={})

        await asyncio.gather(
            client.send_terminal("session_ended"),
            client.send_terminal("session_ended"),
        )

        client.send_event.assert_awaited_once_with(
            "session_ended",
            failure_code=None,
            failure_message=None,
        )

    async def test_heartbeat_delivers_termination_command_once(self):
        client = WorkerLifecycleClient(
            "session-123",
            lease_id="lease-123",
            generation=1,
            instance_id="worker-123",
        )
        command = {"reason": "public_end", "deadlineAt": "2026-08-28T02:01:00.000Z"}
        client.send_event = AsyncMock(return_value={"terminate": command})
        on_terminate = AsyncMock()

        await client._heartbeat_loop(on_terminate)

        client.send_event.assert_awaited_once_with("heartbeat")
        on_terminate.assert_awaited_once_with(command)

    async def test_terminal_failure_message_is_sanitized_before_callback(self):
        client = WorkerLifecycleClient(
            "session-123",
            lease_id="lease-123",
            generation=1,
            instance_id="worker-123",
        )
        client._request = AsyncMock(return_value={})

        await client.send_terminal(
            "session_failed",
            failure_code="PROCESSOR_FAILED",
            failure_message="roomToken=secret-room-token",
        )

        client._request.assert_awaited_once()
        _path, payload = client._request.await_args.args
        self.assertEqual(payload["failureCode"], "PROCESSOR_FAILED")
        self.assertEqual(payload["failureMessage"], "Worker reported failure")
        self.assertNotIn("secret-room-token", str(payload))


if __name__ == "__main__":
    unittest.main()
