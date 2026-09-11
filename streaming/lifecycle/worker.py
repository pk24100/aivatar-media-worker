"""Backend-fenced worker ownership and lifecycle callbacks.

The media worker keeps local queues for performance, but the backend owns the
cross-container lease. This module deliberately stores only opaque lease data,
never room or ingestion credentials.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import jwt as pyjwt

logger = logging.getLogger(__name__)

BACKEND_INTERNAL_URL = os.environ.get("BACKEND_INTERNAL_URL", "").rstrip("/")
WORKER_AUTH_SECRET = os.environ.get("WORKER_AUTH_SECRET", "")
HEARTBEAT_INTERVAL_SECONDS = max(
    float(os.environ.get("WORKER_LIFECYCLE_HEARTBEAT_SECONDS", "15")), 1.0
)
WORKER_INSTANCE_ID = os.environ.get(
    "AIVATAR_WORKER_INSTANCE_ID",
    f"{socket.gethostname()}-{uuid.uuid4()}",
)


class LifecycleError(RuntimeError):
    """Raised when a fenced backend lifecycle operation is rejected."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


def lifecycle_enabled() -> bool:
    """Return whether the deployment has enough configuration for lifecycle fencing."""
    return bool(BACKEND_INTERNAL_URL and WORKER_AUTH_SECRET)


class WorkerLocalLifecycleTransition:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._connection_epoch = 0

    async def commit_connection(self, action: Callable[[], Awaitable[None]]) -> int:
        async with self._lock:
            next_epoch = self._connection_epoch + 1
            await action()
            self._connection_epoch = next_epoch
            return next_epoch

    async def apply_if_current(
        self,
        connection_epoch: int,
        action: Callable[[], Awaitable[None]],
    ) -> bool:
        async with self._lock:
            if connection_epoch != self._connection_epoch:
                return False
            await action()
            return True


class WorkerLifecycleClient:
    """Own one session's backend lease for the lifetime of a worker task."""

    def __init__(
        self,
        session_id: str,
        *,
        lease_id: str | None,
        generation: int | None,
        instance_id: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.lease_id = lease_id
        self.generation = int(generation) if generation is not None else None
        self.instance_id = instance_id or WORKER_INSTANCE_ID
        self._heartbeat_task: asyncio.Task | None = None
        self._closed = False
        self._terminal_sent = False

    @property
    def is_configured(self) -> bool:
        return lifecycle_enabled() and bool(self.lease_id and self.generation is not None)

    def _mint_callback_token(self) -> str:
        if not WORKER_AUTH_SECRET:
            raise LifecycleError("WORKER_AUTH_SECRET is not configured")
        now = int(time.time())
        return pyjwt.encode(
            {
                "sessionId": self.session_id,
                "workerLeaseId": self.lease_id,
                "workerGeneration": self.generation,
                "jti": str(uuid.uuid4()),
                "iat": now,
                "exp": now + 120,
            },
            WORKER_AUTH_SECRET,
            algorithm="HS256",
        )

    async def _request(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        token: str | None = None,
    ) -> dict[str, Any]:
        if not self.is_configured:
            return {}
        auth_token = token or self._mint_callback_token()
        url = f"{BACKEND_INTERNAL_URL}{path}"
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as http_session:
                async with http_session.post(
                    url,
                    headers={"Authorization": f"Bearer {auth_token}"},
                    json=payload,
                ) as response:
                    try:
                        data = await response.json(content_type=None)
                    except Exception:
                        data = {}
                    if response.status >= 400:
                        error = data.get("error") if isinstance(data, dict) else {}
                        code = error.get("code") if isinstance(error, dict) else None
                        raise LifecycleError(
                            "Backend lifecycle request rejected",
                            status=response.status,
                            code=code,
                        )
                    return data if isinstance(data, dict) else {}
        except LifecycleError:
            raise
        except Exception as exc:
            raise LifecycleError(
                "Backend lifecycle request failed",
                code="BACKEND_REQUEST_FAILED",
            ) from exc

    async def claim(self, *, connection_token: str | None = None) -> dict[str, Any]:
        """Claim the persisted lease before creating any local session state."""
        if not self.is_configured:
            return {}
        result = await self._request(
            f"/internal/sessions/{self.session_id}/claim",
            {
                "leaseId": self.lease_id,
                "generation": self.generation,
                "instanceId": self.instance_id,
            },
            token=connection_token,
        )
        self.lease_id = result.get("leaseId", self.lease_id)
        response_generation = result.get("generation", self.generation)
        self.generation = int(response_generation) if response_generation is not None else self.generation
        return result

    async def send_event(
        self,
        event: str,
        *,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> dict[str, Any]:
        """Record an owner-fenced lifecycle event and return any termination command."""
        if not self.is_configured:
            return {}
        payload: dict[str, Any] = {
            "event": event,
            "leaseId": self.lease_id,
            "generation": self.generation,
            "instanceId": self.instance_id,
        }
        if failure_code:
            payload["failureCode"] = failure_code
        if failure_message:
            # Provider exceptions can include request URLs or credentials. Persist
            # only a stable diagnostic instead of forwarding arbitrary exception text.
            payload["failureMessage"] = "Worker reported failure"
        return await self._request(
            f"/internal/sessions/{self.session_id}/lifecycle",
            payload,
        )

    async def send_terminal(
        self,
        event: str,
        *,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> None:
        """Report exactly one terminal state after all local cleanup is complete."""
        if self._terminal_sent:
            return
        self._terminal_sent = True
        try:
            await self.send_event(
                event,
                failure_code=failure_code,
                failure_message=failure_message,
            )
        except LifecycleError as exc:
            logger.warning(
                "WORKER_LIFECYCLE_TERMINAL_FAILED session=%s event=%s code=%s status=%s",
                self.session_id,
                event,
                exc.code,
                exc.status,
            )

    def start_heartbeat(
        self,
        on_terminate: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        if not self.is_configured or self._heartbeat_task is not None:
            return
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(on_terminate),
            name=f"worker-heartbeat-{self.session_id}",
        )

    async def _heartbeat_loop(
        self,
        on_terminate: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> None:
        while not self._closed and not self._terminal_sent:
            try:
                result = await self.send_event("heartbeat")
                terminate = result.get("terminate") if isinstance(result, dict) else None
                if terminate and on_terminate is not None:
                    await on_terminate(terminate)
                    return
            except LifecycleError as exc:
                logger.warning(
                    "WORKER_LIFECYCLE_HEARTBEAT_FAILED session=%s code=%s status=%s",
                    self.session_id,
                    exc.code,
                    exc.status,
                )
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                raise

    async def pause_heartbeat(self) -> None:
        """Stop renewal temporarily while allowing the reconnect grace lease to expire."""
        if self._heartbeat_task is None:
            return
        self._heartbeat_task.cancel()
        try:
            await self._heartbeat_task
        except asyncio.CancelledError:
            pass
        self._heartbeat_task = None

    async def stop(self, *, release: bool = False) -> None:
        self._closed = True
        await self.pause_heartbeat()
        if release and not self._terminal_sent:
            try:
                await self.send_event("owner_released")
            except LifecycleError:
                logger.debug(
                    "WORKER_LIFECYCLE_RELEASE_FAILED session=%s",
                    self.session_id,
                    exc_info=True,
                )
