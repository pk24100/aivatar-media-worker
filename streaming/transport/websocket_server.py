"""Canonical WebSocket ingestion server and session registry."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import parse_qs, urlparse

import websockets

from streaming.transport.adapters.websocket_input import WebsocketInputAdapter
from streaming.core.audio_bus import CanonicalAudioBus
from streaming.orchestration.guards import PARKED_PROVIDER_PROFILES, normalize_optional_name
from streaming.protocol.messages import ErrorMessage, EndedMessage, ProtocolMessage, serialize_message

logger = logging.getLogger(__name__)

PROFILE_DRIVING_INPUT_PROVIDERS = frozenset({"deepgram", "gemini"})


def _create_input_adapter(
    bus: CanonicalAudioBus,
    *,
    session_id: str,
    input_provider: str | None,
    telephony_provider: str | None,
):
    if telephony_provider:
        raise ValueError("Invalid telephonyProvider")
    profile_name = normalize_optional_name(input_provider)
    if profile_name in PARKED_PROVIDER_PROFILES:
        raise ValueError("Invalid inputProvider")
    return WebsocketInputAdapter(
        bus,
        expected_session_id=session_id,
        provider_profile=(
            profile_name if profile_name in PROFILE_DRIVING_INPUT_PROVIDERS else None
        ),
    )


class WebsocketIngestionServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self._server = None
        self._active_connections: dict[str, bool] = {}

    @property
    def is_running(self) -> bool:
        return self._server is not None

    def register_session(
        self,
        session_id: str,
        token: str,
        *,
        input_provider: str | None = None,
        telephony_provider: str | None = None,
    ) -> asyncio.Queue:
        if not token:
            raise ValueError("ingestionToken is required")
        if session_id not in self.active_sessions:
            bus = CanonicalAudioBus()
            input_adapter = _create_input_adapter(
                bus,
                session_id=session_id,
                input_provider=input_provider,
                telephony_provider=telephony_provider,
            )
            self.active_sessions[session_id] = {
                "queue": bus.queue,
                "bus": bus,
                "input_adapter": input_adapter,
                "input_provider": input_provider,
                "telephony_provider": telephony_provider,
                "token": token,
                "websocket": None,
                "ending": False,
            }
            logger.info(
                "Registered WebSocket session: %s input_provider=%s telephony_provider=%s",
                session_id,
                input_provider or "none",
                telephony_provider or "none",
            )
        else:
            state = self.active_sessions[session_id]
            state["token"] = token
            state["input_provider"] = input_provider or state.get("input_provider")
            state["telephony_provider"] = telephony_provider or state.get("telephony_provider")
        return self.active_sessions[session_id]["queue"]

    def get_audio_bus(self, session_id: str) -> CanonicalAudioBus:
        state = self.active_sessions.get(session_id)
        if state is None:
            raise KeyError(session_id)
        return state["bus"]

    def get_input_adapter(self, session_id: str) -> WebsocketInputAdapter:
        state = self.active_sessions.get(session_id)
        if state is None:
            raise KeyError(session_id)
        return state["input_adapter"]

    def acquire_connection(self, session_id: str) -> bool:
        if session_id in self._active_connections:
            return False
        self._active_connections[session_id] = True
        return True

    def release_connection(self, session_id: str) -> None:
        self._active_connections.pop(session_id, None)

    def has_connection(self, session_id: str) -> bool:
        return session_id in self._active_connections

    def unregister_session(self, session_id: str) -> None:
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.release_connection(session_id)
            logger.info("Unregistered WebSocket session: %s", session_id)

    async def send_message(self, session_id: str, message: ProtocolMessage) -> bool:
        state = self.active_sessions.get(session_id)
        if not state:
            return False
        websocket = state.get("websocket")
        if websocket is None or getattr(websocket, "closed", False):
            return False
        payload = serialize_message(message)
        lock = state.setdefault("send_lock", asyncio.Lock())
        try:
            async with lock:
                if getattr(websocket, "closed", False):
                    return False
                if hasattr(websocket, "send_str"):
                    await websocket.send_str(payload)
                else:
                    await websocket.send(payload)
            return True
        except Exception:
            logger.debug("Failed to send protocol message for session %s", session_id, exc_info=True)
            return False

    async def _handle_connection(self, websocket, path: str | None = None):
        parsed_path = urlparse(path or getattr(websocket, "path", ""))
        session_id = parsed_path.path.strip("/")
        query_params = parse_qs(parsed_path.query)
        provided_token = query_params.get("token", [None])[0]

        state = self.active_sessions.get(session_id)
        if state is None:
            await websocket.close(code=4004, reason="Unknown session ID")
            return
        if not state.get("token") or provided_token != state["token"]:
            await websocket.close(code=4001, reason="Unauthorized: Invalid Token")
            return
        if not self.acquire_connection(session_id):
            await websocket.close(code=4000, reason="Session already has an active connection")
            return

        state["websocket"] = websocket
        adapter = state["input_adapter"]
        attach_websocket = getattr(adapter, "attach_websocket", None)
        if attach_websocket is not None:
            attach_websocket(websocket)
        ended = False
        logger.info("Client connected for session: %s", session_id)
        try:
            async for raw in websocket:
                responses = await adapter.handle_message(raw)
                fatal_response = False
                for response in responses:
                    await websocket.send(serialize_message(response))
                    if isinstance(response, ErrorMessage) and response.fatal:
                        fatal_response = True
                    if isinstance(response, EndedMessage):
                        ended = True
                        break
                if fatal_response:
                    await websocket.close(code=4000, reason="Fatal protocol error")
                    break
                if ended:
                    break
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected for session: %s", session_id)
        except Exception:
            logger.exception("Error handling WebSocket connection for %s", session_id)
        finally:
            state["websocket"] = None
            await adapter.disconnect()
            self.release_connection(session_id)
            if not ended:
                with contextlib.suppress(Exception):
                    await state["bus"].queue.put(None)

    async def start(self):
        if self._server is not None:
            return
        self._server = await websockets.serve(self._handle_connection, self.host, self.port)
        logger.info("Started WebSocket ingestion server on ws://%s:%s", self.host, self.port)

    async def stop(self):
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        logger.info("WebSocket ingestion server stopped")


import contextlib

ws_server = WebsocketIngestionServer()
