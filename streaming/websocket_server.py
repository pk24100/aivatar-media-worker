# WebSocket server that ingests raw audio bytes for active streaming sessions.
import asyncio
import websockets
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# WebSocket server that ingests audio bytes for active sessions.
class WebsocketIngestionServer:
    # Initialize server with host, port, and session registry.
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.active_sessions: Dict[str, asyncio.Queue] = {}
        self._server: Optional[websockets.serve] = None
        self._active_connections: Dict[str, bool] = {}

    # Return whether the WebSocket server is currently running.
    @property
    def is_running(self) -> bool:
        return self._server is not None

    def register_session(self, session_id: str, token: str) -> asyncio.Queue:
        """Register a new session to receive audio, returning its queue and saving the expected token."""
        if not token:
            raise ValueError("ingestionToken is required")
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "queue": asyncio.Queue(),
                "token": token
            }
            logger.info(f"Registered WebSocket session: {session_id}")
        return self.active_sessions[session_id]["queue"]

    def acquire_connection(self, session_id: str) -> bool:
        """Try to acquire a connection slot for a session. Returns False if already occupied."""
        if session_id in self._active_connections:
            return False
        self._active_connections[session_id] = True
        return True

    def release_connection(self, session_id: str):
        """Release the connection slot for a session."""
        self._active_connections.pop(session_id, None)

    def has_connection(self, session_id: str) -> bool:
        """Return whether a session currently owns its ingestion connection."""
        return session_id in self._active_connections

    def unregister_session(self, session_id: str):
        """Unregister an existing session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Unregistered WebSocket session: {session_id}")

    async def _handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connections and route audio data."""
        from urllib.parse import urlparse, parse_qs
        
        parsed_path = urlparse(path)
        session_id = parsed_path.path.strip("/")
        query_params = parse_qs(parsed_path.query)
        provided_token = query_params.get("token", [None])[0]
        
        if session_id not in self.active_sessions:
            logger.warning(f"Connection attempted for unknown session: {session_id}")
            await websocket.close(code=4004, reason="Unknown session ID")
            return
            
        expected_token = self.active_sessions[session_id].get("token")
        if not expected_token or provided_token != expected_token:
            logger.warning(f"Unauthorized connection attempt for session {session_id} (invalid token)")
            await websocket.close(code=4001, reason="Unauthorized: Invalid Token")
            return
            
        logger.info(f"Client connected for session: {session_id}")
        audio_queue = self.active_sessions[session_id]["queue"]

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # It's raw audio data (PCM or similar depending on client config)
                    # We put it into the session's queue for processing by the streaming engine
                    await audio_queue.put(message)
                else:
                    logger.debug(f"Received non-binary message on session {session_id}: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected for session: {session_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection for {session_id}: {e}")
        finally:
            # Signal end of stream if needed
            await audio_queue.put(None)

    async def start(self):
        """Start the WebSocket server in the background."""
        if self._server is not None:
            logger.info("WebSocket ingestion server already running.")
            return
        logger.info(f"Starting WebSocket ingestion server on ws://{self.host}:{self.port}")
        self._server = await websockets.serve(self._handle_connection, self.host, self.port)
        
    async def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("WebSocket ingestion server stopped.")

# Global instance
ws_server = WebsocketIngestionServer()
