"""
Shared aiohttp app factory used by the Modal worker (handler.py, modal_app.py).
Do not fork this file per provider - only the serving wrapper differs.
"""

from aiohttp import web
import handler


async def build_app() -> web.Application:
    """Build and return the aiohttp application with all routes wired."""
    # Safe to call on Modal: starts the websocket ingestion server on port 8765,
    # which is unused by Modal but required by handler internals.
    await handler._ensure_ws_server_started()

    app = web.Application()
    app.router.add_get("/ping", handler.app_ping)
    app.router.add_get("/health", handler.pod_health)
    app.router.add_get("/healthz", handler.pod_health)
    app.router.add_get("/readyz", handler.pod_ready)
    app.router.add_post("/sessions/start", handler.pod_session_start)
    app.router.add_post("/sessions/{session_id}/end", handler.pod_session_end)
    app.router.add_get("/sessions/{session_id}/status", handler.pod_session_status)
    app.router.add_get("/ws/{session_id}", handler.app_websocket_ingest)
    return app
