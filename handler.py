"""AiVatar media worker entrypoint and shared-state facade (Modal-only).

The implementation lives in the worker/ package; this module keeps the full
`handler.*` surface (constants, state, helpers, route handlers) so tests,
modal_worker, app_factory and entrypoints keep working unchanged. Extracted
modules read state back through this module at call time, so monkey-patching
attributes here (e.g. handler.ws_server, handler.model_pool) still applies.
Local `python handler.py` boots the same Modal worker app via run_worker_app().
"""

import sys

# When run as `python3 handler.py`, alias __main__ as "handler" so that
# `import handler` inside the worker/ package binds this same module instead
# of re-executing handler.py as a second instance.
if __name__ == "__main__":
    sys.modules.setdefault("handler", sys.modules["__main__"])

import asyncio
import contextlib
import logging
import os
import time
from collections import defaultdict
from datetime import datetime

import aiohttp
import jwt as pyjwt
from aiohttp import web
from huggingface_hub import snapshot_download

from streaming.orchestration.stream_processor import run_streaming_session
from streaming.protocol.messages import AudioReadyMessage, EndedMessage, ErrorMessage, StartedMessage, serialize_message
from streaming.transport.websocket_server import ws_server
from streaming.lifecycle.worker import (
    LifecycleError,
    WorkerLifecycleClient,
    WorkerLocalLifecycleTransition,
)
from streaming.orchestration.guards import reject_parked_session_config
from utils.model_pool import FlashHeadModelPool

_BATCHED_INFERENCE = os.environ.get("AIVATAR_BATCHED_INFERENCE", "1") == "1"
if _BATCHED_INFERENCE:
    from streaming.orchestration.batched_stream_processor import run_batched_streaming_session

# Shared BatchedStreamingEngine instance (set by modal_app.py serve())
batched_engine = None

from worker.config import (
    ALLOWED_WS_ORIGINS,
    BACKEND_INTERNAL_URL,
    DEFAULT_FLASHHEAD_HF_CACHE_DIR,
    DEFAULT_FLASHHEAD_HF_REPO_ID,
    FLASHHEAD_HF_CACHE_DIR,
    FLASHHEAD_HF_REPO_ID,
    FLASHHEAD_HF_TOKEN,
    LEGACY_FLASHHEAD_CKPT_DIR,
    MAX_AUDIO_CHUNK_BYTES,
    NO_AUDIO_SESSION_TIMEOUT,
    SESSION_IDLE_CHECK_INTERVAL,
    WAV2VEC_DIR,
    WORKER_AUTH_SECRET,
    WORKER_HTTP_HOST,
    WORKER_HTTP_PORT,
    WORKER_POOL_SIZE,
    WS_AUDIO_BURST,
    WS_AUDIO_RATE_PER_SEC,
    WS_INACTIVITY_TIMEOUT,
    WS_MAX_BYTES_PER_SEC,
    WS_MAX_CONNECTIONS_PER_IP,
)

# --- Fix 13: One-time JWT tracking (jti -> expiry timestamp) ---
_used_jti: dict = {}

# --- Fix 15: IP-based connection throttling ---
_ip_connections: dict = defaultdict(list)  # ip -> [timestamp, ...]

from worker.security import (
    AudioRateLimiter,
    _check_ip_limit,
    _check_jti,
    _consume_jti,
    _jti_is_available,
    _prune_used_jti,
)
from worker.models import (
    DATA_ROOT,
    FLASHHEAD_CKPT_DIR,
    INITIAL_PIPELINE_COUNT,
    _is_valid_flashhead_ckpt_dir,
    _resolve_flashhead_ckpt_dir,
    _verify_models,
    model_pool,
)
from worker.lifecycle import (
    _cancel_reconnect_expiry,
    _end_session_via_backend,
    _lifecycle_from_event,
    _request_owner_termination,
    _schedule_reconnect_expiry,
    _start_session_heartbeat,
    _terminate_after_reconnect_expiry,
    _watch_session_idle,
)
from worker.finalization import (
    _finalize_session_task,
    _observe_finalization_task,
    _rollback_self_started_session,
    _run_session_finalizer,
    _track_session_task,
    drain_finalization_tasks,
)
from worker.sessions import (
    _cancel_pod_session,
    _ensure_ws_server_started,
    _execute_streaming_event,
    _get_livekit_url,
    _get_room_url,
    _is_streaming,
    _mint_livekit_token,
    _start_pod_session,
    request_active_session_shutdown,
)
from worker.http_routes import (
    app_ping,
    pod_health,
    pod_ready,
    pod_session_end,
    pod_session_start,
    pod_session_status,
)
from worker.ws_ingest import _safe_send_str, app_websocket_ingest

logger = logging.getLogger(__name__)

_ws_started = False
_ws_lock = asyncio.Lock()
_active_sessions = {}
_finalization_tasks: set[asyncio.Task] = set()
_model_ready_event = None
_model_init_error = None


def configure_model_readiness(ready: bool):
    """Initialize model readiness on the aiohttp loop that owns sessions."""
    global _model_ready_event, _model_init_error
    _model_ready_event = asyncio.Event()
    _model_init_error = None
    if ready:
        _model_ready_event.set()


def mark_model_ready(error: Exception = None):
    global _model_init_error
    _model_init_error = error
    if _model_ready_event is not None:
        _model_ready_event.set()


async def wait_for_model_ready():
    if _model_ready_event is None:
        return
    await _model_ready_event.wait()
    if _model_init_error is not None:
        raise RuntimeError("FlashHead model initialization failed") from _model_init_error


# Run the HTTP application for Modal-only worker mode.
async def run_worker_app():
    """Start the Modal-only aiohttp worker app and serve until cancelled."""
    from app_factory import build_app
    try:
        from utils.errors import init_sentry, install_error_hooks
        init_sentry()
        install_error_hooks()
    except Exception:
        pass

    # Narrow health order: build_app -> runner.setup -> site.start -> mark_ready.
    # /health stays always ok, /readyz gates on ws_server + model_ready +
    # batched thread, /sessions/start stays independent of ready.
    app = await build_app()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, WORKER_HTTP_HOST, WORKER_HTTP_PORT)
    await site.start()
    configure_model_readiness(ready=True)

    logger.info("AiVatar load balancer worker listening on http://%s:%s", WORKER_HTTP_HOST, WORKER_HTTP_PORT)

    # Keep running until cancelled
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(run_worker_app())
