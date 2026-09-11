"""
BatchedStreamingEngine: Manages multiple concurrent streaming sessions with
batched inference using run_pipeline_batch.

All sessions share a single pipeline and are batched together with a 20ms
fixed wait window. Uses all GPU SMs (no green context partitioning).

Ported from modal_stress_test.py and adapted for production use:
- No simulated audio (real audio fed via feed_audio)
- Background thread lifecycle (start/stop) instead of fixed-duration run_loop
- SSRF-safe avatar URL download (reused from flashhead_streaming.py)

This package is modularized into:
- _session_states: session state constants and timing config
- _avatar: avatar URL/path resolution and download with SSRF protection
- _batched_session: BatchedSession per-session state class
- _engine: BatchedStreamingEngine central batching engine
"""

import os
import sys

# Ensure SoulX-FlashHead is importable (same path as the original monolithic
# batched.py: resolves to streaming/SoulX-FlashHead from this package's
# __init__.py which is one directory deeper than the original file).
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "SoulX-FlashHead",
    )
)

# Re-export flash_head names that the original module imported (preserves
# import side effects for any code that relied on them being available).
from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, run_pipeline_batch  # noqa: F401

from ._session_states import (
    INITIALIZING,
    ACTIVE,
    DRAINING,
    ENDED,
    WAIT_WINDOW_MS,
    IDLE_TIMEOUT_S,
    REACTIVATION_TIMEOUT_S,
)
from ._avatar import (
    _resolve_avatar_path,
    _is_valid_image_bytes,
    _IMAGE_MAGIC_BYTES,
    MAX_DOWNLOAD_BYTES,
    _AVATAR_DOWNLOAD_MAX_ATTEMPTS,
    _AVATAR_DOWNLOAD_BASE_DELAY,
    _TRANSIENT_NETWORK_ERRORS,
)
from ._batched_session import BatchedSession
from ._engine import BatchedStreamingEngine

__all__ = [
    "INITIALIZING",
    "ACTIVE",
    "DRAINING",
    "ENDED",
    "WAIT_WINDOW_MS",
    "IDLE_TIMEOUT_S",
    "REACTIVATION_TIMEOUT_S",
    "BatchedSession",
    "BatchedStreamingEngine",
    "_resolve_avatar_path",
]
