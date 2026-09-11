"""Environment-derived constants for the media worker (extracted from handler.py)."""

import os

# --- WebSocket Security Constants (Fixes 4, 6, 8, 15) ---
WORKER_AUTH_SECRET = os.environ.get("WORKER_AUTH_SECRET", "")
ALLOWED_WS_ORIGINS = set(o.strip() for o in os.environ.get("ALLOWED_WS_ORIGINS", "").split(",") if o.strip())
MAX_AUDIO_CHUNK_BYTES = 1_048_576   # 1 MB - 5s @ 48kHz 16-bit stereo PCM
WS_INACTIVITY_TIMEOUT = int(os.environ.get("WS_INACTIVITY_TIMEOUT", "45"))
NO_AUDIO_SESSION_TIMEOUT = int(os.environ.get("NO_AUDIO_SESSION_TIMEOUT", "120"))
SESSION_IDLE_CHECK_INTERVAL = int(os.environ.get("SESSION_IDLE_CHECK_INTERVAL", "5"))
BACKEND_INTERNAL_URL = os.environ.get("BACKEND_INTERNAL_URL", "")
WS_AUDIO_RATE_PER_SEC = int(os.environ.get("WS_AUDIO_RATE_PER_SEC", "60"))  # max audio messages/sec (sustained)
WS_AUDIO_BURST = int(os.environ.get("WS_AUDIO_BURST", "200"))  # token bucket burst capacity
WS_MAX_BYTES_PER_SEC = int(os.environ.get("WS_MAX_BYTES_PER_SEC", "192000"))  # 48kHz 16-bit mono real-time ceiling
WS_MAX_CONNECTIONS_PER_IP = int(os.environ.get("WS_MAX_CONNECTIONS_PER_IP", "10"))  # max concurrent WS per IP per 60s window

DEFAULT_FLASHHEAD_HF_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub"
)
DEFAULT_FLASHHEAD_HF_REPO_ID = "pkam24100/aivatar-flashhead-model"
LEGACY_FLASHHEAD_CKPT_DIR = "/app/models/SoulX-FlashHead-1_3B"
WAV2VEC_DIR = "/app/models/wav2vec2-base-960h"
FLASHHEAD_HF_REPO_ID = DEFAULT_FLASHHEAD_HF_REPO_ID
FLASHHEAD_HF_CACHE_DIR = DEFAULT_FLASHHEAD_HF_CACHE_DIR
FLASHHEAD_HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
WORKER_POOL_SIZE = int(os.getenv("AIVATAR_WORKER_CONCURRENCY", "3"))
if WORKER_POOL_SIZE < 1:
    raise ValueError("AIVATAR_WORKER_CONCURRENCY must be at least 1")
WORKER_HTTP_HOST = os.getenv("AIVATAR_HTTP_HOST", "0.0.0.0")
WORKER_HTTP_PORT = int(os.getenv("AIVATAR_HTTP_PORT", "8000"))
