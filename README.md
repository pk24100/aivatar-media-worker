# AI Avatar Media Worker

## Overview

The media worker runs **SoulX-FlashHead Lite** avatar inference and feeds the resulting video into LiveKit rooms. It supports **streaming-only** sessions and is packaged for Modal deployments.

## Layout

```
aivatar-media-worker/
├── handler.py                # Modal worker entry (run_worker_app)
├── entrypoint.sh            # container entrypoint
├── Dockerfile               # GPU-enabled container for Modal
├── requirements.txt         # Python deps
├── scripts/                 # Build + deployment helpers
│   └── download_models.sh
├── streaming/               # LiveKit session logic
│   ├── stream_processor.py
│   ├── flashhead_streaming.py
│   ├── websocket_server.py
│   └── ...
├── utils/                   # FlashHead model pool + helpers
│   └── model_pool.py
├── models/                  # FlashHead weights/config
│   ├── SoulX-FlashHead-1_3B/
│   └── wav2vec2-base-960h/
├── SoulX-FlashHead/         # upstream FlashHead repo (core only)
│   └── flash_head/
│       ├── inference.py
│       ├── configs/
│       ├── src/
│       ├── audio_analysis/
│       ├── ltx_video/
│       └── utils/
└── .env                     # runtime config (should stay private)
```

## Requirements

- Python 3.11+; `pip install -r requirements.txt`.
- CUDA 12+ GPU (NVIDIA) RTX 4090 recommended for 3 concurrent streams.
- Modal secrets for Hugging Face, LiveKit, worker auth, and idle-video R2 (see Modal section).

## Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` (copy from `.env.example` or keep `.env` private) and set:
   - `WORKER_AUTH_SECRET` to the same secret configured on the backend. Use a deployment secret store rather than a literal value in an image, script, or template.
   - `BACKEND_INTERNAL_URL` to an HTTPS backend URL reachable from the worker. The worker uses it for the ownership claim, lifecycle events, heartbeats, and terminal callbacks.
   - `WORKER_LIFECYCLE_HEARTBEAT_SECONDS`, normally `15`. Keep it safely below the backend `WORKER_OWNER_LEASE_MS`, normally `45000` milliseconds.
   - optionally `HUGGING_FACE_HUB_TOKEN` / `HF_TOKEN` for private Hugging Face model access via Modal secrets.
   - `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `LIVEKIT_URL` only when using the stress wrapper or a future managed-room mode.
3. Do not set one shared `AIVATAR_WORKER_INSTANCE_ID` across a scaled deployment. When unset, each worker process creates a unique identity for generation fencing.
4. Download models using `scripts/download_models.sh` before building the Docker image.

The worker defaults internally to:

- cached FlashHead repo `pkam24100/aivatar-flashhead-model`
- Hugging Face cache root `~/.cache/huggingface/hub` for local runs (baked into the Modal image)
- baked wav2vec2 path `/app/models/wav2vec2-base-960h`
- legacy baked FlashHead fallback `/app/models/SoulX-FlashHead-1_3B`

## Running

- `python handler.py` (also invoked by `entrypoint.sh` / Dockerfile) boots the Modal worker app via `run_worker_app()`.
  - `_verify_models()` checks that FlashHead and Wav2Vec2 model directories exist.
  - **Streaming-only BYOLR mode** requires `roomName`, `livekitToken`, `sourceImage`, and `customLivekitUrl` from the customer session; it runs `streaming.stream_processor.run_streaming_session`.
  - Modal restores one CPU-snapshotted pipeline, moves and warms it on GPU, and shares that same pipeline object across the logical slots in each tier (default pool size 3).
- Modal image build bakes weights via `modal_worker/image.py`; see `_archived/runpod-sideline/` for the retired RunPod build helper.
- `scripts/download_models.sh` downloads `wav2vec2-base-960h` by default and supports `DOWNLOAD_FLASHHEAD=1` for an explicit local FlashHead download.

## Provider deployment safety

### Modal

Create all four Modal secrets referenced by `modal_app.py`: `huggingface-secret` for private model-repository access, `livekit-secret` for the LiveKit URL and API credentials, `aivatar-worker-secret` with `WORKER_AUTH_SECRET` and `BACKEND_INTERNAL_URL`, and read-only `aivatar-idle-video-r2` access for the default idle-video bucket. Never put secret values in this README, an image, or a deployment script. `WORKER_AUTH_SECRET` must match the backend value. Model weights are baked into the Modal image via snapshot download and restored from CPU memory snapshots.

Both production tiers use `min_containers=0` and `scaledown_window=15`. A real session starts capacity on demand. Production creates no dummy or pre-warmed rooms, and the default backend compute-health path stays non-waking instead of probing worker `/readyz`.

Modal snapshots one pipeline on CPU. Startup restores it, moves it to GPU, warms it, shares it across all logical pool slots, and only then binds HTTP. `_load_remaining_pipelines()` is retained as a compatibility no-op that reports pool state; it does not load independent models in the background.

Control startup remains the default. Enable backend `MODAL_WEBSOCKET_START_ENABLED=true` only after a deployed WebSocket-authority path has passed owner-claim, reconnect, cleanup, and rollback tests. Backend owner lease and generation remain the persisted authority. Every accepted reconnect rotates that persisted generation, preventing reuse of its short-lived connection JWT. The implemented connection-transition guard is worker-local: a local lock and monotonically increasing connection epoch serialize connect/disconnect transitions, are not stored in `worker_assignment`, and prevent an old disconnect callback from pausing a newer connection. Once the backend returns disconnect grace, later heartbeat activity cannot extend that deadline.

### Retired providers

Retired RunPod helpers live in `_archived/runpod-sideline/` (serverless setup, endpoint tests, overflow pod test, legacy build script). Do not use them for new deployments; Modal is the only supported provider.

For Modal-only operation, do not persist or log room tokens, ingestion tokens, WebSocket JWTs, API keys, or customer credentials. A canonical `ended` message acknowledges the WebSocket protocol. Backend terminal state confirms provider cleanup and billing finalization.

## Streaming & Utilities

- `streaming/stream_processor.py` orchestrates LiveKit connections and FlashHead streaming sessions.
- `streaming/flashhead_streaming.py` contains `FlashHeadStreamingEngine` for real-time video generation from audio chunks.
- `streaming/websocket_server.py` handles server-to-server WebSocket audio ingestion with token authentication.
- `streaming/sip_handler.py` is parked for MVP (SIP/telephony sidelined); WebSocket PCM is the sole ingestion path.
- `utils/model_pool.py` manages a pool of 3 FlashHead pipeline instances for concurrent stream processing.
- `streaming/video_publisher.py` publishes generated video frames to LiveKit rooms.
- `streaming/output_upscale.py` upscales every published frame from the model's native 512x512 to the output size (default 1024x1024) via GPU bicubic + unsharp mask; see `aivatar-documentation/Other Docs/output_upscaling.md`.
- `streaming/state_manager.py` handles transitions between live generation and idle video loops.

## Scripts & Model Assets

- `scripts/download_models.sh` downloads FlashHead Lite (~6.11GB) and Wav2Vec2-base-960h (~360MB) from Hugging Face.
- Modal image build (`modal_worker/image.py`) bakes the GPU container image; retired RunPod build helper lives in `_archived/runpod-sideline/`.
- `models/SoulX-FlashHead-1_3B/` contains the FlashHead Lite model weights, VAE, and config.
- `models/wav2vec2-base-960h/` contains the Wav2Vec2 audio encoder for feature extraction.
- `SoulX-FlashHead/flash_head/` contains the core inference code (trimmed to runtime essentials only).

## Tests

- Manual testing via dashboard and API calls.
- End-to-end testing via LiveKit room connections.

## License

Private – AiVatar Project.
