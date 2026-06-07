# AI Avatar Media Worker

## Overview

The media worker runs **SoulX-FlashHead Lite** avatar inference and feeds the resulting video into LiveKit rooms. It supports **streaming-only** sessions and is packaged for RunPod Serverless deployments.

## Layout

```
aivatar-media-worker/
├── handler.py                # RunPod entry handler
├── entrypoint.sh            # container entrypoint
├── Dockerfile               # GPU-enabled container for RunPod
├── requirements.txt         # Python deps
├── scripts/                 # Build + deployment helpers
│   ├── build_on_runpod.sh
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
- `runpod` client for Serverless handler (`runpod>=1.6.2`).

## Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` (copy from `.env.example` or keep `.env` private) and set:
   - `LIVEKIT_URL`
   - optionally `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` if your surrounding workflow needs them
   - optionally `HUGGING_FACE_HUB_TOKEN` / `HF_TOKEN` for private cached-model access outside the RunPod-managed model cache
3. Download models using `scripts/download_models.sh` before building Docker image.

The worker defaults internally to:

- cached FlashHead repo `pkam24100/aivatar-flashhead-model`
- Hugging Face cache root `/runpod-volume/huggingface-cache/hub` on RunPod
- baked wav2vec2 path `/app/models/wav2vec2-base-960h`
- legacy baked FlashHead fallback `/app/models/SoulX-FlashHead-1_3B`

## Running

- `python handler.py` (also invoked by `entrypoint.sh` / Dockerfile) boots the RunPod Serverless workflow.
  - `_verify_models()` checks that FlashHead and Wav2Vec2 model directories exist.
  - **Streaming-only mode** requires `roomName`, `livekitToken`, `sourceImage`, and `LIVEKIT_URL`; it runs `streaming.stream_processor.run_streaming_session`.
  - The worker maintains a `MODEL_POOL` of 3 concurrent FlashHead instances for parallel stream processing.
- `scripts/build_on_runpod.sh` builds and pushes the Docker image to Docker Hub.
- `scripts/download_models.sh` downloads `wav2vec2-base-960h` by default and supports `DOWNLOAD_FLASHHEAD=1` for an explicit local FlashHead download.

## Streaming & Utilities

- `streaming/stream_processor.py` orchestrates LiveKit connections and FlashHead streaming sessions.
- `streaming/flashhead_streaming.py` contains `FlashHeadStreamingEngine` for real-time video generation from audio chunks.
- `streaming/websocket_server.py` handles server-to-server WebSocket audio ingestion with token authentication.
- `streaming/audio_subscriber.py` and `sip_handler.py` handle LiveKit audio subscription and SIP ingestion.
- `utils/model_pool.py` manages a pool of 3 FlashHead pipeline instances for concurrent stream processing.
- `streaming/video_publisher.py` publishes generated video frames to LiveKit rooms.
- `streaming/state_manager.py` handles transitions between live generation and idle video loops.

## Scripts & Model Assets

- `scripts/download_models.sh` downloads FlashHead Lite (~6.11GB) and Wav2Vec2-base-960h (~360MB) from Hugging Face.
- `scripts/build_on_runpod.sh` builds the GPU container and pushes to Docker Hub.
- `models/SoulX-FlashHead-1_3B/` contains the FlashHead Lite model weights, VAE, and config.
- `models/wav2vec2-base-960h/` contains the Wav2Vec2 audio encoder for feature extraction.
- `SoulX-FlashHead/flash_head/` contains the core inference code (trimmed to runtime essentials only).

## Tests

- Manual testing via dashboard and API calls.
- End-to-end testing via LiveKit room connections.

## License

Private – AiVatar Project.
