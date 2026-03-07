# AI Avatar Media Worker

## Overview

The media worker runs Ditto-based avatar inference and feeds the resulting video into LiveKit rooms. It supports both offline (audio + image → mp4 → LiveKit) and streaming sessions and is packaged for Runpod Serverless deployments.

## Layout

```
aivatar-media-worker/
├── handler.py                # Runpod entry handler
├── entrypoint.sh            # container entrypoint
├── Dockerfile               # GPU-enabled container for Runpod
├── requirements.txt         # Python deps
├── livekit_test.py          # Local LiveKit publishing smoke test
├── test_runpod_endpoint.py   # Runpod endpoint exerciser
├── scripts/                 # Build + deployment helpers
├── streaming/               # LiveKit session logic
├── utils/                   # Ditto runner + LiveKit publishing helpers
├── models/                  # Ditto weights/config (ditto subdirectories)
├── ditto-talkinghead/       # upstream Ditto repo (core, scripts, example)
└── .env                     # runtime config (should stay private)
```

## Requirements

- Python 3.11+ (or 3.10 with backports); `pip install -r requirements.txt`.
- CUDA 12+ GPU (NVIDIA) for TensorRT weights; fallback to ONNX/PyTorch subdirectories when TenorFlow is not available.
- `runpod` client for Serverless handler (`runpod==0.17` is pinned transitively).

## Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` (copy from `.env.example` or keep `.env` private) and set:
   - `MODEL_ROOT` (defaults to `/app/models/ditto`)
   - `DITTO_REPO_PATH` (defaults to `/app/ditto-talkinghead`)
   - `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
   - `LIVEKIT_ROOM`, `LIVEKIT_TOKEN` when exercising streaming locally
   - `RUNPOD_KEY` if you invoke Runpod APIs from helper scripts
3. Build or provision the Ditto model bundles under `models/ditto/<variant>` before running inference. Use `scripts/convert_trt_ada.sh` to rebuild CUDA-accelerated bundles if needed.

## Running

- `python handler.py` (also invoked by `entrypoint.sh` / Dockerfile) boots the Runpod Serverless workflow.
  - `_verify_models()` inspects `MODEL_ROOT/ditto_cfg` and chooses the best TRT/ONNX/PyTorch directory.
  - Offline mode requires `audioPath` + `sourceImage`; it writes a temp mp4 and calls `utils.livekit_publisher.publish_video_to_livekit`.
  - Streaming mode requires `roomName`, `livekitToken`, `sourceImage`, and `LIVEKIT_URL`; it runs `streaming.stream_processor.run_streaming_session`.
- `scripts/build_on_runpod.sh` packages the repo + models for Runpod artifacts.
- `scripts/runpod_serverless_setup.ps1` / `runpod_endpoint_test.ps1` are PowerShell helpers for Azure/Runpod pipelines.
- Use `livekit_test.py` or `streaming/ditto_streaming.py` for manual end-to-end verification.

## Streaming & Utilities

- `streaming/stream_processor.py` orchestrates LiveKit connections, audio chunking, and Ditto inference loops.
- `streaming/video_publisher.py`, `audio_chunker.py`, and `audio_subscriber.py` separate media-side logic for LiveKit clients.
- `utils/ditto_runner.py` wraps inference commands (audio + image → video) and exposes `run_ditto_inference`.
- `utils/livekit_publisher.py` publishes generated mp4s into LiveKit rooms via the SDK or REST bridge.

## Scripts & Model Assets

- `scripts/convert_trt_ada.sh` regenerates TensorRT-optimized weights under `models/ditto/ditto_trt_ada`.
- `scripts/build_on_runpod.sh` builds the GPU container, zips the bundle, and uploads it to Runpod storage.
- `scripts/runpod_serverless_setup.ps1` and `runpod_endpoint_test.ps1` orchestrate deployment tests on Windows/build pipelines.
- `models/ditto/` mirrors the Ditto repo’s expected layout; drop your `.onnx`, `.pt`, or `.plan` files into the appropriate subfolder.
- `ditto-talkinghead/` contains the original Ditto inference scripts and can be referenced if additional tooling or examples are required; stay on the pinned commit (`environment.yaml` lists the pinned versions).

## Tests

- `livekit_test.py` publishes a hardcoded synthetic clip to a LiveKit room (requires env vars).
- `test_runpod_endpoint.py` exercises `handler.py` by simulating Runpod input, so keep it trimmed to your latest endpoints.

## License

Private – AiVatar Project.
