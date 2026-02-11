import asyncio
import os
import tempfile

import runpod
import torch

from streaming.stream_processor import run_streaming_session
from utils.ditto_runner import run_ditto_inference
from utils.livekit_publisher import publish_video_to_livekit

MODEL_ROOT = os.getenv("MODEL_ROOT", "/app/models/ditto")
DITTO_REPO_PATH = os.getenv("DITTO_REPO_PATH", "/app/ditto-talkinghead")


def _verify_models():
    cfg = os.path.join(MODEL_ROOT, "ditto_cfg")
    if not os.path.isdir(cfg):
        raise FileNotFoundError(f"Model config not found at {cfg}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) >= (8, 9):
            preferred = "ditto_trt_ada"
        else:
            preferred = "ditto_trt_Ampere_Plus"
        preferred_path = os.path.join(MODEL_ROOT, preferred)
        if os.path.isdir(preferred_path) and len(os.listdir(preferred_path)) > 0:
            print(f"Using data_root: {preferred_path}")
            return preferred_path

    for candidate in ["ditto_trt_Ampere_Plus", "ditto_trt_ada", "ditto_trt_custom", "ditto_onnx", "ditto_pytorch"]:
        path = os.path.join(MODEL_ROOT, candidate)
        if os.path.isdir(path) and len(os.listdir(path)) > 0:
            print(f"Using data_root: {path}")
            return path
    raise FileNotFoundError("No Ditto model directory found (checked baked-in /app/models/ditto)")


DATA_ROOT = _verify_models()


def _is_streaming(event):
    if event.get("streaming") is True:
        return True
    if str(event.get("mode", "")).lower() == "streaming":
        return True
    env_flag = os.getenv("AIVATAR_STREAMING", "").strip().lower()
    return env_flag in {"1", "true", "yes"}

def handler(job):
    """
    Called per request by Runpod Serverless.
    job["input"] example:
    {
      "roomName": "room_abc",
      "livekitToken": "...",
      "audioPath": "s3://bucket/audio.wav",
      "sourceImage": "s3://bucket/avatar.png"
    }
    """
    event = job["input"]

    roomName = event.get("roomName")
    livekitToken = event.get("livekitToken")
    audioPath = event.get("audioPath")
    sourceImage = event.get("sourceImage")

    if _is_streaming(event):
        if not roomName or not livekitToken:
            raise ValueError("roomName and livekitToken are required for streaming mode")
        if not sourceImage:
            raise ValueError("sourceImage is required for streaming mode")
        livekit_url = os.getenv("LIVEKIT_URL")
        if not livekit_url:
            raise ValueError("LIVEKIT_URL is required for streaming mode")
        asyncio.run(
            run_streaming_session(
                room_name=roomName,
                livekit_token=livekitToken,
                livekit_url=livekit_url,
                model_root=MODEL_ROOT,
                source_image=sourceImage,
            )
        )
        return {"status": "ok", "mode": "streaming"}

    if not audioPath:
        raise ValueError("audioPath is required for offline inference")
    if not sourceImage:
        raise ValueError("sourceImage is required for offline inference")

    # produce a temporary mp4 path
    out_mp4 = tempfile.mktemp(suffix=".mp4")

    # 1. run Ditto inference and generate an mp4
    run_ditto_inference(
        MODEL_ROOT,
        audioPath,
        sourceImage,
        out_mp4
    )

    # 2. publish the video into the LiveKit room
    publish_video_to_livekit(
        out_mp4,
        roomName,
        livekitToken,
        os.getenv("LIVEKIT_URL")
    )

    return {"status": "ok", "output": out_mp4}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
