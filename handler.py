import asyncio
import os
import tempfile

import runpod
import torch

from streaming.stream_processor import run_streaming_session
from streaming.websocket_server import ws_server
from utils.model_pool import DittoModelPool

MODEL_ROOT = os.getenv("MODEL_ROOT", "/app/models/ditto")
DITTO_REPO_PATH = os.getenv("DITTO_REPO_PATH", "/app/ditto-talkinghead")

# Initialize model pool globally so it happens during FlashBoot
model_pool = DittoModelPool(pool_size=3, model_root=MODEL_ROOT)
_ws_started = False

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

async def handler_async(job):
    """
    Called per request by Runpod Serverless.
    job["input"] example:
    {
      "roomName": "room_abc",
      "livekitToken": "...",
      "audioPath": "s3://bucket/audio.wav",
      "sourceImage": "s3://bucket/avatar.png",
      "ingestionMethod": "websocket",
      "sessionId": "1234"
    }
    """
    global _ws_started
    if not _ws_started:
        # Start the WebSocket server in the background on the first request
        # In a real production setup, you might want this to start even before the first request,
        # but starting it here ensures it runs within the RunPod asyncio event loop.
        asyncio.create_task(ws_server.start())
        _ws_started = True

    event = job["input"]

    roomName = event.get("roomName")
    livekitToken = event.get("livekitToken")
    audioPath = event.get("audioPath")
    sourceImage = event.get("sourceImage")
    ingestionMethod = event.get("ingestionMethod", "livekit")
    sessionId = event.get("sessionId", roomName)

    if _is_streaming(event):
        if not roomName or not livekitToken:
            raise ValueError("roomName and livekitToken are required for streaming mode")
        if not sourceImage:
            raise ValueError("sourceImage is required for streaming mode")
        livekit_url = event.get("customLivekitUrl") or os.getenv("LIVEKIT_URL")
        if not livekit_url:
            raise ValueError("LIVEKIT_URL is required for streaming mode")
            
        ingestionToken = event.get("ingestionToken", "")

        model_instance = await model_pool.acquire()
        try:
            await run_streaming_session(
                room_name=roomName,
                livekit_token=livekitToken,
                livekit_url=livekit_url,
                model_root=MODEL_ROOT,
                source_image=sourceImage,
                model_instance=model_instance,
                ingestion_method=ingestionMethod,
                session_id=sessionId,
                ingestion_token=ingestionToken,
            )
            return {"status": "ok", "mode": "streaming"}
        finally:
            model_pool.release(model_instance)

    raise NotImplementedError("Offline inference is currently disabled and stored in reference_offline_code")

# def handler(job):
#     return asyncio.run(handler_async(job))


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler_async,
        "return_aggregate_stream": True,
        "concurrency_modifier": lambda x: 3 # Allow up to 3 concurrent jobs
    })
