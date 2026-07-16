"""Idle video generation entrypoint for asynchronous, avatar-specific idle clips.

This service deliberately has no LiveKit or WebSocket responsibilities. The
backend grants each request a source GET URL, an exact output PUT URL, and a
short-lived callback token. It never receives long-lived R2 credentials.

When used with Modal (modal_idle_video_generator.py), the pipeline is loaded
once at container boot and set via set_pipeline() for CPU memory snapshot reuse.
When _PIPELINE is None (standalone/Vast Docker mode), it falls back to calling
get_pipeline() on each request.
"""
import asyncio
import hashlib
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "SoulX-FlashHead"))

import imageio.v2 as imageio
import numpy as np
import requests
from aiohttp import web

from flash_head.inference import get_audio_embedding, get_infer_params, get_pipeline, get_base_data, run_pipeline


MAX_SOURCE_IMAGE_BYTES = 10 * 1024 * 1024
GENERATION_LOCK = asyncio.Lock()

_PIPELINE = None


def set_pipeline(pipeline):
    global _PIPELINE
    _PIPELINE = pipeline


def _download_source(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    total = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            with requests.get(
                url,
                timeout=(10, 30),
                stream=True,
                headers={"Referer": "https://facemode.io"},
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_SOURCE_IMAGE_BYTES:
                        raise ValueError("source image exceeds size limit")
                    handle.write(chunk)
        return path
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


def _generate_idle_clip(source_path: str, output_path: str, duration_seconds: float) -> tuple[int, int, int, int]:
    if _PIPELINE is not None:
        pipeline = _PIPELINE
    else:
        ckpt_dir = os.getenv("FLASHHEAD_CKPT_DIR", "/app/models/SoulX-FlashHead-1_3B")
        wav2vec_dir = os.getenv("WAV2VEC_DIR", "/app/models/wav2vec2-base-960h")
        pipeline = get_pipeline(1, ckpt_dir, "lite", wav2vec_dir)
    get_base_data(pipeline, source_path, base_seed=42, use_face_crop=False)
    params = get_infer_params()
    fps = int(params["tgt_fps"])
    width = int(params["width"])
    height = int(params["height"])
    cached_duration = int(params["cached_audio_duration"])
    sample_rate = int(params["sample_rate"])
    audio_end_idx = cached_duration * fps
    audio_start_idx = audio_end_idx - int(params["frame_num"])
    target_frames = max(fps, round(duration_seconds * fps))
    silent_audio = np.zeros(cached_duration * sample_rate, dtype=np.float32)
    frames = []

    while len(frames) < target_frames:
        embedding = get_audio_embedding(pipeline, silent_audio, audio_start_idx, audio_end_idx)
        video = run_pipeline(pipeline, embedding)[int(params["motion_frames_num"]):]
        frames.extend(frame.cpu().numpy().astype(np.uint8) for frame in video)

    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_log_level="error",
    ) as writer:
        for frame in frames[:target_frames]:
            writer.append_data(frame)

    return width, height, fps, round(target_frames * 1000 / fps)


def _upload_and_complete(payload: dict):
    source_path = None
    output_path = None
    try:
        source_path = _download_source(payload["sourceGetUrl"])
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        duration_seconds = float(payload.get("durationSeconds", 4))
        width, height, fps, duration_ms = _generate_idle_clip(source_path, output_path, duration_seconds)
        output_bytes = Path(output_path).read_bytes()
        sha256 = hashlib.sha256(output_bytes).hexdigest()

        upload = requests.put(
            payload["outputPutUrl"],
            data=output_bytes,
            headers={
                "Content-Type": "video/mp4",
                "Cache-Control": payload["outputCacheControl"],
            },
            timeout=(10, 60),
        )
        upload.raise_for_status()
        callback = requests.post(
            payload["callbackUrl"],
            headers={"Authorization": f"Bearer {payload['callbackToken']}"},
            json={
                "outputKey": payload["outputKey"],
                "sha256": sha256,
                "width": width,
                "height": height,
                "fps": fps,
                "durationMs": duration_ms,
            },
            timeout=(10, 30),
        )
        callback.raise_for_status()
    except Exception as exc:
        callback_url = payload.get("callbackUrl")
        callback_token = payload.get("callbackToken")
        if callback_url and callback_token:
            requests.post(
                callback_url.replace("/complete", "/failed"),
                headers={"Authorization": f"Bearer {callback_token}"},
                json={"error": str(exc)[:1000]},
                timeout=(10, 30),
            )
        raise
    finally:
        if source_path:
            Path(source_path).unlink(missing_ok=True)
        if output_path:
            Path(output_path).unlink(missing_ok=True)


async def generate(request: web.Request):
    payload = await request.json()
    required = {
        "sourceGetUrl",
        "outputPutUrl",
        "outputKey",
        "outputCacheControl",
        "callbackUrl",
        "callbackToken",
    }
    if not required.issubset(payload):
        return web.json_response({"error": "missing generator payload fields"}, status=400)

    async with GENERATION_LOCK:
        try:
            await asyncio.to_thread(_upload_and_complete, payload)
            return web.json_response({"ok": True, "jobId": payload.get("jobId")})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)


async def healthz(request: web.Request):
    return web.json_response({"status": "ok"})


async def readyz(request: web.Request):
    ready = _PIPELINE is not None
    return web.json_response({"ready": ready, "pipeline_loaded": ready}, status=200 if ready else 503)


app = web.Application(client_max_size=1024 * 1024)
app.router.add_get("/healthz", healthz)
app.router.add_get("/readyz", readyz)
app.router.add_post("/generate", generate)


if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
