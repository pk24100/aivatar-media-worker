"""Idle video generation entrypoint for asynchronous, avatar-specific idle clips.

This service deliberately has no LiveKit or WebSocket responsibilities. The
backend grants each request a source GET URL, an exact output PUT URL, and a
short-lived callback token. It never receives long-lived R2 credentials.

When used with Modal (modal_idle_video_generator.py), the pipeline is loaded
once at container boot and set via set_pipeline() for CPU memory snapshot reuse.
When _PIPELINE is None (standalone mode), it falls back to calling
get_pipeline() on each request.
"""
import asyncio
import hashlib
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "SoulX-FlashHead"))

import cv2
import imageio.v2 as imageio
import numpy as np
import requests
import torch
from aiohttp import web

from flash_head.inference import get_audio_embedding, get_infer_params, get_pipeline, get_base_data, run_pipeline


MAX_SOURCE_IMAGE_BYTES = 10 * 1024 * 1024
GENERATION_LOCK = asyncio.Lock()

_PIPELINE = None


def _apply_2d_sway(frames, fps,
                    sway_pixels=3.0, sway_rate_hz=0.15):
    """Apply head sway as 2D post-processing.

    Operates in pixel space using cv2 affine transforms.
    Avatar-agnostic: produces identical visible motion regardless
    of face type (real human, animated, custom).
    Eye blinking is handled by FlashHead model via murmuring audio.
    """
    n = len(frames)
    if n == 0:
        return frames

    h, w = frames[0].shape[:2]
    processed = []
    for i, frame in enumerate(frames):
        t = i / fps

        # Head sway: smooth sinusoidal translation + slight rotation
        dx = sway_pixels * np.sin(2.0 * np.pi * sway_rate_hz * t)
        dy = sway_pixels * 0.4 * np.sin(2.0 * np.pi * sway_rate_hz * 0.7 * t + 0.5)
        angle = 0.3 * np.sin(2.0 * np.pi * sway_rate_hz * 0.5 * t)

        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        sway_frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        processed.append(sway_frame)

    return processed


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
    audio_samples = cached_duration * sample_rate
    murmur_amplitude = float(os.getenv("IDLE_MURMUR_AMPLITUDE", "0.03"))
    motion_perturbation = float(os.getenv("IDLE_MOTION_PERTURBATION", "0.04"))
    rng = np.random.default_rng(42)
    torch_rng = torch.Generator(device=pipeline.device).manual_seed(42)
    frames = []
    slice_idx = 0

    # Generate video with FlashHead using low-amplitude murmuring audio.
    # The murmuring audio (glottal pulses + formants) produces wav2vec2
    # embeddings that drive FlashHead's natural eye blink generation at
    # the correct facial positions. A small latent perturbation prevents
    # autoregressive convergence so motion continues across all 15 seconds.
    # 2D sway is applied as post-processing for consistent body movement.
    while len(frames) < target_frames:
        if slice_idx == 0:
            # First slice: silence audio to avoid initial lip movement.
            # The model's first generation is most sensitive to audio
            # embeddings - murmuring here causes visible lip motion.
            idle_audio = np.zeros(audio_samples, dtype=np.float32)
        else:
            t = np.arange(audio_samples, dtype=np.float32) / sample_rate
            slice_offset = slice_idx * (int(params["frame_num"]) - int(params["motion_frames_num"])) / fps

            f0 = 150.0
            voicing = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.15 * (t + slice_offset))
            glottal = np.sign(np.sin(2.0 * np.pi * f0 * (t + slice_offset))) * voicing
            formant1 = np.sin(2.0 * np.pi * 700.0 * (t + slice_offset)) * 0.3
            formant2 = np.sin(2.0 * np.pi * 1200.0 * (t + slice_offset)) * 0.2
            murmur = murmur_amplitude * (glottal * 0.5 + formant1 + formant2).astype(np.float32)
            murmur *= (0.6 + 0.4 * np.sin(2.0 * np.pi * 0.08 * (t + slice_offset))).astype(np.float32)
            murmur += (0.01 * rng.standard_normal(audio_samples)).astype(np.float32)
            idle_audio = murmur.astype(np.float32)

        embedding = get_audio_embedding(pipeline, idle_audio, audio_start_idx, audio_end_idx)

        video = run_pipeline(pipeline, embedding)[int(params["motion_frames_num"]):]
        frames.extend(frame.cpu().numpy().astype(np.uint8) for frame in video)

        # Small latent perturbation to prevent autoregressive convergence.
        if motion_perturbation > 0 and slice_idx > 0:
            lmf = pipeline.latent_motion_frames
            lmf_std = lmf.std()
            if lmf_std > 0:
                perturbation = torch.randn(
                    lmf.shape, generator=torch_rng,
                    device=lmf.device, dtype=lmf.dtype,
                ) * (motion_perturbation * lmf_std)
                pipeline.latent_motion_frames = lmf + perturbation

        slice_idx += 1

    # Apply 2D post-processing: head sway only.
    # Eye blinking comes from FlashHead model via murmuring audio.
    sway_pixels = float(os.getenv("IDLE_SWAY_PIXELS", "3.0"))
    sway_rate_hz = float(os.getenv("IDLE_SWAY_RATE_HZ", "0.15"))

    frames = _apply_2d_sway(
        frames[:target_frames], fps,
        sway_pixels=sway_pixels,
        sway_rate_hz=sway_rate_hz,
    )

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
        duration_seconds = float(payload.get("durationSeconds", 15))
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
