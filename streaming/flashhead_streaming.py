# Core streaming engine that drives the FlashHead model to generate lip-synced avatar video frames from audio.
import os
import sys
import numpy as np
import logging
import time
import uuid
import tempfile
from queue import Queue
from urllib.parse import urlparse

import boto3
import requests
import socket
import ipaddress
from collections import deque

from utils.default_avatar_cache import default_avatar_cache

# Add SoulX-FlashHead to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "SoulX-FlashHead"))
from flash_head.inference import get_audio_embedding, run_pipeline, get_infer_params

logger = logging.getLogger("FlashHeadStreamingEngine")

MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

_ALLOWED_IMAGE_DOMAINS = [
    d.strip().lower()
    for d in os.environ.get("ALLOWED_IMAGE_DOMAINS", "").split(",")
    if d.strip()
]

_IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",
}


def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except ValueError:
        return True


def _validate_image_url(parsed):
    hostname = parsed.hostname or ""
    hostname_lower = hostname.lower()

    if _ALLOWED_IMAGE_DOMAINS and hostname_lower not in _ALLOWED_IMAGE_DOMAINS:
        raise ValueError(
            f"Image URL domain '{hostname_lower}' not in allowlist. "
            f"Allowed: {_ALLOWED_IMAGE_DOMAINS}"
        )

    try:
        resolved = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in resolved:
            ip = sockaddr[0]
            if _is_private_ip(ip):
                raise ValueError(
                    f"Image URL resolves to private/internal IP {ip}. "
                    f"SSRF prevention: rejecting."
                )
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {hostname}")

    if os.environ.get("NODE_ENV", "").lower() == "production" and parsed.scheme != "https":
        raise ValueError("Only HTTPS URLs are allowed for image downloads in production")


def _is_valid_image_bytes(data: bytes) -> bool:
    if len(data) < 12:
        return False
    for magic, fmt in _IMAGE_MAGIC_BYTES.items():
        if data.startswith(magic):
            if fmt == "webp" and data[8:12] != b"WEBP":
                continue
            return True
    return False


# Streaming engine that generates lip-synced video frames from audio.
class FlashHeadStreamingEngine:
    # Initialize engine with pipeline, avatar path, and inference params.
    def __init__(
        self,
        pipeline,
        avatar_image_path: str = None,
        seed: int = 42,
        auto_prepare_avatar: bool = True,
        frame_queue: Queue = None,
        audio_queue: Queue = None,
    ):
        self.pipeline = pipeline
        self.seed = seed
        self.infer_params = get_infer_params()
        self.frame_num = self.infer_params["frame_num"]
        self.motion_frames_num = self.infer_params["motion_frames_num"]
        self.slice_len = self.frame_num - self.motion_frames_num
        self.sample_rate = self.infer_params["sample_rate"]
        self.tgt_fps = self.infer_params["tgt_fps"]
        self.cached_audio_duration = self.infer_params["cached_audio_duration"]
        self.height = self.infer_params["height"]
        self.width = self.infer_params["width"]
        self.sample_steps = self.infer_params["sample_steps"]
        self.slice_samples = self.slice_len * self.sample_rate // self.tgt_fps
        self.cached_audio_samples = self.cached_audio_duration * self.sample_rate
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num
        self.audio_context = deque([0.0] * self.cached_audio_samples, maxlen=self.cached_audio_samples)
        self.pending_audio = deque()
        self.frame_queue = frame_queue or Queue()
        self.audio_queue = audio_queue or Queue()
        self._prepared = False
        self._temp_avatar_path = None
        self._metrics_started_at = time.monotonic()
        self._metrics_input_samples = 0
        self._metrics_slices = 0
        self._metrics_inference_total_ms = 0.0
        self._metrics_inference_max_ms = 0.0

        if avatar_image_path and auto_prepare_avatar:
            self.prepare_avatar(avatar_image_path)

    # Prepare the pipeline with the avatar image for inference.
    def prepare_avatar(self, avatar_image_path: str, use_face_crop: bool = False):
        avatar_path = self._resolve_avatar_path(avatar_image_path)
        self.pipeline.prepare_params(
            cond_image_path_or_dir=avatar_path,
            target_size=(self.height, self.width),
            frame_num=self.frame_num,
            motion_frames_num=self.motion_frames_num,
            sampling_steps=self.sample_steps,
            seed=self.seed,
            shift=self.infer_params["sample_shift"],
            color_correction_strength=self.infer_params["color_correction_strength"],
            use_face_crop=use_face_crop,
        )
        self._prepared = True

    # Resolve avatar URL/path to a local file, downloading if needed.
    def _resolve_avatar_path(self, avatar_image_path: str) -> str:
        cached_avatar_path = default_avatar_cache.get_cached_path(avatar_image_path)
        if cached_avatar_path:
            logger.info("Using cached default avatar for %s -> %s", avatar_image_path, cached_avatar_path)
            return cached_avatar_path

        parsed = urlparse(avatar_image_path)
        if parsed.scheme in ("http", "https"):
            _validate_image_url(parsed)
            with requests.get(avatar_image_path, timeout=(10, 30), stream=True, headers={"Referer": "https://facemode.io"}) as response:
                response.raise_for_status()

                content_length = int(response.headers.get("Content-Length", 0))
                if content_length > MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Avatar image exceeds {MAX_DOWNLOAD_BYTES} bytes (got {content_length})"
                    )

                suffix = os.path.splitext(parsed.path)[1] or ".jpg"
                temp_path = os.path.join(tempfile.gettempdir(), f"avatar_{uuid.uuid4().hex}{suffix}")
                downloaded = b""
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += chunk
                    if len(downloaded) > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"Avatar image exceeded {MAX_DOWNLOAD_BYTES} bytes during download"
                        )

            if not _is_valid_image_bytes(downloaded):
                raise ValueError("Downloaded content is not a valid image (JPEG/PNG/WebP)")

            with open(temp_path, "wb") as handle:
                handle.write(downloaded)
            self._temp_avatar_path = temp_path
            return temp_path
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            if not bucket or not key:
                raise ValueError(f"Invalid S3 sourceImage: {avatar_image_path}")
            suffix = os.path.splitext(key)[1] or ".png"
            temp_path = os.path.join(tempfile.gettempdir(), f"avatar_{uuid.uuid4().hex}{suffix}")
            boto3.client("s3").download_file(bucket, key, temp_path)
            self._temp_avatar_path = temp_path
            return temp_path
        if parsed.scheme == "file":
            avatar_image_path = parsed.path
        elif parsed.scheme:
            raise ValueError(f"Unsupported sourceImage scheme: {parsed.scheme}")
        if not os.path.exists(avatar_image_path):
            raise FileNotFoundError(f"Avatar image not found: {avatar_image_path}")
        return avatar_image_path

    # Process buffered audio slices into video frames via FlashHead.
    def _process_available_audio(self):
        # Real-time pacing: each slice represents slice_len frames at tgt_fps,
        # i.e. slice_len / tgt_fps seconds of real-time content.  We measure
        # from the START of the previous slice, not the end, so that inference
        # time (~0.75s) overlaps with the wait period.  This gives a total
        # cycle of max(slice_realtime, inference_time) instead of
        # slice_realtime + inference_time, keeping production matched to
        # consumption at 25 fps.
        slice_realtime = self.slice_len / float(self.tgt_fps)  # e.g. 24/25 = 0.96s
        while len(self.pending_audio) >= self.slice_samples:
            now = time.monotonic()
            if hasattr(self, '_last_slice_time') and (now - self._last_slice_time) < slice_realtime:
                # Not enough real-time has elapsed since the last slice STARTED.
                # Wait for the next run_chunk call to try again.
                return
            # Stamp the start time BEFORE inference so the wait period
            # overlaps with inference time.
            self._last_slice_time = now
            human_speech_array = np.array(
                [self.pending_audio.popleft() for _ in range(self.slice_samples)],
                dtype=np.float32,
            )
            self.audio_context.extend(human_speech_array.tolist())
            audio_embedding = get_audio_embedding(
                self.pipeline,
                np.array(self.audio_context, dtype=np.float32),
                self.audio_start_idx,
                self.audio_end_idx,
            )
            _t0 = time.monotonic()
            video = run_pipeline(self.pipeline, audio_embedding)
            _infer_ms = round((time.monotonic() - _t0) * 1000, 1)
            video = video[self.motion_frames_num:]
            _n_frames = video.shape[0]
            for i in range(video.shape[0]):
                self.frame_queue.put_nowait(video[i].cpu().numpy().astype(np.uint8))
            self.audio_queue.put_nowait(human_speech_array)
            self._metrics_slices += 1
            self._metrics_inference_total_ms += _infer_ms
            self._metrics_inference_max_ms = max(self._metrics_inference_max_ms, _infer_ms)
            self._log_metrics()

    def _log_metrics(self, force: bool = False):
        elapsed = time.monotonic() - self._metrics_started_at
        if not force and elapsed < 5.0:
            return
        if elapsed <= 0:
            return

        average_inference_ms = (
            self._metrics_inference_total_ms / self._metrics_slices
            if self._metrics_slices else 0.0
        )
        logger.info(
            "ENGINE_METRICS windowMs=%.0f inputSamples=%d inputRateHz=%.0f "
            "pendingSamples=%d pendingSeconds=%.2f slices=%d avgInferenceMs=%.1f "
            "maxInferenceMs=%.1f frameQueue=%d audioQueue=%d",
            elapsed * 1000,
            self._metrics_input_samples,
            self._metrics_input_samples / elapsed,
            len(self.pending_audio),
            len(self.pending_audio) / float(self.sample_rate),
            self._metrics_slices,
            average_inference_ms,
            self._metrics_inference_max_ms,
            self.frame_queue.qsize(),
            self.audio_queue.qsize(),
        )
        self._metrics_started_at = time.monotonic()
        self._metrics_input_samples = 0
        self._metrics_slices = 0
        self._metrics_inference_total_ms = 0.0
        self._metrics_inference_max_ms = 0.0

    # Feed an audio chunk into the engine and trigger frame generation.
    def run_chunk(self, audio_data: np.ndarray):
        if not self._prepared:
            raise RuntimeError("Avatar not prepared. Call prepare_avatar() first.")
        audio_array = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        self.pending_audio.extend(audio_array.tolist())
        self._metrics_input_samples += len(audio_array)
        self._log_metrics()
        self._process_available_audio()

    # Pad and process one final real-time slice. Returns whether audio remains.
    def flush(self) -> bool:
        if self.pending_audio:
            pad = (-len(self.pending_audio)) % self.slice_samples
            if pad:
                self.pending_audio.extend([0.0] * pad)
            self._process_available_audio()
        return bool(self.pending_audio)

    # Return the delay before the next slice can be generated at the target FPS.
    def next_slice_delay(self) -> float:
        if not hasattr(self, '_last_slice_time'):
            return 0.0
        slice_realtime = self.slice_len / float(self.tgt_fps)
        return max(0.0, slice_realtime - (time.monotonic() - self._last_slice_time))

    def next_live_slice_delay(self):
        """Return the next generation deadline only when a full live slice is buffered."""
        if len(self.pending_audio) < self.slice_samples:
            return None
        return self.next_slice_delay()

    def process_pending_audio(self):
        """Generate a due live slice even when no new WebSocket packet arrives."""
        self._process_available_audio()

    # Clear queues and clean up temporary avatar files.
    def close(self):
        self._log_metrics(force=True)
        self.pending_audio.clear()
        self.audio_context.clear()
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
        if self._temp_avatar_path and os.path.exists(self._temp_avatar_path):
            os.remove(self._temp_avatar_path)
        self._temp_avatar_path = None
        self._prepared = False
        logger.info("FlashHeadStreamingEngine closed")
