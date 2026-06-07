import os
import sys
import numpy as np
import logging
import uuid
import tempfile
from queue import Queue
from urllib.parse import urlparse

import boto3
import requests
from collections import deque

# Add SoulX-FlashHead to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "SoulX-FlashHead"))
from flash_head.inference import get_audio_embedding, run_pipeline, get_infer_params

logger = logging.getLogger("FlashHeadStreamingEngine")


# Streaming engine that generates lip-synced video frames from audio.
class FlashHeadStreamingEngine:
    # Initialize engine with pipeline, avatar path, and inference params.
    def __init__(self, pipeline, avatar_image_path: str = None, seed: int = 42):
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
        self.frame_queue = Queue()
        self.audio_queue = Queue()
        self._prepared = False
        self._temp_avatar_path = None

        if avatar_image_path:
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
        parsed = urlparse(avatar_image_path)
        if parsed.scheme in ("http", "https"):
            response = requests.get(avatar_image_path, timeout=30)
            response.raise_for_status()
            suffix = os.path.splitext(parsed.path)[1] or ".png"
            temp_path = os.path.join(tempfile.gettempdir(), f"avatar_{uuid.uuid4().hex}{suffix}")
            with open(temp_path, "wb") as handle:
                handle.write(response.content)
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
        while len(self.pending_audio) >= self.slice_samples:
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
            video = run_pipeline(self.pipeline, audio_embedding)
            video = video[self.motion_frames_num:]
            for i in range(video.shape[0]):
                self.frame_queue.put_nowait(video[i].cpu().numpy().astype(np.uint8))
            # Pair the audio slice with its video frames so the publisher
            # can emit them together for lip-sync.
            self.audio_queue.put_nowait(human_speech_array)

    # Feed an audio chunk into the engine and trigger frame generation.
    def run_chunk(self, audio_data: np.ndarray):
        if not self._prepared:
            raise RuntimeError("Avatar not prepared. Call prepare_avatar() first.")
        audio_array = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        self.pending_audio.extend(audio_array.tolist())
        self._process_available_audio()

    # Pad and process any remaining buffered audio.
    def flush(self):
        if self.pending_audio:
            pad = (-len(self.pending_audio)) % self.slice_samples
            if pad:
                self.pending_audio.extend([0.0] * pad)
            self._process_available_audio()

    # Clear queues and clean up temporary avatar files.
    def close(self):
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
