# Loads and manages idle video frames in memory for seamless looping and crossfade transitions.
import contextlib
import cv2
import numpy as np
import tempfile
import os
import logging
import torch
from typing import List, Optional
from urllib.parse import urlparse

import requests

from streaming.core.upscale import upscale_slice_torch, upscale_frames_numpy, upscale_frame_cv2
from utils.ssrf_fetch import validate_url

logger = logging.getLogger(__name__)
MAX_IDLE_VIDEO_BYTES = int(os.getenv("IDLE_VIDEO_MAX_BYTES", str(20 * 1024 * 1024)))
MAX_IDLE_VIDEO_FRAMES = int(os.getenv("IDLE_VIDEO_MAX_FRAMES", "750"))
MAX_IDLE_VIDEO_DIMENSION = int(os.getenv("IDLE_VIDEO_MAX_DIMENSION", "1024"))

_USE_BEST_MATCH = os.getenv("IDLE_BEST_MATCH", "1") == "1"


def _get_model_native_size(fallback_width: int, fallback_height: int) -> tuple[int, int]:
    """Model's native generation resolution (512x512). Composing the static
    fallback at this size lets it pass through the same GPU upscale as live
    frames, keeping crossfades filter-consistent."""
    try:
        from flash_head.inference import get_infer_params

        params = get_infer_params()
        return int(params["width"]), int(params["height"])
    except Exception as exc:
        logger.warning(
            "Unable to read model native size (%s); composing fallback at output size", exc,
        )
        return fallback_width, fallback_height


def _upscale_fallback_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Run one RGB uint8 frame through the shared GPU upscale path."""
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    try:
        tensor = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda(non_blocking=True)
        tensor = upscale_slice_torch(tensor, width, height)
        return tensor.squeeze(0).cpu().numpy().astype(np.uint8)
    except Exception as exc:
        logger.warning("GPU fallback-frame upscale failed (%s); using cv2", exc)
        return upscale_frame_cv2(frame, width, height)


class IdleVideoLoop:
    """
    Loads and manages idle video frames in RAM for fast access.
    Implements seamless looping and crossfade transitions.
    """
    
    # Load idle video frames into memory for looping playback.
    def __init__(self, idle_video_url: str = None, crossfade_frames: int = 8, video_bytes: bytes = None):
        self.crossfade_frames = crossfade_frames
        self.frames: List[np.ndarray] = []
        self.current_idx = 0
        self.total_frames = 0
        self._loop_crossfade_frames = int(os.getenv("IDLE_LOOP_CROSSFADE_FRAMES", "8"))
        self._loop_transition_buffer: List[np.ndarray] = []
        self._in_loop_transition: bool = False
        self._loop_transition_idx: int = 0
        self._loop_crossfade_frames_effective: int = 0
        self._loop_crossfade_cache: Optional[List[np.ndarray]] = None
        self._loop_crossfade_cache_n: int = 0

        if video_bytes is not None:
            self._load_bytes(video_bytes)
        elif idle_video_url:
            self._load_video(idle_video_url)

    def _load_bytes(self, video_bytes: bytes):
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(video_bytes)
            self._decode_local_video(temp_path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_path)

    @classmethod
    def fallback_from_source_image(cls, source_image: str, width: int, height: int):
        """Create a safe static fallback so a pending idle asset cannot deadlock a session."""
        loop = cls()
        if not source_image:
            return loop
        try:
            parsed = urlparse(source_image)
            if parsed.scheme in ("http", "https"):
                try:
                    validate_url(source_image, "image")
                except Exception as exc:
                    logger.warning("SSRF_WOULD_BLOCK kind=image url=%s err=%s", source_image, exc)
                    if os.getenv("SSRF_ENFORCE", "0").strip() == "1":
                        raise
                    # LOG-ONLY: still proceed with existing download below.
                chunks = []
                total = 0
                with requests.get(
                    source_image,
                    timeout=(5, 30),
                    stream=True,
                    headers={"Referer": "https://facemode.io"},
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_IDLE_VIDEO_BYTES:
                            raise ValueError("source image exceeds fallback size limit")
                        chunks.append(chunk)
                image = cv2.imdecode(np.frombuffer(b"".join(chunks), dtype=np.uint8), cv2.IMREAD_COLOR)
            elif parsed.scheme:
                return loop
            else:
                image = cv2.imread(source_image, cv2.IMREAD_COLOR)
            if image is None:
                return loop
            source_height, source_width = image.shape[:2]
            # Compose the cover-crop at the model's native resolution so the
            # frame can go through the SAME GPU upscale (bicubic + unsharp) as
            # live frames - otherwise the fallback looks sharper/softer than
            # the live stream at the idle-to-live crossfade.
            native_width, native_height = _get_model_native_size(width, height)
            scale = max(native_width / source_width, native_height / source_height)
            resized_width = max(native_width, int(np.ceil(source_width * scale)))
            resized_height = max(native_height, int(np.ceil(source_height * scale)))
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
            left = (resized_width - native_width) // 2
            top = (resized_height - native_height) // 2
            cropped = cv2.cvtColor(
                resized[top:top + native_height, left:left + native_width], cv2.COLOR_BGR2RGB
            )
            frame = _upscale_fallback_frame(cropped, width, height)
            loop.frames = [frame]
            loop.total_frames = 1
            return loop
        except Exception as exc:
            logger.warning("Unable to create fallback idle frame: %s", exc)
            return loop
    
    def _load_video(self, url_or_path: str):
        """Load all frames into RAM."""
        local_path = url_or_path
        
        # Download if it's a URL
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            logger.info("Downloading idle video asset")
            try:
                temp_path = self._download(url_or_path)
                local_path = temp_path
            except Exception as e:
                logger.error(f"Failed to download idle video: {e}")
                # Fallback to empty if failed
                self.frames = []
                return

        self._decode_local_video(local_path)

        if local_path != url_or_path:
            with contextlib.suppress(OSError):
                os.remove(local_path)

    @staticmethod
    def _download(url: str) -> str:
        try:
            validate_url(url, "video")
        except Exception as exc:
            logger.warning("SSRF_WOULD_BLOCK kind=video url=%s err=%s", url, exc)
            if os.getenv("SSRF_ENFORCE", "0").strip() == "1":
                raise
            # LOG-ONLY: still proceed with existing download below.
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        total = 0
        try:
            with os.fdopen(fd, "wb") as handle:
                with requests.get(url, timeout=(5, 30), stream=True) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_IDLE_VIDEO_BYTES:
                            raise ValueError(f"idle video exceeds {MAX_IDLE_VIDEO_BYTES} bytes")
                        handle.write(chunk)
            return temp_path
        except Exception:
            with contextlib.suppress(OSError):
                os.remove(temp_path)
            raise

    def _decode_local_video(self, local_path: str):
        logger.info("Loading idle video frames from %s", local_path)
        cap = cv2.VideoCapture(local_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file {local_path}")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if (
            frame_count <= 0
            or frame_count > MAX_IDLE_VIDEO_FRAMES
            or width <= 0
            or height <= 0
            or width > MAX_IDLE_VIDEO_DIMENSION
            or height > MAX_IDLE_VIDEO_DIMENSION
        ):
            cap.release()
            logger.error("Idle video metadata exceeds configured limits")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Store as RGB since LiveKit expects RGB/RGBA
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        cap.release()
        
        self.total_frames = len(self.frames)
        logger.info(f"Loaded {self.total_frames} frames for idle loop")

    def normalize(self, width: int, height: int):
        """Match the one published track's fixed dimensions before its first frame.

        Stale 512 assets are upscaled through the SAME GPU function as live
        frames (chunked batch, bounded VRAM) so idle-to-live crossfades show
        no filter shift during the asset migration window. Regenerated 1024
        assets hit the fast path and are returned untouched.
        """
        if not self.is_valid():
            return
        if self.frames[0].shape[:2] == (height, width):
            return
        try:
            self.frames = upscale_frames_numpy(self.frames, width, height)
        except Exception as exc:
            logger.warning("Idle normalize GPU upscale failed (%s); using cv2 fallback", exc)
            self.frames = [
                frame if frame.shape[:2] == (height, width) else upscale_frame_cv2(frame, width, height)
                for frame in self.frames
            ]
    
    # Return whether idle video frames were loaded successfully.
    def is_valid(self) -> bool:
        return self.total_frames > 0
        
    def _find_best_match_frame(self, target: np.ndarray) -> int:
        """Find idle frame index with lowest MSE to target (downscaled for speed)."""
        if self.total_frames <= 1:
            return 0
        target_rgb = target[:, :, :3] if target.shape[2] == 4 else target
        target_small = cv2.resize(target_rgb, (64, 64))
        target_f = target_small.astype(np.float32)
        best_idx = 0
        best_mse = float('inf')
        for i in range(self.total_frames):
            f = self.frames[i]
            f_rgb = f[:, :, :3] if f.shape[2] == 4 else f
            f_small = cv2.resize(f_rgb, (64, 64))
            diff = f_small.astype(np.float32) - target_f
            mse = float(np.mean(diff * diff))
            if mse < best_mse:
                best_mse = mse
                best_idx = i
        return best_idx

    def _get_effective_loop_crossfade_frames(self) -> int:
        """Get the effective crossfade frame count, clamped to video length."""
        fade = self._loop_crossfade_frames
        if fade <= 0 or self.total_frames < 4:
            return 0
        if self.total_frames < 2 * fade:
            fade = max(2, self.total_frames // 4)
        return fade

    def _generate_loop_crossfade(self, fade_frames: int) -> List[np.ndarray]:
        """Generate crossfade from last frame to first frame for seamless loop.

        Uses plain linear alpha blend between the last and first idle frames.
        """
        if fade_frames <= 0:
            return []

        tail_frame = self.frames[self.total_frames - 1]
        head_frame = self.frames[0]

        blended = []
        for i in range(fade_frames):
            alpha = i / fade_frames
            blended.append(cv2.addWeighted(tail_frame, 1 - alpha, head_frame, alpha, 0))
        return blended

    def get_next_frame(self) -> np.ndarray:
        """Get next frame in loop, with crossfade at the wrap-around boundary."""
        if not self.is_valid():
            return None

        if self._in_loop_transition:
            frame = self._loop_transition_buffer[self._loop_transition_idx]
            self._loop_transition_idx += 1
            if self._loop_transition_idx >= len(self._loop_transition_buffer):
                self._in_loop_transition = False
                self._loop_transition_buffer = []
                self._loop_transition_idx = 0
                self.current_idx = 1
            return frame

        fade = self._get_effective_loop_crossfade_frames()
        if fade > 0 and self.current_idx >= self.total_frames - fade:
            self._loop_crossfade_frames_effective = fade
            if self._loop_crossfade_cache is not None and self._loop_crossfade_cache_n == fade:
                self._loop_transition_buffer = self._loop_crossfade_cache
            else:
                self._loop_transition_buffer = self._generate_loop_crossfade(fade)
                self._loop_crossfade_cache = self._loop_transition_buffer
                self._loop_crossfade_cache_n = fade
            self._in_loop_transition = True
            self._loop_transition_idx = 0
            frame = self._loop_transition_buffer[self._loop_transition_idx]
            self._loop_transition_idx += 1
            return frame

        frame = self.frames[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.total_frames
        return frame
    
    def crossfade_to_idle(self, last_live_frame: np.ndarray, fade_frames: int = 8) -> List[np.ndarray]:
        """Generate crossfade frames from last live frame to idle loop start.

        Uses best-match frame selection to find the idle frame closest to the
        live frame, then plain linear alpha blend.
        """
        if not self.is_valid():
            return []

        h, w = last_live_frame.shape[:2]
        idle_h, idle_w = self.frames[0].shape[:2]
        need_resize = (h, w) != (idle_h, idle_w)

        # Best-match: find idle frame closest to last live frame
        start_idx = 0
        if _USE_BEST_MATCH and self.total_frames > 1:
            try:
                start_idx = self._find_best_match_frame(last_live_frame)
                logger.info("[idle] best-match frame %d/%d", start_idx, self.total_frames)
            except Exception:
                start_idx = 0

        target_idle = self.frames[start_idx]
        if need_resize:
            target_idle = cv2.resize(target_idle, (w, h))

        blended_frames = []
        for i in range(fade_frames):
            alpha = i / fade_frames
            if last_live_frame.shape[2] == 4 and target_idle.shape[2] == 3:
                last_live_rgb = last_live_frame[:, :, :3]
                blended_rgb = cv2.addWeighted(last_live_rgb, 1 - alpha, target_idle, alpha, 0)
                blended = np.concatenate([blended_rgb, last_live_frame[:, :, 3:]], axis=2)
            else:
                blended = cv2.addWeighted(last_live_frame, 1 - alpha, target_idle, alpha, 0)
            blended_frames.append(blended)

        self.current_idx = (start_idx + 1) % self.total_frames
        return blended_frames
    
    def crossfade_from_idle(self, live_frames, current_idle_idx: int, fade_frames: int = 8) -> List[np.ndarray]:
        """Generate crossfade frames from idle loop to buffered live frames.

        Uses plain linear alpha blend between the current idle frame and the
        first live frame.
        """
        if not self.is_valid():
            return []

        if not isinstance(live_frames, (list, tuple)):
            live_frames = [live_frames]
        if not live_frames:
            return []

        h, w = live_frames[0].shape[:2]
        idle_h, idle_w = self.frames[0].shape[:2]
        need_resize = (h, w) != (idle_h, idle_w)

        first_live = live_frames[0]
        first_idle = self.frames[current_idle_idx % self.total_frames]
        if need_resize:
            first_idle = cv2.resize(first_idle, (w, h))

        blended_frames = []
        for i in range(fade_frames):
            alpha = i / fade_frames
            if first_live.shape[2] == 4 and first_idle.shape[2] == 3:
                live_rgb = first_live[:, :, :3]
                blended_rgb = cv2.addWeighted(first_idle, 1 - alpha, live_rgb, alpha, 0)
                blended = np.concatenate([blended_rgb, first_live[:, :, 3:]], axis=2)
            else:
                blended = cv2.addWeighted(first_idle, 1 - alpha, first_live, alpha, 0)
            blended_frames.append(blended)

        return blended_frames
