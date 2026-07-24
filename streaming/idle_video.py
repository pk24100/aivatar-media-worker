# Loads and manages idle video frames in memory for seamless looping and crossfade transitions.
import contextlib
import cv2
import numpy as np
import tempfile
import os
import logging
from typing import List
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)
MAX_IDLE_VIDEO_BYTES = int(os.getenv("IDLE_VIDEO_MAX_BYTES", str(20 * 1024 * 1024)))
MAX_IDLE_VIDEO_FRAMES = int(os.getenv("IDLE_VIDEO_MAX_FRAMES", "750"))
MAX_IDLE_VIDEO_DIMENSION = int(os.getenv("IDLE_VIDEO_MAX_DIMENSION", "1024"))

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
                allowed = {
                    host.strip().lower()
                    for host in os.getenv("ALLOWED_IMAGE_DOMAINS", "").split(",")
                    if host.strip()
                }
                if allowed and parsed.hostname not in allowed:
                    raise ValueError("source image host is not allowed")
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
            scale = max(width / source_width, height / source_height)
            resized_width = max(width, int(np.ceil(source_width * scale)))
            resized_height = max(height, int(np.ceil(source_height * scale)))
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            left = (resized_width - width) // 2
            top = (resized_height - height) // 2
            frame = cv2.cvtColor(resized[top:top + height, left:left + width], cv2.COLOR_BGR2RGB)
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
        """Match the one published track's fixed dimensions before its first frame."""
        if not self.is_valid():
            return
        self.frames = [
            frame if frame.shape[:2] == (height, width) else cv2.resize(frame, (width, height))
            for frame in self.frames
        ]
    
    # Return whether idle video frames were loaded successfully.
    def is_valid(self) -> bool:
        return self.total_frames > 0
        
    def get_next_frame(self) -> np.ndarray:
        """Get next frame in loop, wrapping around."""
        if not self.is_valid():
            return None
            
        frame = self.frames[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.total_frames
        return frame
    
    def crossfade_to_idle(self, last_live_frame: np.ndarray, fade_frames: int = 8) -> List[np.ndarray]:
        """
        Generate crossfade frames from last live frame to idle loop start.
        """
        if not self.is_valid():
            return []
            
        # Ensure sizes match
        h, w = last_live_frame.shape[:2]
        idle_h, idle_w = self.frames[0].shape[:2]
        
        blended_frames = []
        for i in range(fade_frames):
            alpha = i / fade_frames  # 0.0 → 1.0
            idle_frame = self.frames[i % self.total_frames]
            
            # Resize idle frame if it doesn't match the live frame size
            if (h, w) != (idle_h, idle_w):
                idle_frame = cv2.resize(idle_frame, (w, h))
                
            # If live frame is RGBA, ignore alpha for blending, or handle it
            if last_live_frame.shape[2] == 4 and idle_frame.shape[2] == 3:
                last_live_rgb = last_live_frame[:, :, :3]
                blended_rgb = cv2.addWeighted(last_live_rgb, 1 - alpha, idle_frame, alpha, 0)
                # Keep original alpha
                blended = np.concatenate([blended_rgb, last_live_frame[:, :, 3:]], axis=2)
            else:
                blended = cv2.addWeighted(last_live_frame, 1 - alpha, idle_frame, alpha, 0)
                
            blended_frames.append(blended)
            
        # Set next index to where crossfade ends
        self.current_idx = fade_frames % self.total_frames
        return blended_frames
    
    def crossfade_from_idle(self, live_frames, current_idle_idx: int, fade_frames: int = 8) -> List[np.ndarray]:
        """
        Generate crossfade frames from idle loop to buffered live frames.
        """
        if not self.is_valid():
            return []

        if not isinstance(live_frames, (list, tuple)):
            live_frames = [live_frames]
        if not live_frames:
            return []

        h, w = live_frames[0].shape[:2]
        blended_frames = []
        
        for i in range(fade_frames):
            alpha = i / fade_frames  # 0.0 → 1.0
            live_frame = live_frames[min(i, len(live_frames) - 1)]
            idle_frame = self.frames[(current_idle_idx + i) % self.total_frames]
            
            if (h, w) != (idle_frame.shape[0], idle_frame.shape[1]):
                idle_frame = cv2.resize(idle_frame, (w, h))
                
            if live_frame.shape[2] == 4 and idle_frame.shape[2] == 3:
                live_rgb = live_frame[:, :, :3]
                blended_rgb = cv2.addWeighted(idle_frame, 1 - alpha, live_rgb, alpha, 0)
                blended = np.concatenate([blended_rgb, live_frame[:, :, 3:]], axis=2)
            else:
                blended = cv2.addWeighted(idle_frame, 1 - alpha, live_frame, alpha, 0)
                
            blended_frames.append(blended)
            
        return blended_frames
