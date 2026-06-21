# Loads and manages idle video frames in memory for seamless looping and crossfade transitions.
import cv2
import numpy as np
import tempfile
import urllib.request
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

class IdleVideoLoop:
    """
    Loads and manages idle video frames in RAM for fast access.
    Implements seamless looping and crossfade transitions.
    """
    
    # Load idle video frames into memory for looping playback.
    def __init__(self, idle_video_url: str, crossfade_frames: int = 8):
        self.crossfade_frames = crossfade_frames
        self.frames: List[np.ndarray] = []
        self.current_idx = 0
        self.total_frames = 0
        
        self._load_video(idle_video_url)
    
    def _load_video(self, url_or_path: str):
        """Load all frames into RAM."""
        local_path = url_or_path
        
        # Download if it's a URL
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            logger.info(f"Downloading idle video from {url_or_path}")
            fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            try:
                urllib.request.urlretrieve(url_or_path, temp_path)
                local_path = temp_path
            except Exception as e:
                logger.error(f"Failed to download idle video: {e}")
                # Fallback to empty if failed
                self.frames = []
                return

        logger.info(f"Loading idle video frames from {local_path}")
        cap = cv2.VideoCapture(local_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file {local_path}")
            if local_path != url_or_path:
                try: os.remove(local_path)
                except: pass
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Store as RGB since LiveKit expects RGB/RGBA
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        cap.release()
        
        # Cleanup temp file
        if local_path != url_or_path:
            try:
                os.remove(local_path)
            except OSError:
                pass
                
        self.total_frames = len(self.frames)
        logger.info(f"Loaded {self.total_frames} frames for idle loop")
    
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
    
    def crossfade_from_idle(self, first_live_frame: np.ndarray, current_idle_idx: int, fade_frames: int = 8) -> List[np.ndarray]:
        """
        Generate crossfade frames from idle loop to first live frame.
        """
        if not self.is_valid():
            return []
            
        h, w = first_live_frame.shape[:2]
        blended_frames = []
        
        for i in range(fade_frames):
            alpha = i / fade_frames  # 0.0 → 1.0
            idle_frame = self.frames[(current_idle_idx + i) % self.total_frames]
            
            if (h, w) != (idle_frame.shape[0], idle_frame.shape[1]):
                idle_frame = cv2.resize(idle_frame, (w, h))
                
            if first_live_frame.shape[2] == 4 and idle_frame.shape[2] == 3:
                first_live_rgb = first_live_frame[:, :, :3]
                blended_rgb = cv2.addWeighted(idle_frame, 1 - alpha, first_live_rgb, alpha, 0)
                blended = np.concatenate([blended_rgb, first_live_frame[:, :, 3:]], axis=2)
            else:
                blended = cv2.addWeighted(idle_frame, 1 - alpha, first_live_frame, alpha, 0)
                
            blended_frames.append(blended)
            
        return blended_frames
