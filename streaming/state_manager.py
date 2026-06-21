# Manages live/idle state transitions and crossfading for the avatar stream.
import time
import logging
from enum import Enum
from queue import Queue, Empty
from typing import Optional, List
import numpy as np

from streaming.idle_video import IdleVideoLoop

logger = logging.getLogger(__name__)

# Possible states of the avatar stream.
class StreamState(Enum):
    LIVE = "live"
    IDLE = "idle"
    TRANSITION_TO_IDLE = "to_idle"
    TRANSITION_TO_LIVE = "to_live"

class StreamStateManager:
    """
    Manages transitions between live generation and idle playback.
    Acts as a proxy queue for the VideoPublisher.
    """
    
    # Initialize state manager with live queue, idle video, and timeouts.
    def __init__(
        self,
        live_frame_queue: Queue,
        idle_video: Optional[IdleVideoLoop],
        idle_timeout_ms: int = 500,
        crossfade_frames: int = 8,
    ):
        self.live_frame_queue = live_frame_queue
        self.idle_video = idle_video
        self.idle_timeout = idle_timeout_ms / 1000.0
        self.crossfade_frames = crossfade_frames
        
        # We start in IDLE mode to catch cold starts if idle_video is available
        self.state = StreamState.IDLE if (idle_video and idle_video.is_valid()) else StreamState.LIVE
        self.last_frame_time = time.time()
        self.last_live_frame: Optional[np.ndarray] = None
        
        self.transition_frames: List[np.ndarray] = []
        self.transition_idx = 0
        self._first_live_frame = None
        self._transition_start_idle_idx = 0
        
    # Begin crossfade transition from live to idle playback.
    def _start_transition_to_idle(self):
        if not self.idle_video or not self.idle_video.is_valid() or self.last_live_frame is None:
            self.state = StreamState.IDLE
            return
            
        self.transition_frames = self.idle_video.crossfade_to_idle(
            self.last_live_frame, 
            self.crossfade_frames
        )
        self.transition_idx = 0
        self.state = StreamState.TRANSITION_TO_IDLE
        logger.debug("Starting transition to IDLE")
        
    # Begin crossfade transition from idle to live playback.
    def _start_transition_to_live(self):
        if not self.idle_video or not self.idle_video.is_valid():
            self.state = StreamState.LIVE
            return
            
        self._transition_start_idle_idx = self.idle_video.current_idx
        self.transition_frames = []
        self.transition_idx = 0
        self.state = StreamState.TRANSITION_TO_LIVE
        self._first_live_frame = None
        logger.debug("Starting transition to LIVE")

    # Return the next frame based on current stream state.
    def get_next_frame(self) -> Optional[np.ndarray]:
        current_time = time.time()
        
        # === STATE: LIVE ===
        if self.state == StreamState.LIVE:
            try:
                frame = self.live_frame_queue.get_nowait()
                # If the queue has built up beyond ~2 slices, skip to the
                # freshest frame.  This prevents video from lagging behind
                # audio when the engine produces a burst of frames (e.g.
                # during initial buffering or after a WebSocket reconnect).
                _drained = 0
                while self.live_frame_queue.qsize() > 48:
                    frame = self.live_frame_queue.get_nowait()
                    _drained += 1
                if _drained > 0:
                    logger.info(
                        "[SM] Drained %d stale frames from live queue (qsize now %d)",
                        _drained, self.live_frame_queue.qsize(),
                    )
                self.last_live_frame = frame
                self.last_frame_time = current_time
                return frame
            except Empty:
                if current_time - self.last_frame_time > self.idle_timeout:
                    self._start_transition_to_idle()
                    return self.get_next_frame()
                else:
                    return self.last_live_frame
                    
        # === STATE: IDLE ===
        elif self.state == StreamState.IDLE:
            # Check if there's any new live frame arrived
            if self.live_frame_queue.qsize() > 0:
                self._start_transition_to_live()
                return self.get_next_frame()
                
            if self.idle_video and self.idle_video.is_valid():
                return self.idle_video.get_next_frame()
            else:
                return self.last_live_frame
                
        # === STATE: TRANSITION_TO_IDLE ===
        elif self.state == StreamState.TRANSITION_TO_IDLE:
            # If live frame arrives during transition, abort and go back to live
            if self.live_frame_queue.qsize() > 0:
                self._start_transition_to_live()
                return self.get_next_frame()
                
            if self.transition_idx < len(self.transition_frames):
                frame = self.transition_frames[self.transition_idx]
                self.transition_idx += 1
                return frame
            else:
                self.state = StreamState.IDLE
                return self.get_next_frame()
                
        # === STATE: TRANSITION_TO_LIVE ===
        elif self.state == StreamState.TRANSITION_TO_LIVE:
            if self._first_live_frame is None:
                try:
                    first_live = self.live_frame_queue.get_nowait()
                    self._first_live_frame = first_live
                    self.last_live_frame = first_live
                    self.last_frame_time = current_time
                    self.transition_frames = self.idle_video.crossfade_from_idle(
                        first_live,
                        self._transition_start_idle_idx,
                        self.crossfade_frames
                    )
                    self.transition_idx = 0
                except Empty:
                    return self.idle_video.get_next_frame()
            
            if self.transition_idx < len(self.transition_frames):
                frame = self.transition_frames[self.transition_idx]
                self.transition_idx += 1
                return frame
            else:
                self.state = StreamState.LIVE
                return self._first_live_frame

        return self.last_live_frame
