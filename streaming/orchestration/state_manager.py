# Manages live/idle state transitions and crossfading for the avatar stream.
import math
import time
import logging
from enum import Enum
from queue import Queue, Empty
from typing import Optional, List
import numpy as np

from streaming.orchestration.idle_video import IdleVideoLoop

logger = logging.getLogger(__name__)

# Slice and FPS constants for adaptive cap computation.
_SLICE_LEN = 24
_TGT_FPS = 25
# Fallback per-session inference time estimate (ms) when engine metrics
# are not yet available (e.g. engine just started, non-batched path).
_FALLBACK_INFER_MS_PER_SESSION = 350
# How often to recompute adaptive queue caps (seconds).
_CAP_UPDATE_INTERVAL_S = 30.0

# Default caps used before the first _update_queue_caps() call.  These are
# overwritten in __init__ and every _CAP_UPDATE_INTERVAL_S thereafter.
MAX_LIVE_FRAME_QUEUE = 60
TARGET_LIVE_FRAME_QUEUE = 48

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
        engine=None,
    ):
        self.live_frame_queue = live_frame_queue
        self.idle_video = idle_video
        self.idle_timeout = idle_timeout_ms / 1000.0
        self.crossfade_frames = crossfade_frames
        self._engine = engine
        self._last_cap_update = 0.0
        self.max_live_frame_queue = MAX_LIVE_FRAME_QUEUE
        self.target_live_frame_queue = TARGET_LIVE_FRAME_QUEUE

        # Compute initial caps immediately
        self._update_queue_caps()

        # We start in IDLE mode to catch cold starts if idle_video is available
        self.state = StreamState.IDLE if (idle_video and idle_video.is_valid()) else StreamState.LIVE
        self.last_frame_time = time.time()
        self.last_live_frame: Optional[np.ndarray] = None
        self.last_frame_source = "none"

        self.transition_frames: List[np.ndarray] = []
        self.transition_idx = 0
        self._first_live_frame = None
        self._transition_live_frames: List[np.ndarray] = []
        self._transition_start_idle_idx = 0
        
    # Begin crossfade transition from live to idle playback.
    def _start_transition_to_idle(self):
        if not self.idle_video or not self.idle_video.is_valid() or self.last_live_frame is None:
            self.state = StreamState.IDLE
            logger.info("[SM] LIVE -> IDLE (no idle video, direct switch)")
            return
            
        self.transition_frames = self.idle_video.crossfade_to_idle(
            self.last_live_frame, 
            self.crossfade_frames
        )
        self.transition_idx = 0
        self.state = StreamState.TRANSITION_TO_IDLE
        logger.info("[SM] LIVE -> TRANSITION_TO_IDLE (crossfade %d frames)", self.crossfade_frames)
        
    # Begin crossfade transition from idle to live playback.
    def _start_transition_to_live(self):
        if not self.idle_video or not self.idle_video.is_valid():
            self.state = StreamState.LIVE
            logger.info("[SM] IDLE -> LIVE (no idle video, direct switch)")
            return
            
        self._transition_start_idle_idx = self.idle_video.current_idx
        self.transition_frames = []
        self.transition_idx = 0
        self.state = StreamState.TRANSITION_TO_LIVE
        self._first_live_frame = None
        self._transition_live_frames = []
        logger.info("[SM] IDLE -> TRANSITION_TO_LIVE (crossfade %d frames, queue=%d)",
                     self.crossfade_frames, self.live_frame_queue.qsize())

    # Recompute adaptive queue caps from engine metrics or session count.
    def _update_queue_caps(self):
        """Hybrid adaptive cap computation.

        Primary: use p95 inference latency from engine.get_metrics() when
        enough cycles have been recorded (>5 samples).
        Fallback: estimate from current active session count with a linear
        per-session inference time approximation.

        Formula:
          buffer_frames = ceil(p95_ms / 1000 * tgt_fps)
          natural_peak = buffer_frames + slice_len
          max_cap = natural_peak * 1.3  (safety margin)
          target_cap = max_cap * 0.8   (drain excess but keep buffer)
        """
        p95_ms = None

        if self._engine is not None and hasattr(self._engine, 'get_metrics'):
            try:
                metrics = self._engine.get_metrics()
                if metrics.get('cycles', 0) > 5:
                    p95_ms = metrics.get('p95_latency')
            except Exception:
                pass

        if p95_ms is None:
            # Fallback: estimate from active session count
            n = 1
            if self._engine is not None and hasattr(self._engine, 'sessions'):
                try:
                    n = max(len(self._engine.sessions), 1)
                except Exception:
                    pass
            p95_ms = _FALLBACK_INFER_MS_PER_SESSION * n

        buffer_frames = math.ceil(p95_ms / 1000.0 * _TGT_FPS)
        natural_peak = buffer_frames + _SLICE_LEN
        self.max_live_frame_queue = int(natural_peak * 1.3)
        self.target_live_frame_queue = int(self.max_live_frame_queue * 0.8)

        logger.info(
            "[SM] Adaptive caps: p95_ms=%.0f buffer=%d peak=%d MAX=%d TARGET=%d",
            p95_ms, buffer_frames, natural_peak,
            self.max_live_frame_queue, self.target_live_frame_queue,
        )

    # Return the next frame based on current stream state.
    def get_next_frame(self) -> Optional[np.ndarray]:
        current_time = time.time()

        # Periodically recompute adaptive queue caps
        if current_time - self._last_cap_update > _CAP_UPDATE_INTERVAL_S:
            self._update_queue_caps()
            self._last_cap_update = current_time

        # === STATE: LIVE ===
        if self.state == StreamState.LIVE:
            try:
                frame = self.live_frame_queue.get_nowait()
                # Keep the normal slice buffer, but recover before delayed
                # video can visibly lag realtime audio after a publisher stall.
                _drained = 0
                if self.live_frame_queue.qsize() > self.max_live_frame_queue:
                    while self.live_frame_queue.qsize() > self.target_live_frame_queue:
                        frame = self.live_frame_queue.get_nowait()
                        _drained += 1
                if _drained > 0:
                    logger.info(
                        "[SM] Dropped %d stale frames to protect A/V sync (qsize now %d)",
                        _drained, self.live_frame_queue.qsize(),
                    )
                self.last_live_frame = frame
                self.last_frame_time = current_time
                self.last_frame_source = "live"
                return frame
            except Empty:
                if current_time - self.last_frame_time > self.idle_timeout:
                    self._start_transition_to_idle()
                    return self.get_next_frame()
                else:
                    self.last_frame_source = "repeated_live"
                    return self.last_live_frame
                    
        # === STATE: IDLE ===
        elif self.state == StreamState.IDLE:
            # Buffer the full crossfade window before leaving idle. A single
            # generated frame produces a visibly static transition.
            if self.live_frame_queue.qsize() >= self.crossfade_frames:
                self._start_transition_to_live()
                return self.get_next_frame()
                
            if self.idle_video and self.idle_video.is_valid():
                self.last_frame_source = "idle"
                return self.idle_video.get_next_frame()
            else:
                self.last_frame_source = "repeated_live"
                return self.last_live_frame
                
        # === STATE: TRANSITION_TO_IDLE ===
        elif self.state == StreamState.TRANSITION_TO_IDLE:
            # If live frame arrives during transition, abort and go back to live
            if self.live_frame_queue.qsize() >= self.crossfade_frames:
                self._start_transition_to_live()
                return self.get_next_frame()
                
            if self.transition_idx < len(self.transition_frames):
                frame = self.transition_frames[self.transition_idx]
                self.transition_idx += 1
                self.last_frame_source = "transition_to_idle"
                return frame
            else:
                self.state = StreamState.IDLE
                logger.info("[SM] TRANSITION_TO_IDLE -> IDLE")
                return self.get_next_frame()
                
        # === STATE: TRANSITION_TO_LIVE ===
        elif self.state == StreamState.TRANSITION_TO_LIVE:
            if self._first_live_frame is None:
                try:
                    self._first_live_frame = self.live_frame_queue.get_nowait()
                    self.last_live_frame = self._first_live_frame
                    self.last_frame_time = current_time
                    self.transition_frames = self.idle_video.crossfade_from_idle(
                        [self._first_live_frame],
                        self._transition_start_idle_idx,
                        self.crossfade_frames
                    )
                    self._transition_live_frames = [self._first_live_frame]
                    self.transition_idx = 0
                except Empty:
                    self.last_frame_source = "idle"
                    return self.idle_video.get_next_frame()
            
            if self.transition_idx < len(self.transition_frames):
                frame = self.transition_frames[self.transition_idx]
                self.transition_idx += 1
                self.last_frame_source = "transition_to_live"
                return frame
            else:
                self.state = StreamState.LIVE
                self.last_frame_source = "live"
                logger.info("[SM] TRANSITION_TO_LIVE -> LIVE")
                return self._transition_live_frames[0]

        self.last_frame_source = "repeated_live"
        return self.last_live_frame
