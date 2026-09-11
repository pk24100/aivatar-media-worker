"""BatchedSession - Per-session state management."""

import os
import time
import logging
import threading
from collections import deque
from queue import Queue
from urllib.parse import urlparse

import numpy as np

from utils.default_avatar_cache import default_avatar_cache
from flash_head.inference import get_audio_embedding, get_base_data

from ._avatar import _resolve_avatar_path
from ._session_states import (
    INITIALIZING,
    ACTIVE,
    DRAINING,
    ENDED,
    IDLE_TIMEOUT_S,
    REACTIVATION_TIMEOUT_S,
)

logger = logging.getLogger("BatchedStreamingEngine")


class BatchedSession:
    """Manages per-session state for batched inference.

    Each session has its own audio context, pending audio buffer, motion frames,
    ref latents, and output queues. The BatchedStreamingEngine forms batches
    from multiple ready sessions and calls run_pipeline_batch once.
    """

    def __init__(self, session_id, pipeline, avatar_path, seed,
                 infer_params):
        self.id = session_id
        self.pipeline = pipeline
        self.seed = seed
        self.state = INITIALIZING
        # Per-session lock: protects pending_audio/audio_context deques,
        # state flips, timestamps and metrics. Engine _lock protects only
        # the sessions dict. Ordering is always engine -> session, never
        # reverse (session code never acquires engine lock). Queues
        # (frame_queue/audio_queue put/get) stay lock-free.
        self._lock = threading.Lock()

        # Inference params (from model config, not env vars)
        self.frame_num = infer_params["frame_num"]
        self.motion_frames_num = infer_params["motion_frames_num"]
        self.slice_len = self.frame_num - self.motion_frames_num
        self.sample_rate = infer_params["sample_rate"]
        self.tgt_fps = infer_params["tgt_fps"]
        self.cached_audio_duration = infer_params["cached_audio_duration"]
        self.slice_samples = self.slice_len * self.sample_rate // self.tgt_fps
        self.cached_audio_samples = self.cached_audio_duration * self.sample_rate
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num

        # Audio state
        self.audio_context = deque(
            [0.0] * self.cached_audio_samples,
            maxlen=self.cached_audio_samples,
        )
        self.pending_audio = deque()

        # Output queues (for downstream frame/audio consumers)
        self.frame_queue = Queue()
        self.audio_queue = Queue()

        # Per-session GPU state (cloned from pipeline after prepare)
        self.ref_img_latent = None
        self.latent_motion_frames = None
        self.original_color_reference = None
        self.color_correction_strength = None

        # Real-time pacing
        self._last_slice_time = None

        # Idle tracking
        self._last_audio_time = time.monotonic()
        self._audio_silence_log = False
        self._has_received_audio = False
        self._reactivated_at = None

        # Metrics
        self.latencies = []
        self.slices_processed = 0
        self._metrics_started_at = time.monotonic()
        self._metrics_input_samples = 0
        self._metrics_slices = 0
        self._metrics_inference_total_ms = 0.0

        # Error state
        self.error = None

        # keep_alive: when True, session is reactivated after drain instead of
        # hard-removed. Used for end_utterance so subsequent utterances can feed
        # audio into the same session without re-registering.
        self.keep_alive = False

        # Temp avatar path (cleaned up on close)
        self._temp_avatar_path = None

    def prepare_avatar(self, avatar_path):
        """Prepare the pipeline with the avatar image and extract per-session state.

        Legacy single-threaded path (used by tests). Production engine path
        (BatchedStreamingEngine.add_session) splits this into: resolve outside
        any lock, get_base_data+clone under engine _avatar_lock, publish ACTIVE
        under engine _lock. This method keeps the same semantics for direct
        callers: network resolve outside session lock, GPU outside session lock
        where possible, state/temp assignment under session lock.
        """
        import torch

        local_path = _resolve_avatar_path(avatar_path)
        parsed = urlparse(avatar_path)
        is_temp = False
        if parsed.scheme in ("http", "https", "s3"):
            try:
                cached_check = default_avatar_cache.get_cached_path(avatar_path)
            except Exception:
                cached_check = None
            if cached_check is None:
                is_temp = True
            else:
                try:
                    if os.path.abspath(local_path) != os.path.abspath(cached_check):
                        is_temp = True
                except Exception:
                    is_temp = True

        get_base_data(self.pipeline, local_path, base_seed=self.seed, use_face_crop=False)
        with self._lock:
            if is_temp:
                self._temp_avatar_path = local_path
            self.ref_img_latent = self.pipeline.ref_img_latent.clone()
            self.latent_motion_frames = self.pipeline.latent_motion_frames.clone()
            self.original_color_reference = self.pipeline.original_color_reference.clone()
            self.color_correction_strength = self.pipeline.color_correction_strength
            self.state = ACTIVE

    def _is_slice_ready_locked(self):
        """Unlocked helper: caller must hold self._lock."""
        if self.state not in (ACTIVE, DRAINING):
            return False
        return len(self.pending_audio) >= self.slice_samples

    def _is_idle_locked(self, now, timeout_s=IDLE_TIMEOUT_S):
        """Unlocked helper: caller must hold self._lock."""
        if self.state != ACTIVE:
            return False
        if self._has_received_audio:
            return (now - self._last_audio_time) > timeout_s
        if self._reactivated_at is not None:
            return (now - self._reactivated_at) > REACTIVATION_TIMEOUT_S
        return False

    def _flush_one_locked(self):
        """Unlocked helper: caller must hold self._lock."""
        if self.pending_audio and self.state == DRAINING:
            pad = (-len(self.pending_audio)) % self.slice_samples
            if pad:
                self.pending_audio.extend([0.0] * pad)
            return True
        return False

    def _pop_audio_slice_locked(self):
        """Atomically check readiness, pop one slice and copy audio context.

        Caller must hold self._lock. Returns (audio_array_copy, chunk) or
        (None, None) if not ready. Holds lock only for deque ops; GPU
        embedding happens outside the lock to preserve inference latency.
        """
        if self.state not in (ACTIVE, DRAINING):
            return None, None
        if len(self.pending_audio) < self.slice_samples:
            return None, None
        human_speech_array = np.array(
            [self.pending_audio.popleft() for _ in range(self.slice_samples)],
            dtype=np.float32,
        )
        self.audio_context.extend(human_speech_array.tolist())
        audio_array = np.array(self.audio_context, dtype=np.float32)
        return audio_array, human_speech_array

    def pop_audio_slice(self):
        """Thread-safe pop of one slice. Returns (audio_array, chunk) or (None, None)."""
        with self._lock:
            return self._pop_audio_slice_locked()

    def is_fresh(self):
        """Thread-safe check for off-tick fast path (_last_slice_time is None)."""
        with self._lock:
            return self._last_slice_time is None

    def try_mark_idle_drain(self, timeout_s=IDLE_TIMEOUT_S):
        """Atomically check idle and mark DRAINING. Returns (drained, had_audio)."""
        now = time.monotonic()
        with self._lock:
            if not self._is_idle_locked(now, timeout_s):
                return False, False
            had_audio = self._has_received_audio
            if self.state in (ACTIVE, INITIALIZING):
                self.state = DRAINING
            return True, had_audio

    def feed_audio(self, audio_data):
        """Feed an audio chunk into the session's pending buffer."""
        audio_array = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        with self._lock:
            if self.state in (DRAINING, ENDED):
                return
            self.pending_audio.extend(audio_array.tolist())
            self._metrics_input_samples += len(audio_array)
            self._last_audio_time = time.monotonic()
            self._audio_silence_log = False
            self._has_received_audio = True
            self._reactivated_at = None

    def is_slice_ready(self):
        """Check if this session has enough audio for a full slice.

        Real-time pacing is enforced engine-wide by the global batch tick in
        run_inference_cycle, not per session. Per-session pacing phase-locks
        concurrent sessions one cycle apart so they can never coalesce into a
        batch; the engine tick paces all sessions on a shared grid instead."""
        with self._lock:
            return self._is_slice_ready_locked()

    def is_idle(self, timeout_s=IDLE_TIMEOUT_S):
        """Check if session has been idle (no audio) for too long.

        Three cases:
        1. Session received audio then went silent -> idle timeout after IDLE_TIMEOUT_S.
        2. Session was reactivated (keep_alive) and is waiting for next
           utterance -> reactivation timeout after REACTIVATION_TIMEOUT_S.
        3. Brand-new session waiting for first audio -> no timeout (handled
           by handler-level _watch_session_idle safety net)."""
        now = time.monotonic()
        with self._lock:
            return self._is_idle_locked(now, timeout_s)

    def get_audio_embedding(self):
        """Extract audio embedding from current audio context.

        Pops one slice atomically under session lock, then runs GPU
        preprocess outside the lock to avoid blocking feed_audio.
        Raises RuntimeError if no slice is ready (e.g. cancelled between
        is_slice_ready check and pop); caller should treat as not-ready.
        """
        audio_array, human_speech_array = self.pop_audio_slice()
        if audio_array is None:
            raise RuntimeError("slice not ready (cancelled or drained)")
        emb = get_audio_embedding(self.pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
        return emb, human_speech_array

    def update_motion_frames(self, new_mf):
        """Update motion frames after inference (chaining)."""
        with self._lock:
            self.latent_motion_frames = new_mf

    def get_inference_state(self):
        """Return per-session state needed for run_pipeline_batch."""
        import torch
        return {
            "latent_motion_frames": self.latent_motion_frames.clone(),
            "generator": torch.Generator(device=self.pipeline.device).manual_seed(self.seed),
            "original_color_reference": self.original_color_reference.clone(),
            "color_correction_strength": self.color_correction_strength,
            "ref_img_latent": self.ref_img_latent.clone(),
        }

    def receive_frames(self, frames_np, audio_chunk, inference_ms):
        """Receive processed frames and audio, enqueue for downstream consumers.

        Queues stay lock-free; metrics and pacing stamp are protected by
        session lock.
        """
        for i in range(frames_np.shape[0]):
            self.frame_queue.put_nowait(frames_np[i])
        self.audio_queue.put_nowait(audio_chunk)

        with self._lock:
            self.latencies.append(inference_ms)
            self.slices_processed += 1
            self._metrics_slices += 1
            self._metrics_inference_total_ms += inference_ms
            # Real-time pacing stamp
            self._last_slice_time = time.monotonic()

    def next_slice_delay(self):
        """Return delay before next slice can be generated at target FPS."""
        with self._lock:
            last = self._last_slice_time
        if last is None:
            return 0.0
        slice_realtime = self.slice_len / float(self.tgt_fps)
        return max(0.0, slice_realtime - (time.monotonic() - last))

    def start_drain(self):
        """Mark session for draining - finish queued audio then end."""
        with self._lock:
            if self.state in (ACTIVE, INITIALIZING):
                self.state = DRAINING

    def reactivate(self):
        """Reactivate a drained session for continued use (keep_alive=True).

        Called by the engine after drain completes when keep_alive is True.
        Resets state to ACTIVE so new audio can be fed and processed.
        Clears keep_alive so idle timeout removal works normally.
        Resets _has_received_audio so the session won't be timed out
        while waiting for the next utterance's audio."""
        with self._lock:
            self.state = ACTIVE
            self.keep_alive = False
            self._last_audio_time = time.monotonic()
            self._last_slice_time = None
            self._audio_silence_log = False
            self._has_received_audio = False
            self._reactivated_at = time.monotonic()

    def flush_one(self):
        """Pad and process one final slice during drain. Returns True if audio remains."""
        with self._lock:
            return self._flush_one_locked()

    def close(self):
        """Clean up session state.

        Deques/state under session lock; queues and temp-file unlink stay
        outside the lock to avoid blocking feed_audio during IO.
        """
        with self._lock:
            self.state = ENDED
            self.pending_audio.clear()
            self.audio_context.clear()
            temp_path = self._temp_avatar_path
            self._temp_avatar_path = None
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break
        # Clean up temp avatar file (only per-session downloads, never the
        # shared default-avatar cache).
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    def mark_error(self, err):
        """Mark session as errored."""
        with self._lock:
            self.error = str(err)
            self.state = ENDED

    def metrics_summary(self):
        """Return per-session metrics dict."""
        with self._lock:
            lat_copy = list(self.latencies)
            slices = self.slices_processed
            m_slices = self._metrics_slices
            m_total = self._metrics_inference_total_ms
            state = self.state
            err = self.error
        avg_inf = (m_total / m_slices if m_slices else 0.0)
        return {
            "slices": slices,
            "avg_latency": float(np.mean(lat_copy)) if lat_copy else 0.0,
            "p50_latency": float(np.percentile(lat_copy, 50)) if lat_copy else 0.0,
            "p95_latency": float(np.percentile(lat_copy, 95)) if lat_copy else 0.0,
            "max_latency": float(np.max(lat_copy)) if lat_copy else 0.0,
            "avg_inference_ms": avg_inf,
            "state": state,
            "error": err,
        }
