"""
BatchedStreamingEngine: Manages multiple concurrent streaming sessions with
batched inference using run_pipeline_batch.

All sessions share a single pipeline and are batched together with a 20ms
fixed wait window. Uses all GPU SMs (no green context partitioning).

Ported from modal_stress_test.py and adapted for production use:
- No simulated audio (real audio fed via feed_audio)
- Background thread lifecycle (start/stop) instead of fixed-duration run_loop
- SSRF-safe avatar URL download (reused from flashhead_streaming.py)
"""

import os
import sys
import time
import logging
import threading
import tempfile
import uuid
from collections import deque
from queue import Queue
from urllib.parse import urlparse

import numpy as np
import requests
import socket
import ipaddress
import boto3

from utils.default_avatar_cache import default_avatar_cache

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "SoulX-FlashHead"))
from flash_head.inference import get_audio_embedding, get_base_data, get_infer_params, run_pipeline_batch

logger = logging.getLogger("BatchedStreamingEngine")

# --- Config ---
WAIT_WINDOW_MS = 20
IDLE_TIMEOUT_S = 10
REACTIVATION_TIMEOUT_S = int(os.environ.get("AIVATAR_REACTIVATION_TIMEOUT_S", "120"))

# --- SSRF prevention (reused from flashhead_streaming.py) ---
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


def _resolve_avatar_path(avatar_image_path: str) -> str:
    """Resolve avatar URL/path to a local file, downloading if needed."""
    cached_avatar_path = default_avatar_cache.get_cached_path(avatar_image_path)
    if cached_avatar_path:
        logger.info("Using cached default avatar for %s -> %s", avatar_image_path, cached_avatar_path)
        return cached_avatar_path

    parsed = urlparse(avatar_image_path)
    if parsed.scheme in ("http", "https"):
        _validate_image_url(parsed)
        with requests.get(avatar_image_path, timeout=(10, 30), stream=True,
                          headers={"Referer": "https://facemode.io"}) as response:
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
        return temp_path
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 sourceImage: {avatar_image_path}")
        suffix = os.path.splitext(key)[1] or ".png"
        temp_path = os.path.join(tempfile.gettempdir(), f"avatar_{uuid.uuid4().hex}{suffix}")
        boto3.client("s3").download_file(bucket, key, temp_path)
        return temp_path
    if parsed.scheme == "file":
        avatar_image_path = parsed.path
    elif parsed.scheme:
        raise ValueError(f"Unsupported sourceImage scheme: {parsed.scheme}")
    if not os.path.exists(avatar_image_path):
        raise FileNotFoundError(f"Avatar image not found: {avatar_image_path}")
    return avatar_image_path


# =============================================================================
# Session States
# =============================================================================
INITIALIZING = "INITIALIZING"
ACTIVE = "ACTIVE"
DRAINING = "DRAINING"
ENDED = "ENDED"


# =============================================================================
# BatchedSession - Per-session state management
# =============================================================================
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
        """Prepare the pipeline with the avatar image and extract per-session state."""
        import torch

        local_path = _resolve_avatar_path(avatar_path)
        # Track temp path for cleanup
        parsed = urlparse(avatar_path)
        if parsed.scheme in ("http", "https", "s3"):
            self._temp_avatar_path = local_path

        get_base_data(self.pipeline, local_path, base_seed=self.seed, use_face_crop=False)
        self.ref_img_latent = self.pipeline.ref_img_latent.clone()
        self.latent_motion_frames = self.pipeline.latent_motion_frames.clone()
        self.original_color_reference = self.pipeline.original_color_reference.clone()
        self.color_correction_strength = self.pipeline.color_correction_strength
        self.state = ACTIVE

    def feed_audio(self, audio_data):
        """Feed an audio chunk into the session's pending buffer."""
        if self.state in (DRAINING, ENDED):
            return
        audio_array = np.asarray(audio_data, dtype=np.float32).reshape(-1)
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
        if self.state not in (ACTIVE, DRAINING):
            return False
        if len(self.pending_audio) < self.slice_samples:
            return False
        return True

    def is_idle(self, timeout_s=IDLE_TIMEOUT_S):
        """Check if session has been idle (no audio) for too long.

        Three cases:
        1. Session received audio then went silent -> idle timeout after IDLE_TIMEOUT_S.
        2. Session was reactivated (keep_alive) and is waiting for next
           utterance -> reactivation timeout after REACTIVATION_TIMEOUT_S.
        3. Brand-new session waiting for first audio -> no timeout (handled
           by handler-level _watch_session_idle safety net)."""
        if self.state != ACTIVE:
            return False
        now = time.monotonic()
        if self._has_received_audio:
            return (now - self._last_audio_time) > timeout_s
        if self._reactivated_at is not None:
            return (now - self._reactivated_at) > REACTIVATION_TIMEOUT_S
        return False

    def get_audio_embedding(self):
        """Extract audio embedding from current audio context."""
        human_speech_array = np.array(
            [self.pending_audio.popleft() for _ in range(self.slice_samples)],
            dtype=np.float32,
        )
        self.audio_context.extend(human_speech_array.tolist())
        audio_array = np.array(self.audio_context, dtype=np.float32)
        emb = get_audio_embedding(self.pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
        return emb, human_speech_array

    def update_motion_frames(self, new_mf):
        """Update motion frames after inference (chaining)."""
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
        """Receive processed frames and audio, enqueue for downstream consumers."""
        for i in range(frames_np.shape[0]):
            self.frame_queue.put_nowait(frames_np[i])
        self.audio_queue.put_nowait(audio_chunk)

        self.latencies.append(inference_ms)
        self.slices_processed += 1
        self._metrics_slices += 1
        self._metrics_inference_total_ms += inference_ms

        # Real-time pacing stamp
        self._last_slice_time = time.monotonic()

    def next_slice_delay(self):
        """Return delay before next slice can be generated at target FPS."""
        if self._last_slice_time is None:
            return 0.0
        slice_realtime = self.slice_len / float(self.tgt_fps)
        return max(0.0, slice_realtime - (time.monotonic() - self._last_slice_time))

    def start_drain(self):
        """Mark session for draining - finish queued audio then end."""
        if self.state in (ACTIVE, INITIALIZING):
            self.state = DRAINING

    def reactivate(self):
        """Reactivate a drained session for continued use (keep_alive=True).

        Called by the engine after drain completes when keep_alive is True.
        Resets state to ACTIVE so new audio can be fed and processed.
        Clears keep_alive so idle timeout removal works normally.
        Resets _has_received_audio so the session won't be timed out
        while waiting for the next utterance's audio."""
        self.state = ACTIVE
        self.keep_alive = False
        self._last_audio_time = time.monotonic()
        self._last_slice_time = None
        self._audio_silence_log = False
        self._has_received_audio = False
        self._reactivated_at = time.monotonic()

    def flush_one(self):
        """Pad and process one final slice during drain. Returns True if audio remains."""
        if self.pending_audio and self.state == DRAINING:
            pad = (-len(self.pending_audio)) % self.slice_samples
            if pad:
                self.pending_audio.extend([0.0] * pad)
            return True
        return False

    def close(self):
        """Clean up session state."""
        self.state = ENDED
        self.pending_audio.clear()
        self.audio_context.clear()
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
        # Clean up temp avatar file
        if self._temp_avatar_path and os.path.exists(self._temp_avatar_path):
            try:
                os.remove(self._temp_avatar_path)
            except Exception:
                pass
            self._temp_avatar_path = None

    def mark_error(self, err):
        """Mark session as errored."""
        self.error = str(err)
        self.state = ENDED

    def metrics_summary(self):
        """Return per-session metrics dict."""
        avg_inf = (
            self._metrics_inference_total_ms / self._metrics_slices
            if self._metrics_slices else 0.0
        )
        return {
            "slices": self.slices_processed,
            "avg_latency": float(np.mean(self.latencies)) if self.latencies else 0.0,
            "p50_latency": float(np.percentile(self.latencies, 50)) if self.latencies else 0.0,
            "p95_latency": float(np.percentile(self.latencies, 95)) if self.latencies else 0.0,
            "max_latency": float(np.max(self.latencies)) if self.latencies else 0.0,
            "avg_inference_ms": avg_inf,
            "state": self.state,
            "error": self.error,
        }


# =============================================================================
# BatchedStreamingEngine - Central batching engine
# =============================================================================
class BatchedStreamingEngine:
    """Manages multiple concurrent sessions with batched inference.

    Core loop:
    1. Check all active sessions for slice readiness
    2. If none ready, wait briefly and retry
    3. If some ready, wait up to wait_window_ms for more
    4. Form batch, call run_pipeline_batch
    5. Distribute frames back to per-session queues
    6. Handle real-time pacing, idle timeouts, and session lifecycle
    """

    def __init__(self, pipeline, wait_window_ms=WAIT_WINDOW_MS):
        self.pipeline = pipeline
        self.wait_window_ms = wait_window_ms
        self.sessions = {}
        self._lock = threading.RLock()
        self._cycle_count = 0
        self._batch_sizes = []
        self._stopped = False
        self._all_latencies = []
        self._thread = None
        # Global batch tick (monotonic timestamp of the next scheduled batch).
        # All ready sessions are processed together once per slice_realtime.
        self._next_batch_time = None

    def add_session(self, session_id, avatar_path, seed, infer_params):
        """Register and prepare a new session. Can be called mid-stream."""
        with self._lock:
            if session_id in self.sessions:
                logger.warning("Session %s already exists, skipping", session_id)
                return
            session = BatchedSession(
                session_id=session_id,
                pipeline=self.pipeline,
                avatar_path=avatar_path,
                seed=seed,
                infer_params=infer_params,
            )
            self.sessions[session_id] = session
            logger.info("Session %s added (total=%d)", session_id, len(self.sessions))
        session.prepare_avatar(avatar_path)
        return session

    def remove_session(self, session_id, keep_alive=False):
        """Gracefully mark a session for removal. Drains on next cycle.

        If keep_alive=True, the session is reactivated after drain completes
        instead of being hard-removed. Used for end_utterance so subsequent
        utterances can feed audio into the same session.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return
            session.keep_alive = keep_alive
            session.start_drain()
            logger.info("Session %s marked for drain (keep_alive=%s)", session_id, keep_alive)

    def _hard_remove_session(self, session_id):
        """Actually remove and close a session from the registry."""
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if session:
                self._all_latencies.extend(session.latencies)
                session.close()
                logger.info("Session %s removed (total=%d)", session_id, len(self.sessions))

    def feed_audio(self, session_id, audio_data):
        """Feed audio to a specific session."""
        with self._lock:
            session = self.sessions.get(session_id)
        if session:
            session.feed_audio(audio_data)

    def get_session(self, session_id):
        """Get a session by ID (for queue access by stream processor)."""
        with self._lock:
            return self.sessions.get(session_id)

    def start(self):
        """Start the background inference loop thread."""
        self._stopped = False
        self._cycle_count = 0
        self._batch_sizes = []
        self._all_latencies = []
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="batched-engine")
        self._thread.start()
        logger.info("BatchedStreamingEngine background thread started")

    def stop(self):
        """Stop the inference loop and drain all sessions."""
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        logger.info("BatchedStreamingEngine stopped")

    def _run_loop(self):
        """Background loop: run inference cycles until stopped."""
        while not self._stopped:
            batch_size, _ = self.run_inference_cycle()
            if batch_size == 0:
                time.sleep(0.001)

        # Drain all remaining sessions on shutdown
        logger.info("Engine shutting down, draining remaining sessions...")
        with self._lock:
            for session in self.sessions.values():
                session.start_drain()

        drain_start = time.monotonic()
        while self.sessions and (time.monotonic() - drain_start) < 10:
            batch_size, _ = self.run_inference_cycle()
            if batch_size == 0:
                with self._lock:
                    for sid in list(self.sessions.keys()):
                        session = self.sessions[sid]
                        if not session.pending_audio:
                            self._hard_remove_session(sid)
                if not self.sessions:
                    break
                time.sleep(0.01)

        # Force-close any remaining
        with self._lock:
            for session in self.sessions.values():
                self._all_latencies.extend(session.latencies)
                session.close()
            self.sessions.clear()

        logger.info("Engine drain complete: %d cycles total", self._cycle_count)

    def run_inference_cycle(self):
        """Run one batching cycle. Returns (batch_size, cycle_latency_ms) or (0, 0)."""
        import torch

        # Snapshot active sessions under lock
        with self._lock:
            active_sessions = {
                sid: s for sid, s in self.sessions.items()
                if s.state == ACTIVE
            }

        # Handle idle timeouts
        for sid, session in list(active_sessions.items()):
            if session.is_idle():
                if session._has_received_audio:
                    logger.info("Session %s idle timeout (%ds), removing", sid, IDLE_TIMEOUT_S)
                else:
                    logger.info("Session %s reactivation timeout (%ds), removing", sid, REACTIVATION_TIMEOUT_S)
                session.start_drain()
                del active_sessions[sid]

        # Handle draining sessions - process remaining audio
        with self._lock:
            draining_sessions = {
                sid: s for sid, s in self.sessions.items()
                if s.state == DRAINING and s.pending_audio
            }

        # Include draining sessions that have enough audio for one final slice
        for sid, session in list(draining_sessions.items()):
            if len(session.pending_audio) < session.slice_samples:
                session.flush_one()

        # Clean up drained sessions that have no pending audio.
        # keep_alive sessions are reactivated instead of removed.
        # This runs before the early return so empty DRAINING sessions
        # don't hang forever when no other sessions are ready.
        with self._lock:
            for sid, session in list(self.sessions.items()):
                if session.state == DRAINING and not session.pending_audio:
                    if session.keep_alive:
                        session.reactivate()
                        logger.info("Session %s reactivated (keep_alive)", sid)
                    else:
                        self._hard_remove_session(sid)

        # Remove hard-removed sessions from draining_sessions snapshot
        draining_sessions = {
            sid: s for sid, s in draining_sessions.items()
            if sid in self.sessions
        }

        all_candidates = {**active_sessions, **draining_sessions}
        if not all_candidates:
            return 0, 0.0

        # Check which sessions have enough audio for a slice (audio gate only;
        # pacing is enforced by the global batch tick below).
        ready_sessions = []
        for sid, session in all_candidates.items():
            if session.is_slice_ready():
                ready_sessions.append((sid, session))

        if not ready_sessions:
            return 0, 0.0

        # Global batch tick: grid-aligned sessions run together in one batch
        # every slice_realtime seconds. Fresh sessions (no slice produced yet,
        # e.g. just registered or reactivated) may run off-tick so first-slice
        # latency after speech start does not regress; they join the grid on
        # the next tick.
        now = time.monotonic()
        tick_due = self._next_batch_time is None or now >= self._next_batch_time
        if not tick_due:
            ready_sessions = [
                (sid, s) for sid, s in ready_sessions
                if s._last_slice_time is None
            ]
            if not ready_sessions:
                return 0, 0.0

        cycle_t0 = time.monotonic()

        # Wait up to wait_window_ms for more sessions to become ready.
        # Only useful at a tick; off-tick fresh sessions run immediately.
        if tick_due:
            wait_deadline = time.monotonic() + (self.wait_window_ms / 1000.0)
            for sid, session in all_candidates.items():
                if (sid, session) in ready_sessions:
                    continue
                if session.is_slice_ready():
                    ready_sessions.append((sid, session))
                elif time.monotonic() < wait_deadline:
                    remaining = wait_deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(remaining, 0.005))
                    if session.is_slice_ready():
                        ready_sessions.append((sid, session))

        if not ready_sessions:
            return 0, 0.0

        batch_size = len(ready_sessions)
        self._batch_sizes.append(batch_size)
        self._cycle_count += 1

        # Generate audio embeddings for ready sessions
        t_embed = time.monotonic()
        audio_embs = []
        audio_chunks = []
        for sid, session in ready_sessions:
            emb, chunk = session.get_audio_embedding()
            audio_embs.append(emb)
            audio_chunks.append(chunk)
        embed_ms = (time.monotonic() - t_embed) * 1000

        # Prepare per-session state for batched call
        states = [session.get_inference_state() for _, session in ready_sessions]
        current_mf = [s["latent_motion_frames"] for s in states]
        current_gens = [s["generator"] for s in states]
        current_refs = [s["original_color_reference"] for s in states]
        current_ccs = [s["color_correction_strength"] for s in states]
        current_ref_latents = [s["ref_img_latent"] for s in states]

        # Batched inference
        t_infer = time.monotonic()
        try:
            frames_list, updated_mf_list = run_pipeline_batch(
                self.pipeline,
                audio_embs,
                current_mf,
                current_ref_latents,
                current_gens,
                current_refs,
                current_ccs,
            )
        except Exception as exc:
            logger.error("Batch inference failed: %s", exc, exc_info=True)
            for sid, session in ready_sessions:
                session.mark_error(exc)
                self._hard_remove_session(sid)
            return batch_size, (time.monotonic() - cycle_t0) * 1000

        infer_ms = (time.monotonic() - t_infer) * 1000

        # Transfer to CPU and distribute frames per session
        t_xfer = time.monotonic()
        for i, (sid, session) in enumerate(ready_sessions):
            try:
                video = frames_list[i][session.motion_frames_num:]
                frames_np = video.cpu().numpy().astype(np.uint8)
                session.update_motion_frames(updated_mf_list[i])
                cycle_latency = (time.monotonic() - cycle_t0) * 1000
                session.receive_frames(frames_np, audio_chunks[i], cycle_latency)
            except Exception as exc:
                logger.error("Frame distribution failed for session %s: %s", sid, exc, exc_info=True)
                session.mark_error(exc)
                self._hard_remove_session(sid)
        xfer_ms = (time.monotonic() - t_xfer) * 1000

        cycle_latency = (time.monotonic() - cycle_t0) * 1000

        # Advance the global batch grid. Anchor to the previous tick so the
        # grid does not drift; clamp to now if a cycle overran its budget.
        if tick_due:
            first_session = ready_sessions[0][1]
            slice_realtime = first_session.slice_len / float(first_session.tgt_fps)
            base = self._next_batch_time if self._next_batch_time is not None else cycle_t0
            self._next_batch_time = max(base + slice_realtime, time.monotonic())

        # Log every 10 cycles
        if self._cycle_count % 10 == 0:
            sids = [sid for sid, _ in ready_sessions]
            logger.info(
                "cycle=%d batch=%d sessions=%s embed=%.1fms infer=%.1fms "
                "xfer=%.1fms total=%.1fms",
                self._cycle_count, batch_size, sids,
                embed_ms, infer_ms, xfer_ms, cycle_latency,
            )

        return batch_size, cycle_latency

    def get_metrics(self):
        """Return aggregate metrics across all sessions."""
        with self._lock:
            sessions = list(self.sessions.values())

        all_latencies = list(self._all_latencies)
        for s in sessions:
            all_latencies.extend(s.latencies)

        fill_1 = self._batch_sizes.count(1) / len(self._batch_sizes) * 100 if self._batch_sizes else 0
        fill_2 = self._batch_sizes.count(2) / len(self._batch_sizes) * 100 if self._batch_sizes else 0
        fill_3 = self._batch_sizes.count(3) / len(self._batch_sizes) * 100 if self._batch_sizes else 0

        return {
            "cycles": self._cycle_count,
            "avg_batch": float(np.mean(self._batch_sizes)) if self._batch_sizes else 0.0,
            "fill_1": fill_1,
            "fill_2": fill_2,
            "fill_3": fill_3,
            "avg_latency": float(np.mean(all_latencies)) if all_latencies else 0.0,
            "p50_latency": float(np.percentile(all_latencies, 50)) if all_latencies else 0.0,
            "p95_latency": float(np.percentile(all_latencies, 95)) if all_latencies else 0.0,
            "max_latency": float(np.max(all_latencies)) if all_latencies else 0.0,
            "active_sessions": len(sessions),
        }
