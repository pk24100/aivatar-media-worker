"""
BatchedStreamingEngine stress test on L40S GPU.

Implements a production-grade BatchedStreamingEngine that manages multiple
concurrent streaming sessions, forms batches with a 20ms fixed wait window,
and calls run_pipeline_batch for all ready sessions.

Scenarios tested:
  A: 3 steady sessions (baseline)
  B: Dynamic join/leave (session joins at t=10s, leaves at t=20s)
  C: Audio gaps (session goes silent for 5s, then resumes)
  D: Variable audio frequency (sessions send audio at different rates)
  E: Session crash recovery (one session errors mid-stream)

The existing FlashHeadStreamingEngine + green context code is preserved
as dead code for future use. This engine does NOT use green contexts -
batched inference uses all 142 SMs efficiently.

Usage:
    modal run modal_stress_test.py
"""

import os
import time
import modal
import threading
import numpy as np
from collections import deque
from queue import Queue

# --- Config ---
NUM_SESSIONS = 3
DURATION_SECONDS = 30  # Per scenario
WAIT_WINDOW_MS = 20  # Fixed wait window (proven optimal from benchmarks)
AUDIO_JITTER_MS = 50  # Simulated audio arrival jitter
NUM_WARMUP_SLICES = 3
IDLE_TIMEOUT_S = 10  # Auto-remove session after N seconds of no audio
LOG_FILE = "/tmp/stress_test_results.log"

# --- Modal Image (same as benchmark) ---
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.05-py3")
    .env({
        "UCX_TLS": "self",
        "UCX_NET_DEVICES": "none",
        "UCX_UNIFIED_TLS_MODE": "y",
        "UCX_MEMTYPE_CACHE": "n",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
    .apt_install("git", "git-lfs", "ffmpeg", "libsndfile1", "wget", "ca-certificates")
    .pip_install("ninja")
    .run_commands(
        "TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 NVCC_THREADS=4 pip install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation || echo 'SAGEATTN_INSTALL_FAILED'",
    )
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "huggingface-cli download pkam24100/aivatar-flashhead-model --local-dir /app/models/SoulX-FlashHead-1_3B",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_dir("SoulX-FlashHead", "/app/SoulX-FlashHead", copy=True)
    .add_local_file(
        "flash_head_model_snapshot_patch.py",
        "/app/SoulX-FlashHead/flash_head/src/modules/flash_head_model.py",
        copy=True,
    )
    .add_local_dir("models/wav2vec2-base-960h", "/app/models/wav2vec2-base-960h", copy=True)
)

app = modal.App("aivatar-batched-stress-test", image=image)


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

    def __init__(self, session_id, pipeline, avatar_path, seed, audio_freq,
                 infer_params):
        self.id = session_id
        self.pipeline = pipeline
        self.seed = seed
        self.audio_freq = audio_freq
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

        # Metrics
        self.latencies = []
        self.slices_processed = 0
        self._metrics_started_at = time.monotonic()
        self._metrics_input_samples = 0
        self._metrics_slices = 0
        self._metrics_inference_total_ms = 0.0

        # Generate continuous audio stream for simulation
        total_samples = (DURATION_SECONDS + 30) * self.sample_rate + self.cached_audio_samples
        t = np.linspace(0, total_samples / self.sample_rate, total_samples, dtype=np.float32)
        self.audio_stream = (
            0.3 * np.sin(2 * np.pi * audio_freq * t)
            + 0.2 * np.sin(2 * np.pi * (audio_freq * 1.5) * t)
            + 0.1 * np.sin(2 * np.pi * (audio_freq * 0.5) * t)
        ).astype(np.float32)
        self.audio_pos = 0

        # Error state
        self.error = None

        # Prepare avatar immediately
        self._prepare_avatar(avatar_path)

    def _prepare_avatar(self, avatar_path):
        """Prepare the pipeline with the avatar image and extract per-session state."""
        from flash_head.inference import get_base_data
        import torch

        get_base_data(self.pipeline, avatar_path, base_seed=self.seed, use_face_crop=False)
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

    def feed_simulated_audio(self, jitter_ms=0):
        """Feed one slice of simulated audio with optional jitter delay."""
        if self.state in (DRAINING, ENDED):
            return
        if jitter_ms > 0:
            time.sleep(jitter_ms / 1000.0)
        chunk = self.audio_stream[self.audio_pos:self.audio_pos + self.slice_samples]
        self.audio_pos += self.slice_samples
        self.feed_audio(chunk)

    def is_slice_ready(self):
        """Check if this session has enough audio for a full slice."""
        return self.state == ACTIVE and len(self.pending_audio) >= self.slice_samples

    def is_idle(self, timeout_s=IDLE_TIMEOUT_S):
        """Check if session has been idle (no audio) for too long."""
        return self.state == ACTIVE and (time.monotonic() - self._last_audio_time) > timeout_s

    def get_audio_embedding(self):
        """Extract audio embedding from current audio context."""
        from flash_head.inference import get_audio_embedding as _get_emb

        human_speech_array = np.array(
            [self.pending_audio.popleft() for _ in range(self.slice_samples)],
            dtype=np.float32,
        )
        self.audio_context.extend(human_speech_array.tolist())
        audio_array = np.array(self.audio_context, dtype=np.float32)
        emb = _get_emb(self.pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
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
        if self.state == ACTIVE:
            self.state = DRAINING

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
        self._all_latencies = []  # Accumulated latencies from all sessions (survives session removal)

    def add_session(self, session_id, avatar_path, seed, audio_freq,
                    infer_params):
        """Register and prepare a new session. Can be called mid-stream."""
        with self._lock:
            if session_id in self.sessions:
                print(f"[ENGINE] Session {session_id} already exists, skipping", flush=True)
                return
            session = BatchedSession(
                session_id=session_id,
                pipeline=self.pipeline,
                avatar_path=avatar_path,
                seed=seed,
                audio_freq=audio_freq,
                infer_params=infer_params,
            )
            self.sessions[session_id] = session
            print(f"[ENGINE] Session {session_id} added (total={len(self.sessions)})", flush=True)

    def remove_session(self, session_id):
        """Gracefully mark a session for removal. Drains on next cycle."""
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return
            session.start_drain()
            print(f"[ENGINE] Session {session_id} marked for drain", flush=True)

    def _hard_remove_session(self, session_id):
        """Actually remove and close a session from the registry."""
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if session:
                self._all_latencies.extend(session.latencies)
                session.close()
                print(f"[ENGINE] Session {session_id} removed (total={len(self.sessions)})", flush=True)

    def feed_audio(self, session_id, audio_data):
        """Feed audio to a specific session."""
        with self._lock:
            session = self.sessions.get(session_id)
        if session:
            session.feed_audio(audio_data)

    def stop(self):
        """Stop the inference loop."""
        self._stopped = True

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
                print(f"[ENGINE] Session {sid} idle timeout ({IDLE_TIMEOUT_S}s), removing", flush=True)
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
            if not session.pending_audio:
                self._hard_remove_session(sid)
                del draining_sessions[sid]

        all_candidates = {**active_sessions, **draining_sessions}
        if not all_candidates:
            return 0, 0.0

        # Check which sessions have enough audio for a slice
        ready_sessions = []
        for sid, session in all_candidates.items():
            if len(session.pending_audio) >= session.slice_samples:
                ready_sessions.append((sid, session))

        if not ready_sessions:
            return 0, 0.0

        cycle_t0 = time.monotonic()

        # Wait up to wait_window_ms for more sessions to become ready
        wait_deadline = time.monotonic() + (self.wait_window_ms / 1000.0)
        for sid, session in all_candidates.items():
            if (sid, session) in ready_sessions:
                continue
            if len(session.pending_audio) >= session.slice_samples:
                ready_sessions.append((sid, session))
            elif time.monotonic() < wait_deadline:
                # Brief wait for more audio to arrive
                remaining = wait_deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(remaining, 0.005))  # 5ms increments
                if len(session.pending_audio) >= session.slice_samples:
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
        from flash_head.inference import run_pipeline_batch

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
            print(f"[ENGINE] Batch inference failed: {exc}", flush=True)
            # Mark all ready sessions as errored
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
                print(f"[ENGINE] Frame distribution failed for session {sid}: {exc}", flush=True)
                session.mark_error(exc)
                self._hard_remove_session(sid)
        xfer_ms = (time.monotonic() - t_xfer) * 1000

        cycle_latency = (time.monotonic() - cycle_t0) * 1000

        # Log every 10 cycles
        if self._cycle_count % 10 == 0:
            sids = [sid for sid, _ in ready_sessions]
            print(f"[ENGINE] cycle={self._cycle_count} batch={batch_size} "
                  f"sessions={sids} embed={embed_ms:.1f}ms infer={infer_ms:.1f}ms "
                  f"xfer={xfer_ms:.1f}ms total={cycle_latency:.1f}ms", flush=True)

        # Clean up drained sessions that are now empty
        with self._lock:
            for sid, session in list(self.sessions.items()):
                if session.state == DRAINING and not session.pending_audio:
                    self._hard_remove_session(sid)

        return batch_size, cycle_latency

    def run_loop(self, duration_seconds, audio_feed_fn=None):
        """Run the inference loop for a fixed duration.

        Args:
            duration_seconds: How long to run the loop.
            audio_feed_fn: Optional callback(engine, elapsed) to feed audio
                           and manage sessions during the loop. If None,
                           uses default audio feeding logic.
        """
        self._stopped = False
        self._cycle_count = 0
        self._batch_sizes = []

        loop_start = time.monotonic()
        while not self._stopped and (time.monotonic() - loop_start) < duration_seconds:
            elapsed = time.monotonic() - loop_start

            # Call audio feed callback if provided
            if audio_feed_fn is not None:
                audio_feed_fn(self, elapsed)

            # Run one inference cycle
            batch_size, cycle_latency = self.run_inference_cycle()

            # If no sessions were ready, brief sleep to avoid busy-spin
            if batch_size == 0:
                time.sleep(0.001)

        # Drain all remaining sessions
        print("[ENGINE] Draining remaining sessions...", flush=True)
        with self._lock:
            for session in self.sessions.values():
                session.start_drain()

        drain_start = time.monotonic()
        while self.sessions and (time.monotonic() - drain_start) < 10:
            batch_size, _ = self.run_inference_cycle()
            if batch_size == 0:
                # Remove remaining empty sessions
                with self._lock:
                    for sid in list(self.sessions.keys()):
                        session = self.sessions[sid]
                        if not session.pending_audio:
                            self._hard_remove_session(sid)
                if not self.sessions:
                    break
                time.sleep(0.01)

        # Force-close any remaining (save latencies first)
        with self._lock:
            for session in self.sessions.values():
                self._all_latencies.extend(session.latencies)
                session.close()
            self.sessions.clear()

        elapsed = time.monotonic() - loop_start
        print(f"[ENGINE] Loop complete: {self._cycle_count} cycles in {elapsed:.1f}s", flush=True)

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
            "throughput": self._cycle_count / (DURATION_SECONDS) if DURATION_SECONDS > 0 else 0,
        }


# =============================================================================
# Stress Test Scenarios
# =============================================================================
def scenario_a_steady(engine, sessions_data, infer_params):
    """Scenario A: 3 steady sessions (baseline).

    All 3 sessions active for entire duration with continuous audio.
    """
    print("\n[SCENARIO A] 3 steady sessions, continuous audio", flush=True)

    # Add all sessions upfront
    for sd in sessions_data:
        engine.add_session(
            session_id=sd["id"],
            avatar_path=sd["avatar_path"],
            seed=sd["seed"],
            audio_freq=sd["audio_freq"],
            infer_params=infer_params,
        )

    def feed_fn(eng, elapsed):
        for sd in sessions_data:
            sid = sd["id"]
            with engine._lock:
                session = engine.sessions.get(sid)
            if session and session.state == ACTIVE:
                jitter = np.random.uniform(0, AUDIO_JITTER_MS)
                session.feed_simulated_audio(jitter_ms=jitter)

    engine.run_loop(DURATION_SECONDS, audio_feed_fn=feed_fn)
    return engine.get_metrics()


def scenario_b_dynamic_join_leave(engine, sessions_data, infer_params):
    """Scenario B: Dynamic join/leave.

    Session 0 and 1 start at t=0. Session 2 joins at t=10s.
    Session 1 leaves at t=20s. Tests dynamic session lifecycle.
    """
    print("\n[SCENARIO B] Dynamic join (t=10s) and leave (t=20s)", flush=True)

    # Start with 2 sessions
    engine.add_session(
        session_id=sessions_data[0]["id"],
        avatar_path=sessions_data[0]["avatar_path"],
        seed=sessions_data[0]["seed"],
        audio_freq=sessions_data[0]["audio_freq"],
        infer_params=infer_params,
    )
    engine.add_session(
        session_id=sessions_data[1]["id"],
        avatar_path=sessions_data[1]["avatar_path"],
        seed=sessions_data[1]["seed"],
        audio_freq=sessions_data[1]["audio_freq"],
        infer_params=infer_params,
    )

    join_time = 10.0
    leave_time = 20.0
    session2_added = False
    session1_removed = False

    def feed_fn(eng, elapsed):
        nonlocal session2_added, session1_removed

        # Join session 2 at t=10s
        if not session2_added and elapsed >= join_time:
            engine.add_session(
                session_id=sessions_data[2]["id"],
                avatar_path=sessions_data[2]["avatar_path"],
                seed=sessions_data[2]["seed"],
                audio_freq=sessions_data[2]["audio_freq"],
                infer_params=infer_params,
            )
            session2_added = True

        # Remove session 1 at t=20s
        if not session1_removed and elapsed >= leave_time:
            engine.remove_session(sessions_data[1]["id"])
            session1_removed = True

        # Feed audio to all active sessions
        for sd in sessions_data:
            sid = sd["id"]
            with engine._lock:
                session = engine.sessions.get(sid)
            if session and session.state == ACTIVE:
                jitter = np.random.uniform(0, AUDIO_JITTER_MS)
                session.feed_simulated_audio(jitter_ms=jitter)

    engine.run_loop(DURATION_SECONDS, audio_feed_fn=feed_fn)
    return engine.get_metrics()


def scenario_c_audio_gaps(engine, sessions_data, infer_params):
    """Scenario C: Audio gaps.

    Session 1 goes silent from t=10s to t=15s, then resumes.
    Tests that the engine skips silent sessions in batches and
    resumes correctly when audio returns.
    """
    print("\n[SCENARIO C] Audio gap (session 1 silent t=10s to t=15s)", flush=True)

    for sd in sessions_data:
        engine.add_session(
            session_id=sd["id"],
            avatar_path=sd["avatar_path"],
            seed=sd["seed"],
            audio_freq=sd["audio_freq"],
            infer_params=infer_params,
        )

    gap_start = 10.0
    gap_end = 15.0

    def feed_fn(eng, elapsed):
        for sd in sessions_data:
            sid = sd["id"]
            with engine._lock:
                session = engine.sessions.get(sid)
            if not session or session.state != ACTIVE:
                continue

            # Session 1 goes silent during gap period
            if sid == sessions_data[1]["id"] and gap_start <= elapsed < gap_end:
                if not session._audio_silence_log:
                    print(f"[SCENARIO C] Session {sid} going silent at t={elapsed:.1f}s", flush=True)
                    session._audio_silence_log = True
                continue

            # Log resumption
            if sid == sessions_data[1]["id"] and elapsed >= gap_end and session._audio_silence_log:
                print(f"[SCENARIO C] Session {sid} audio resumed at t={elapsed:.1f}s", flush=True)
                session._audio_silence_log = False

            jitter = np.random.uniform(0, AUDIO_JITTER_MS)
            session.feed_simulated_audio(jitter_ms=jitter)

    engine.run_loop(DURATION_SECONDS, audio_feed_fn=feed_fn)
    return engine.get_metrics()


def scenario_d_variable_freq(engine, sessions_data, infer_params):
    """Scenario D: Variable audio frequency.

    Sessions send audio at different rates:
    - Session 0: fast (every 30ms)
    - Session 1: normal (every 50ms)
    - Session 2: slow (every 100ms)
    Tests batch formation with uneven audio arrival.
    """
    print("\n[SCENARIO D] Variable audio frequency (30/50/100ms)", flush=True)

    for sd in sessions_data:
        engine.add_session(
            session_id=sd["id"],
            avatar_path=sd["avatar_path"],
            seed=sd["seed"],
            audio_freq=sd["audio_freq"],
            infer_params=infer_params,
        )

    feed_intervals = [0.030, 0.050, 0.100]
    last_feed = [0.0, 0.0, 0.0]

    def feed_fn(eng, elapsed):
        for i, sd in enumerate(sessions_data):
            sid = sd["id"]
            with engine._lock:
                session = engine.sessions.get(sid)
            if session and session.state == ACTIVE:
                if elapsed - last_feed[i] >= feed_intervals[i]:
                    jitter = np.random.uniform(0, AUDIO_JITTER_MS)
                    session.feed_simulated_audio(jitter_ms=jitter)
                    last_feed[i] = elapsed

    engine.run_loop(DURATION_SECONDS, audio_feed_fn=feed_fn)
    return engine.get_metrics()


def scenario_e_crash_recovery(engine, sessions_data, infer_params):
    """Scenario E: Session crash recovery.

    Session 2 is forced to error at t=12s (simulated by marking it errored).
    The engine should continue processing sessions 0 and 1 without interruption.
    Tests per-session error isolation.
    """
    print("\n[SCENARIO E] Crash recovery (session 2 errors at t=12s)", flush=True)

    for sd in sessions_data:
        engine.add_session(
            session_id=sd["id"],
            avatar_path=sd["avatar_path"],
            seed=sd["seed"],
            audio_freq=sd["audio_freq"],
            infer_params=infer_params,
        )

    crash_time = 12.0
    crash_triggered = False

    def feed_fn(eng, elapsed):
        nonlocal crash_triggered

        # Simulate crash for session 2
        if not crash_triggered and elapsed >= crash_time:
            sid = sessions_data[2]["id"]
            with engine._lock:
                session = engine.sessions.get(sid)
            if session:
                print(f"[SCENARIO E] Simulating crash for session {sid} at t={elapsed:.1f}s", flush=True)
                session.mark_error(RuntimeError(f"Simulated crash in session {sid}"))
                engine._hard_remove_session(sid)
                crash_triggered = True

        # Feed audio to remaining active sessions
        for sd in sessions_data:
            sid = sd["id"]
            with engine._lock:
                session = engine.sessions.get(sid)
            if session and session.state == ACTIVE:
                jitter = np.random.uniform(0, AUDIO_JITTER_MS)
                session.feed_simulated_audio(jitter_ms=jitter)

    engine.run_loop(DURATION_SECONDS, audio_feed_fn=feed_fn)
    return engine.get_metrics()


# =============================================================================
# Main Benchmark Function (runs on Modal)
# =============================================================================
@app.function(gpu="L40S", timeout=1800, secrets=[modal.Secret.from_name("huggingface-secret")])
def run_stress_test():
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/SoulX-FlashHead")

    import torch
    from PIL import Image

    os.environ["ENGINE_PROFILE"] = "1"

    # --- File + terminal logging setup ---
    class DualLogger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log_file = open(filepath, "w", buffering=1)
        def write(self, message):
            self.terminal.write(message)
            self.log_file.write(message)
        def flush(self):
            self.terminal.flush()
            self.log_file.flush()

    sys.stdout = DualLogger(LOG_FILE)
    print(f"[STRESS] Logging to file: {LOG_FILE}", flush=True)

    from flash_head.inference import (
        get_pipeline, run_pipeline, get_infer_params,
    )

    ckpt_dir = "/app/models/SoulX-FlashHead-1_3B"
    wav2vec_dir = "/app/models/wav2vec2-base-960h"
    model_type = "lite"

    print("=" * 80, flush=True)
    print("[STRESS] BatchedStreamingEngine Stress Test - L40S", flush=True)
    print(f"[STRESS] Sessions={NUM_SESSIONS}, Duration={DURATION_SECONDS}s/scenario, "
          f"WaitWindow={WAIT_WINDOW_MS}ms, Jitter={AUDIO_JITTER_MS}ms", flush=True)
    print("=" * 80, flush=True)

    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"[STRESS] GPU: {gpu_name} ({gpu_props.multi_processor_count} SMs)", flush=True)

    try:
        from sageattention import sageattn
        print("[STRESS] Attention: SageAttention", flush=True)
    except ImportError:
        pass
    try:
        import flash_attn
        print(f"[STRESS] Attention: flash_attn {flash_attn.__version__}", flush=True)
    except ImportError:
        pass

    # Load pipeline
    print("[STRESS] Loading pipeline...", flush=True)
    load_t0 = time.monotonic()
    pipeline = get_pipeline(1, ckpt_dir, model_type, wav2vec_dir)
    print(f"[STRESS] Pipeline loaded in {(time.monotonic()-load_t0)*1000:.1f}ms", flush=True)

    params = get_infer_params()

    # Create avatar images
    avatar_colors = [(128, 128, 128), (200, 100, 50), (50, 150, 200)]
    avatar_paths = []
    for i in range(NUM_SESSIONS):
        p = f"/tmp/avatar_{i}.png"
        Image.new("RGB", (512, 512), color=avatar_colors[i]).save(p)
        avatar_paths.append(p)

    # Prepare session data
    sessions_data = [
        {"id": i, "avatar_path": avatar_paths[i], "seed": 42 + i, "audio_freq": 200 + i * 100}
        for i in range(NUM_SESSIONS)
    ]

    # Warmup - must call get_base_data first to set frame_num etc. on pipeline
    print("[STRESS] Warming up...", flush=True)
    from flash_head.inference import get_base_data, get_audio_embedding
    get_base_data(pipeline, avatar_paths[0], base_seed=42, use_face_crop=False)
    for _ in range(NUM_WARMUP_SLICES):
        dummy_audio = np.zeros(params["cached_audio_duration"] * params["sample_rate"], dtype=np.float32)
        emb = get_audio_embedding(pipeline, dummy_audio,
                                  params["cached_audio_duration"] * params["tgt_fps"] - params["frame_num"],
                                  params["cached_audio_duration"] * params["tgt_fps"])
        _ = run_pipeline(pipeline, emb)
    torch.cuda.synchronize()
    print("[STRESS] Warmup complete", flush=True)

    # ==========================================
    # Run all scenarios
    # ==========================================
    all_results = {}

    # --- Scenario A: Steady ---
    torch.cuda.empty_cache()
    engine_a = BatchedStreamingEngine(pipeline, wait_window_ms=WAIT_WINDOW_MS)
    result_a = scenario_a_steady(engine_a, sessions_data, params)
    all_results["A_steady"] = result_a
    print(f"\n[SCENARIO A] Result: avg_lat={result_a['avg_latency']:.1f}ms "
          f"p95={result_a['p95_latency']:.1f}ms batch={result_a['avg_batch']:.2f} "
          f"fill=1:{result_a['fill_1']:.0f}% 2:{result_a['fill_2']:.0f}% 3:{result_a['fill_3']:.0f}%",
          flush=True)
    del engine_a
    torch.cuda.empty_cache()

    # --- Scenario B: Dynamic join/leave ---
    engine_b = BatchedStreamingEngine(pipeline, wait_window_ms=WAIT_WINDOW_MS)
    result_b = scenario_b_dynamic_join_leave(engine_b, sessions_data, params)
    all_results["B_dynamic"] = result_b
    print(f"\n[SCENARIO B] Result: avg_lat={result_b['avg_latency']:.1f}ms "
          f"p95={result_b['p95_latency']:.1f}ms batch={result_b['avg_batch']:.2f} "
          f"fill=1:{result_b['fill_1']:.0f}% 2:{result_b['fill_2']:.0f}% 3:{result_b['fill_3']:.0f}%",
          flush=True)
    del engine_b
    torch.cuda.empty_cache()

    # --- Scenario C: Audio gaps ---
    engine_c = BatchedStreamingEngine(pipeline, wait_window_ms=WAIT_WINDOW_MS)
    result_c = scenario_c_audio_gaps(engine_c, sessions_data, params)
    all_results["C_gaps"] = result_c
    print(f"\n[SCENARIO C] Result: avg_lat={result_c['avg_latency']:.1f}ms "
          f"p95={result_c['p95_latency']:.1f}ms batch={result_c['avg_batch']:.2f} "
          f"fill=1:{result_c['fill_1']:.0f}% 2:{result_c['fill_2']:.0f}% 3:{result_c['fill_3']:.0f}%",
          flush=True)
    del engine_c
    torch.cuda.empty_cache()

    # --- Scenario D: Variable frequency ---
    engine_d = BatchedStreamingEngine(pipeline, wait_window_ms=WAIT_WINDOW_MS)
    result_d = scenario_d_variable_freq(engine_d, sessions_data, params)
    all_results["D_variable"] = result_d
    print(f"\n[SCENARIO D] Result: avg_lat={result_d['avg_latency']:.1f}ms "
          f"p95={result_d['p95_latency']:.1f}ms batch={result_d['avg_batch']:.2f} "
          f"fill=1:{result_d['fill_1']:.0f}% 2:{result_d['fill_2']:.0f}% 3:{result_d['fill_3']:.0f}%",
          flush=True)
    del engine_d
    torch.cuda.empty_cache()

    # --- Scenario E: Crash recovery ---
    engine_e = BatchedStreamingEngine(pipeline, wait_window_ms=WAIT_WINDOW_MS)
    result_e = scenario_e_crash_recovery(engine_e, sessions_data, params)
    all_results["E_crash"] = result_e
    print(f"\n[SCENARIO E] Result: avg_lat={result_e['avg_latency']:.1f}ms "
          f"p95={result_e['p95_latency']:.1f}ms batch={result_e['avg_batch']:.2f} "
          f"fill=1:{result_e['fill_1']:.0f}% 2:{result_e['fill_2']:.0f}% 3:{result_e['fill_3']:.0f}%",
          flush=True)
    del engine_e
    torch.cuda.empty_cache()

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 80, flush=True)
    print("[STRESS] SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n{'Scenario':>15} | {'Cycles':>7} | {'AvgLat':>10} | {'P50':>10} | {'P95':>10} | "
          f"{'Max':>10} | {'AvgBatch':>8} | {'Fill1':>6} | {'Fill2':>6} | {'Fill3':>6} | "
          f"{'Throughput':>10}", flush=True)
    print("-" * 130, flush=True)

    for name, r in all_results.items():
        print(f"{name:>15} | {r['cycles']:>7} | {r['avg_latency']:>8.1f}ms | "
              f"{r['p50_latency']:>8.1f}ms | {r['p95_latency']:>8.1f}ms | "
              f"{r['max_latency']:>8.1f}ms | {r['avg_batch']:>8.2f} | "
              f"{r['fill_1']:>5.0f}% | {r['fill_2']:>5.0f}% | {r['fill_3']:>5.0f}% | "
              f"{r['throughput']:>8.1f}/s", flush=True)

    # GPU memory at end
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"\n[STRESS] GPU mem: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("[STRESS] Stress test complete", flush=True)
    print("=" * 80, flush=True)

    # Close log file and restore stdout
    sys.stdout.log_file.close()
    sys.stdout = sys.stdout.terminal

    # Read log content to return to local machine
    with open(LOG_FILE, "r") as f:
        log_content = f.read()

    return log_content


if __name__ == "__main__":
    log_content = run_stress_test.remote()
    local_log = os.path.join(os.path.dirname(__file__), "stress_test_results.log")
    with open(local_log, "w") as f:
        f.write(log_content)
    print(f"\n[LOCAL] Log saved to: {local_log}")
