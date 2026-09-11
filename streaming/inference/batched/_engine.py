"""BatchedStreamingEngine - Central batching engine."""

import os
import time
import logging
import threading
from urllib.parse import urlparse

import numpy as np

from utils.default_avatar_cache import default_avatar_cache
from streaming.core.upscale import get_output_size, upscale_slice_torch
from flash_head.inference import get_base_data, run_pipeline_batch

from ._avatar import _resolve_avatar_path
from ._session_states import (
    ACTIVE,
    DRAINING,
    IDLE_TIMEOUT_S,
    REACTIVATION_TIMEOUT_S,
    WAIT_WINDOW_MS,
)
from ._batched_session import BatchedSession

logger = logging.getLogger("BatchedStreamingEngine")


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
        # _avatar_lock serializes pipeline.prepare_params (get_base_data)
        # which mutates pipeline.frame_num / motion_frames_num / timesteps /
        # ref latents that generate_batch reads. get_base_data + clone are
        # done under _avatar_lock so two prepares never run concurrently and
        # a prepare cannot clobber pipeline state mid-clone.
        # Lock ordering (avoid deadlock):
        #   engine._lock -> session._lock,
        #   engine._avatar_lock -> session._lock,
        # never session -> engine, never hold _lock + _avatar_lock together.
        # add_session holds _lock only for dict reserve/publish (us), never
        # for network download or GPU prepare, so inference latency is flat.
        self._avatar_lock = threading.Lock()
        # _wake coalesces producer events (feed/add/remove/reactivate/cancel)
        # so _run_loop sleeps with dynamic timeout instead of 1ms polling.
        self._wake = threading.Event()
        self._cycle_count = 0
        self._batch_sizes = []
        self._stopped = False
        self._all_latencies = []
        self._thread = None
        # Global batch tick (monotonic timestamp of the next scheduled batch).
        # All ready sessions are processed together once per slice_realtime.
        self._next_batch_time = None

    def add_session(self, session_id, avatar_path, seed, infer_params):
        """Register and prepare a new session. Can be called mid-stream.

        Latency-safe split:
        1) with _lock: duplicate check + insert INITIALIZING placeholder.
        2) outside any lock: _resolve_avatar_path download (network).
        3) with _avatar_lock only: get_base_data + clone.
        4) with _lock: publish ACTIVE if still present else cleanup + abort.
        Raises ValueError on duplicate (caller must handle), re-raises
        prepare failures after pop + close + temp unlink.
        WAIT_WINDOW_MS, tick grid and off-tick fresh fast path unchanged.
        """
        with self._lock:
            if session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")
            session = BatchedSession(
                session_id=session_id,
                pipeline=self.pipeline,
                avatar_path=avatar_path,
                seed=seed,
                infer_params=infer_params,
            )
            self.sessions[session_id] = session
            logger.info("Session %s reserving INITIALIZING (total=%d)", session_id, len(self.sessions))

        local_path = None
        is_temp = False
        try:
            # Phase 2: network/download outside all locks.
            local_path = _resolve_avatar_path(avatar_path)
            parsed = urlparse(avatar_path)
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

            # Phase 3: GPU prepare + clone under _avatar_lock only.
            with self._avatar_lock:
                get_base_data(self.pipeline, local_path, base_seed=seed, use_face_crop=False)
                # Clone under avatar + session locks (avatar -> session order;
                # never reverse, never hold engine _lock here).
                with session._lock:
                    session.ref_img_latent = self.pipeline.ref_img_latent.clone()
                    session.latent_motion_frames = self.pipeline.latent_motion_frames.clone()
                    session.original_color_reference = self.pipeline.original_color_reference.clone()
                    session.color_correction_strength = self.pipeline.color_correction_strength
                    if is_temp:
                        session._temp_avatar_path = local_path

            # Phase 4: publish ACTIVE under engine lock (engine -> session).
            with self._lock:
                current = self.sessions.get(session_id)
                if current is not session:
                    raise RuntimeError(f"Session {session_id} removed during avatar preparation")
                with session._lock:
                    # Keep temp assignment idempotent (already set above).
                    if is_temp:
                        session._temp_avatar_path = local_path
                    session.state = ACTIVE
                logger.info("Session %s added ACTIVE (total=%d)", session_id, len(self.sessions))
            self._wake.set()
            return session
        except Exception:
            # On prepare failure: pop if still ours, close, unlink temp, re-raise.
            # Duplicate ValueError never reaches here (raised before reserve).
            with self._lock:
                current = self.sessions.get(session_id)
                if current is session:
                    self.sessions.pop(session_id, None)
            if local_path and is_temp:
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except Exception:
                    pass
            try:
                # Avoid clearing shared cache path: close() only unlinks its
                # own _temp_avatar_path, which is None if not temp.
                session.close()
            except Exception:
                pass
            raise

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
            # keep_alive flip needs session lock (engine -> session order).
            with session._lock:
                session.keep_alive = keep_alive
            session.start_drain()
            logger.info("Session %s marked for drain (keep_alive=%s)", session_id, keep_alive)
        # Wake loop so drain is observed without waiting for poll timeout.
        self._wake.set()

    def _hard_remove_session(self, session_id):
        """Actually remove and close a session from the registry."""
        with self._lock:
            session = self.sessions.pop(session_id, None)
        if session:
            try:
                with session._lock:
                    lat_copy = list(session.latencies)
            except Exception:
                lat_copy = []
            self._all_latencies.extend(lat_copy)
            session.close()
            logger.info("Session %s removed (total=%d)", session_id, len(self.sessions))

    def feed_audio(self, session_id, audio_data):
        """Feed audio to a specific session."""
        with self._lock:
            session = self.sessions.get(session_id)
        if session:
            session.feed_audio(audio_data)
            self._wake.set()

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
        self._wake.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="batched-engine")
        self._thread.start()
        logger.info("BatchedStreamingEngine background thread started")

    def stop(self):
        """Stop the inference loop and drain all sessions."""
        self._stopped = True
        # Wake loop so it observes _stopped without waiting for timeout.
        try:
            self._wake.set()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        logger.info("BatchedStreamingEngine stopped")

    def _run_loop(self):
        """Background loop: run inference cycles until stopped.

        Idle wait uses event + dynamic timeout min(time_to_next_tick, 20ms)
        instead of fixed 1ms poll. WAIT_WINDOW_MS, 5ms coalescing and
        off-tick fresh fast path are preserved in run_inference_cycle.
        """
        while not self._stopped:
            batch_size, _ = self.run_inference_cycle()
            if batch_size == 0 and not self._stopped:
                if self._next_batch_time is None:
                    timeout = 0.02
                else:
                    now = time.monotonic()
                    if self._next_batch_time <= now:
                        timeout = 0.02
                    else:
                        timeout = min(self._next_batch_time - now, 0.02)
                # Clear immediately before wait to avoid lost wake during
                # inference. If producers set _wake during the cycle above,
                # clearing discards it, but the cycle just ran and observed
                # (or will observe on next tick) that work; timeout bounds
                # the delay to 20ms. A set after clear wakes wait immediately.
                self._wake.clear()
                if not self._stopped:
                    self._wake.wait(timeout)

        # Drain all remaining sessions on shutdown
        logger.info("Engine shutting down, draining remaining sessions...")
        with self._lock:
            snapshot = list(self.sessions.values())
        for session in snapshot:
            session.start_drain()
        # Wake in case any waiter depends on drain progress.
        try:
            self._wake.set()
        except Exception:
            pass

        drain_start = time.monotonic()
        while True:
            with self._lock:
                has_sessions = bool(self.sessions)
            if not has_sessions:
                break
            if (time.monotonic() - drain_start) >= 10:
                break
            batch_size, _ = self.run_inference_cycle()
            if batch_size == 0:
                # Remove fully-drained sessions (no pending audio) under
                # engine -> session ordering to avoid losing racing audio.
                with self._lock:
                    sids = list(self.sessions.keys())
                for sid in sids:
                    with self._lock:
                        session = self.sessions.get(sid)
                    if session is None:
                        continue
                    with session._lock:
                        empty = not session.pending_audio
                    if empty:
                        # Re-check under both locks to close the race where
                        # feed_audio arrives between check and removal.
                        remove = False
                        with self._lock:
                            cur = self.sessions.get(sid)
                            if cur is session:
                                with session._lock:
                                    if not session.pending_audio and session.state == DRAINING and not session.keep_alive:
                                        remove = True
                                    # keep_alive reactivation is handled in
                                    # run_inference_cycle cleanup; do not force-remove here.
                        if remove:
                            self._hard_remove_session(sid)
                with self._lock:
                    if not self.sessions:
                        break
                time.sleep(0.01)

        # Force-close any remaining (engine lock only for dict, session locks in close).
        with self._lock:
            remaining = list(self.sessions.values())
            self.sessions.clear()
        for session in remaining:
            try:
                with session._lock:
                    lat_copy = list(session.latencies)
            except Exception:
                lat_copy = []
            self._all_latencies.extend(lat_copy)
            try:
                session.close()
            except Exception:
                pass

        logger.info("Engine drain complete: %d cycles total", self._cycle_count)

    def run_inference_cycle(self):
        """Run one batching cycle. Returns (batch_size, cycle_latency_ms) or (0, 0).

        Thread-safety: engine _lock only for sessions dict; per-session state
        under session._lock (engine -> session order, never reverse). Queues
        stay lock-free. WAIT_WINDOW_MS, tick grid, 5ms coalescing and
        off-tick fresh fast path are unchanged.
        """
        import torch

        # Snapshot dict under engine lock only (no session locks held).
        with self._lock:
            snapshot_items = list(self.sessions.items())

        active_sessions = {}
        for sid, session in snapshot_items:
            # Atomic idle check + drain mark under session lock.
            try:
                drained, had_audio = session.try_mark_idle_drain()
            except Exception:
                drained, had_audio = False, False
            if drained:
                if had_audio:
                    logger.info("Session %s idle timeout (%ds), removing", sid, IDLE_TIMEOUT_S)
                else:
                    logger.info("Session %s reactivation timeout (%ds), removing", sid, REACTIVATION_TIMEOUT_S)
                # Wake so drain progress is observed promptly.
                try:
                    self._wake.set()
                except Exception:
                    pass
                continue
            try:
                with session._lock:
                    is_active = (session.state == ACTIVE)
            except Exception:
                is_active = False
            if is_active:
                active_sessions[sid] = session

        # Handle draining sessions - process remaining audio.
        # Check state + pending atomically under session lock.
        draining_sessions = {}
        for sid, session in snapshot_items:
            try:
                with session._lock:
                    if session.state == DRAINING and session.pending_audio:
                        draining_sessions[sid] = session
            except Exception:
                continue

        # Include draining sessions that have enough audio for one final slice.
        # flush_one pads under session lock; length check inside flush is atomic.
        for sid, session in list(draining_sessions.items()):
            try:
                with session._lock:
                    needs_flush = (
                        session.state == DRAINING
                        and bool(session.pending_audio)
                        and len(session.pending_audio) < session.slice_samples
                    )
                if needs_flush:
                    session.flush_one()
            except Exception:
                continue

        # Clean up drained sessions that have no pending audio.
        # keep_alive sessions are reactivated instead of removed.
        # This runs before the early return so empty DRAINING sessions
        # don't hang forever when no other sessions are ready.
        # Re-check empty under engine -> session ordering to avoid losing
        # racing feed_audio between check and removal.
        with self._lock:
            cleanup_items = list(self.sessions.items())
        for sid, session in cleanup_items:
            try:
                with session._lock:
                    empty_draining = (session.state == DRAINING and not session.pending_audio)
                    keep = session.keep_alive if empty_draining else False
            except Exception:
                continue
            if not empty_draining:
                continue
            if keep:
                session.reactivate()
                try:
                    self._wake.set()
                except Exception:
                    pass
                logger.info("Session %s reactivated (keep_alive)", sid)
            else:
                # Atomic re-check + pop under engine -> session.
                should_remove = False
                lat_copy = []
                with self._lock:
                    cur = self.sessions.get(sid)
                    if cur is session:
                        with session._lock:
                            if session.state == DRAINING and not session.pending_audio and not session.keep_alive:
                                should_remove = True
                                try:
                                    lat_copy = list(session.latencies)
                                except Exception:
                                    lat_copy = []
                                self.sessions.pop(sid, None)
                if should_remove:
                    self._all_latencies.extend(lat_copy)
                    try:
                        session.close()
                    except Exception:
                        pass
                    logger.info("Session %s removed (total=%d)", sid, len(self.sessions))

        # Remove hard-removed sessions from draining_sessions snapshot.
        with self._lock:
            valid_sids = set(self.sessions.keys())
        draining_sessions = {
            sid: s for sid, s in draining_sessions.items()
            if sid in valid_sids
        }
        # Also drop active entries that were removed concurrently.
        active_sessions = {
            sid: s for sid, s in active_sessions.items()
            if sid in valid_sids
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
            # Off-tick fast path: only fresh sessions (no slice yet) run.
            fresh_ready = []
            for sid, s in ready_sessions:
                try:
                    if s.is_fresh():
                        fresh_ready.append((sid, s))
                except Exception:
                    continue
            ready_sessions = fresh_ready
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

        # Generate audio embeddings for ready sessions.
        # pop_audio_slice re-checks readiness under session lock so a
        # concurrent cancel/clear between is_slice_ready and pop cannot
        # cause IndexError; skipped sessions are dropped from this batch.
        t_embed = time.monotonic()
        audio_embs = []
        audio_chunks = []
        still_ready = []
        for sid, session in ready_sessions:
            try:
                emb, chunk = session.get_audio_embedding()
            except RuntimeError:
                continue
            except IndexError:
                continue
            audio_embs.append(emb)
            audio_chunks.append(chunk)
            still_ready.append((sid, session))
        ready_sessions = still_ready
        if not ready_sessions:
            return 0, 0.0

        batch_size = len(ready_sessions)
        self._batch_sizes.append(batch_size)
        self._cycle_count += 1
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
            # NOTE: audio for this tick was already popped from pending_audio via
            # get_audio_embedding() above, so even a successful retry leaves a
            # gap for the failed tick. Keep hard_remove semantics as default;
            # attempt a single bounded retry after empty_cache first (do NOT
            # blindly continue survivors without retry).
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                logger.warning(
                    "Batch inference retrying once after empty_cache batch=%d",
                    batch_size,
                )
                frames_list, updated_mf_list = run_pipeline_batch(
                    self.pipeline,
                    audio_embs,
                    current_mf,
                    current_ref_latents,
                    current_gens,
                    current_refs,
                    current_ccs,
                )
                logger.info(
                    "Batch inference retry succeeded batch=%d "
                    "(gap: failed tick audio already popped)",
                    batch_size,
                )
            except Exception as retry_exc:
                logger.error("Batch inference retry failed: %s", retry_exc, exc_info=True)
                for sid, session in ready_sessions:
                    session.mark_error(retry_exc)
                    self._hard_remove_session(sid)
                return batch_size, (time.monotonic() - cycle_t0) * 1000

        infer_ms = (time.monotonic() - t_infer) * 1000

        # Transfer to CPU and distribute frames per session
        t_xfer = time.monotonic()
        out_w, out_h = get_output_size()
        for i, (sid, session) in enumerate(ready_sessions):
            try:
                video = frames_list[i][session.motion_frames_num:]
                video = upscale_slice_torch(video, out_w, out_h)
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
            try:
                with s._lock:
                    lat_copy = list(s.latencies)
            except Exception:
                continue
            all_latencies.extend(lat_copy)

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
