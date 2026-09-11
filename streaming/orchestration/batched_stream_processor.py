"""
Batched stream processor: Orchestrates a streaming session that delegates
inference to a shared BatchedStreamingEngine.

Mirrors stream_processor.py structure (LiveKit connection, publishers,
WebSocket ingestion, idle video) but instead of creating a per-session
FlashHeadStreamingEngine, it registers with a shared BatchedStreamingEngine
that batches inference across all concurrent sessions.
"""

import asyncio
import contextlib
import logging
import os
import time
from queue import Queue
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from streaming.inference.batched import ACTIVE
from streaming.core.upscale import get_output_size
from streaming.transport.adapters.factory import EgressAdapterFactory
from streaming.core.audio_bus import normalize_channels, resample_audio
from streaming.core.session import CanonicalAudioFrame, MediaEgressAdapter
from streaming.protocol.messages import AudioReadyMessage, UtteranceEndedMessage
from streaming.transport.websocket_server import ws_server
from streaming.orchestration.state_manager import StreamStateManager
from streaming.orchestration.idle_video import IdleVideoLoop

_logger = logging.getLogger("batched_stream_processor")


def _short_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        return value.split("?", 1)[0].split("#", 1)[0]


def _decode_raw_pcm(audio_bytes: bytes, num_channels: int) -> np.ndarray:
    bytes_per_frame = 2 * num_channels
    if len(audio_bytes) % bytes_per_frame:
        raise ValueError(
            f"Raw PCM payload has {len(audio_bytes)} bytes, which is not aligned to "
            f"{num_channels} channel(s) of 16-bit audio"
        )
    audio_array = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if num_channels > 1:
        audio_array = audio_array.reshape(-1, num_channels).mean(axis=1)
    return audio_array


async def _audio_publish_loop(session, egress_adapter, session_id: str | None = None):
    """Publish generated audio to the selected egress sink."""
    consecutive_failures = 0
    max_consecutive_failures = 10
    while True:
        audio_chunk = await asyncio.to_thread(session.audio_queue.get)
        if audio_chunk is None:
            break
        try:
            sample_rate = getattr(egress_adapter, "sample_rate", 16_000)
            await egress_adapter.publish_audio_chunk(audio_chunk, sample_rate)
            if session_id:
                state = ws_server.active_sessions.get(session_id)
                input_adapter = state.get("input_adapter") if state else None
                send_output_audio = getattr(input_adapter, "send_output_audio", None)
                if send_output_audio is not None:
                    await send_output_audio(audio_chunk, sample_rate)
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            _logger.warning(
                "AudioPublisher push failed (%d/%d): %s",
                consecutive_failures, max_consecutive_failures, exc,
            )
            if consecutive_failures >= max_consecutive_failures:
                _logger.error(
                    "AudioPublisher exceeded max consecutive failures (%d). Aborting.",
                    max_consecutive_failures,
                )
                break


async def _drain_session(engine, session_id, session_state, keep_alive=False) -> int:
    """Wait for the engine to finish processing remaining audio for this session.

    If keep_alive=True, the session is reactivated after drain instead of being
    removed, so subsequent utterances can feed audio into the same session.

    After the engine finishes draining, we also wait for the session's
    frame_queue to be fully consumed by the video publisher. This blocks new
    utterance audio from being processed until the previous utterance's video
    frames are published, preventing A/V desync. Mirrors the old
    _drain_utterance() behavior in stream_processor.py.
    """
    session_state["is_draining"] = True
    drain_started_at = time.monotonic()
    drain_cycles = 0
    _logger.info("BATCHED_DRAIN_STARTED session=%s keep_alive=%s", session_id, keep_alive)

    try:
        engine.remove_session(session_id, keep_alive=keep_alive)
        # Wait for the engine to drain and (reactivate or remove) the session
        while True:
            session = engine.get_session(session_id)
            if session is None:
                break
            if keep_alive and session.state == ACTIVE:
                break
            drain_cycles += 1
            await asyncio.sleep(0.1)

        # Wait for all video frames to be consumed by the publisher before
        # allowing new utterance audio to be processed. This prevents the
        # engine from producing new slices (with new audio) while old frames
        # are still in the queue, which would cause A/V desync.
        session = engine.get_session(session_id) if keep_alive else None
        if session is not None:
            frame_wait_start = time.monotonic()
            while not session.frame_queue.empty():
                await asyncio.sleep(0.05)
                if time.monotonic() - frame_wait_start > 10.0:
                    _logger.warning(
                        "BATCHED_DRAIN_FRAME_WAIT_TIMEOUT session=%s qsize=%d",
                        session_id, session.frame_queue.qsize(),
                    )
                    break
            # Wait for audio_queue to also be fully consumed by the audio
            # publish loop. Without this, leftover audio from the previous
            # utterance plays alongside the next utterance's audio, causing
            # an echo when the same audio is sent (e.g. test WAV loop).
            audio_wait_start = time.monotonic()
            while not session.audio_queue.empty():
                await asyncio.sleep(0.05)
                if time.monotonic() - audio_wait_start > 10.0:
                    _logger.warning(
                        "BATCHED_DRAIN_AUDIO_WAIT_TIMEOUT session=%s qsize=%d",
                        session_id, session.audio_queue.qsize(),
                    )
                    break
            # Extra slice delay for audio source retention (AudioSource can
            # retain one slice after its matching frames leave the queue).
            slice_realtime = session.slice_len / float(session.tgt_fps)
            await asyncio.sleep(slice_realtime)
    finally:
        session_state["is_draining"] = False
        session_state["last_activity_time"] = time.monotonic()

    _logger.info(
        "BATCHED_DRAIN_COMPLETED session=%s cycles=%d elapsed=%.1fms keep_alive=%s",
        session_id, drain_cycles,
        (time.monotonic() - drain_started_at) * 1000,
        keep_alive,
    )
    return drain_cycles


async def _announce_audio_ready(session_id: str, egress_adapter: MediaEgressAdapter) -> None:
    await egress_adapter.wait_for_ready()
    session_state = ws_server.active_sessions.get(session_id)
    if session_state is not None and not session_state.get("audio_ready"):
        session_state["audio_ready"] = True
        input_adapter = session_state.get("input_adapter")
        if input_adapter is not None and input_adapter.is_negotiated:
            await ws_server.send_message(session_id, AudioReadyMessage(0))


def _cancel_batched_audio(engine, session_id: str) -> None:
    session = engine.get_session(session_id)
    if session is None:
        return
    # Mutate deques/timestamps under session lock (atomic vs inference
    # pop); queues stay lock-free.
    try:
        with session._lock:
            session.pending_audio.clear()
            session.audio_context.clear()
            session.audio_context.extend([0.0] * session.cached_audio_samples)
            session._last_slice_time = None
    except Exception:
        # Fallback for sessions without _lock (tests/mocks).
        session.pending_audio.clear()
        session.audio_context.clear()
        session.audio_context.extend([0.0] * session.cached_audio_samples)
        session._last_slice_time = None
    while not session.frame_queue.empty():
        try:
            session.frame_queue.get_nowait()
        except Exception:
            break
    while not session.audio_queue.empty():
        try:
            session.audio_queue.get_nowait()
        except Exception:
            break
    # Wake engine so reset pacing is observed promptly.
    try:
        engine._wake.set()
    except Exception:
        pass


async def run_batched_streaming_session(
    room_name: str,
    room_token: str,
    room_url: str,
    egress_type: str | None = None,
    engine: any = None,
    source_image: str = None,
    session_id: str = None,
    ingestion_token: str = None,
    idle_video_url: str = None,
    idle_video_key: str = None,
    seed: int = 42,
    room_metadata: dict | None = None,
):
    """
    Run a streaming session with batched inference via BatchedStreamingEngine.

    Args:
        room_name: Transport room name or session label
        room_token: Transport token for authentication
        room_url: Transport room URL or endpoint
        egress_type: Explicit transport type, or None for URL detection
        room_metadata: Explicit transport metadata such as Agora appId, channelName, and uid
        engine: Shared BatchedStreamingEngine instance
        source_image: URL or path to the avatar/source image
        session_id: Unique session identifier
        ingestion_token: Token for websocket ingestion
        idle_video_url: URL for idle video loop
        idle_video_key: Immutable idle asset key for the default snapshot cache
        seed: Random seed for avatar generation
    """
    import time

    os.environ.setdefault("LIVEKIT_RTC_DEBUG", "false")
    _rtc_debug = os.environ.get("LIVEKIT_RTC_DEBUG", "false").strip().lower() in ("true", "1")
    for _lk_name in ("livekit", "livekit.rtc", "livekit.api"):
        logging.getLogger(_lk_name).setLevel(logging.DEBUG if _rtc_debug else logging.INFO)

    import threading as _th
    _loop = asyncio.get_event_loop()
    _logger.info(
        "[BATCHED-DIAG] threading context: thread_id=%s thread_name=%s event_loop_id=%s",
        _th.get_ident(), _th.current_thread().name, id(_loop),
    )

    _diag_events = []
    _diag_t0 = time.monotonic()

    def _diag(ev: str, **kwargs):
        entry = {"event": ev, "t": round(time.monotonic() - _diag_t0, 3), **kwargs}
        _diag_events.append(entry)
        _logger.info("[BATCHED-DIAG] %s", entry)

    def _pub_sid(pub):
        return getattr(pub, "sid", getattr(pub, "track_sid", "unknown"))

    _logger.info(
        "Connecting to egress type=%s room=%s url=%s session=%s",
        egress_type or "auto",
        room_name,
        _short_url(room_url),
        session_id,
    )
    room = None

    from flash_head.inference import get_infer_params

    infer_params = get_infer_params()
    output_width, output_height = get_output_size()
    num_channels = int(os.getenv("AUDIO_CHANNELS", "1"))
    idle_timeout_ms = int(os.getenv("IDLE_TIMEOUT_MS", "500"))

    # The session's frame_queue will be created by BatchedSession inside the engine.
    # We need to get it after registration. Use a placeholder that we'll replace.
    live_frame_queue = Queue()  # temporary, replaced after engine registration
    publish_task = None
    audio_publish_task = None
    egress_adapter = None
    input_adapter = None
    audio_bus = None
    audio_ready_task = None
    batched_session = None

    try:
        # Start egress connection as early as possible so the WebRTC ICE/DTLS
        # handshake overlaps with idle video setup and avatar preparation below.
        room_metadata = room_metadata if isinstance(room_metadata, dict) else {}
        room_config = {
            "type": egress_type,
            "url": room_url,
            "token": room_token,
            "room_name": room_name,
            "app_id": room_metadata.get("app_id") or room_metadata.get("appId"),
            "channel_name": room_metadata.get("channel_name") or room_metadata.get("channelName"),
            "uid": room_metadata.get("uid"),
            "fps": int(infer_params["tgt_fps"]),
            "sample_rate": int(infer_params["sample_rate"]),
            "channels": num_channels,
            "session_id": session_id or "",
            "width": output_width,
            "height": output_height,
        }
        egress_adapter = EgressAdapterFactory.create(egress_type, room_config)
        try:
            from streaming.transport.adapters.livekit_egress import LiveKitEgressAdapter

            if isinstance(egress_adapter, LiveKitEgressAdapter):
                egress_adapter.start_probe_early()
        except Exception:
            pass
        connect_task = asyncio.create_task(
            egress_adapter.connect(room_config),
            name=f"egress-connect-{session_id}",
        )

        async def _setup_idle():
            idle_loop = None
            if idle_video_key or idle_video_url:
                from utils.default_idle_video_cache import default_idle_video_cache
                cached_idle_bytes = default_idle_video_cache.get_bytes(idle_video_key)
                idle_loop = await asyncio.to_thread(
                    IdleVideoLoop, idle_video_url, video_bytes=cached_idle_bytes,
                )
                idle_loop.normalize(output_width, output_height)
            if idle_loop is None or not idle_loop.is_valid():
                idle_loop = await asyncio.to_thread(
                    IdleVideoLoop.fallback_from_source_image,
                    source_image, output_width, output_height,
                )
                if idle_loop.is_valid():
                    _logger.info("IDLE_FALLBACK_PUBLISHED session=%s", session_id)
            return idle_loop

        async def _register_engine():
            # Register session with the engine (this creates the BatchedSession
            # and prepares the avatar on the pipeline)
            _logger.info("Registering session %s with BatchedStreamingEngine", session_id)
            prep_t0 = time.monotonic()
            try:
                _batched_session = await asyncio.to_thread(
                    engine.add_session, session_id, source_image, seed, infer_params,
                )
            except ValueError:
                # Duplicate registration (add_session raises): reuse existing
                # instead of crashing on None (old None-return path).
                _logger.warning("Session %s already registered, reusing existing", session_id)
                _batched_session = engine.get_session(session_id)
                if _batched_session is None:
                    raise
            prep_ms = round((time.monotonic() - prep_t0) * 1000, 1)
            _logger.info(
                "Session %s registered with engine in %.1f ms (avatar prepared)",
                session_id, prep_ms,
            )
            return _batched_session

        idle_task = asyncio.create_task(_setup_idle(), name=f"idle-setup-{session_id}")
        register_task = asyncio.create_task(_register_engine(), name=f"engine-register-{session_id}")

        try:
            idle_loop, batched_session, _ = await asyncio.gather(
                idle_task, register_task, connect_task
            )
        except Exception:
            # Cancel any unfinished tasks so a failed connect doesn't leave a
            # partially prepared engine session or idle download running.
            for t in (connect_task, idle_task, register_task):
                if t is not None and not t.done():
                    t.cancel()
            with contextlib.suppress(Exception):
                cancel = getattr(egress_adapter, "cancel_probe_early", None)
                if callable(cancel):
                    cancel()
            await asyncio.gather(
                connect_task, idle_task, register_task, return_exceptions=True
            )
            raise

        # Now use the session's actual frame_queue for the state manager
        live_frame_queue = batched_session.frame_queue

        state_manager = StreamStateManager(
            live_frame_queue=live_frame_queue,
            idle_video=idle_loop,
            idle_timeout_ms=idle_timeout_ms,
            engine=engine,
        )

        room = getattr(egress_adapter, "room", None)
        publish_task = egress_adapter.start_video(state_manager)
        audio_ready_task = asyncio.create_task(
            _announce_audio_ready(session_id, egress_adapter),
            name=f"audio-ready-{session_id}",
        )

        if idle_loop and idle_loop.is_valid():
            try:
                await asyncio.wait_for(egress_adapter.wait_for_ready(), timeout=15)
                _logger.info("IDLE_TRACK_PUBLISHED session=%s room=%s", session_id, room_name)
            except asyncio.TimeoutError:
                raise RuntimeError("Timed out publishing the idle video track")

        audio_publish_task = asyncio.create_task(
            _audio_publish_loop(batched_session, egress_adapter, session_id)
        )

        audio_bus = ws_server.get_audio_bus(session_id)
        input_adapter = ws_server.get_input_adapter(session_id)
        audio_queue = audio_bus.queue
        session_state = ws_server.active_sessions[session_id]
        session_ended = False

        if egress_adapter.first_frame_published is not None and egress_adapter.first_frame_published.is_set():
            session_state["audio_ready"] = True
            if input_adapter.is_negotiated:
                await ws_server.send_message(session_id, AudioReadyMessage(0))

        while True:
            audio_event = await audio_queue.get()
            if audio_event is None:
                session_ended = True
                break
            if not isinstance(audio_event, CanonicalAudioFrame):
                continue
            if audio_event.is_cancelled:
                _cancel_batched_audio(engine, session_id)
                session_state["is_draining"] = False
                continue
            if audio_event.is_end_of_utterance:
                cycles = await _drain_session(engine, session_id, session_state, keep_alive=True)
                ended_seq = audio_event.protocol_sequence_number if audio_event.protocol_sequence_number is not None else audio_event.sequence_number
                await ws_server.send_message(
                    session_id,
                    UtteranceEndedMessage(ended_seq, cycles),
                )
                continue
            normalized = normalize_channels(audio_event.pcm, audio_event.channels, 1).reshape(-1)
            await asyncio.to_thread(
                engine.feed_audio,
                session_id,
                resample_audio(normalized, audio_event.sample_rate, engine.get_session(session_id).sample_rate),
            )

        if session_ended:
            await _drain_session(engine, session_id, session_state)

    finally:
        # Signal audio publish loop to stop
        if batched_session is not None:
            batched_session.audio_queue.put_nowait(None)

        if audio_publish_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await audio_publish_task

        # Remove session from engine if still present
        if engine is not None:
            engine.remove_session(session_id)

        if audio_ready_task is not None:
            audio_ready_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await audio_ready_task
        if publish_task is not None:
            publish_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await publish_task
        if input_adapter is not None:
            with contextlib.suppress(Exception):
                await input_adapter.disconnect()
        if egress_adapter is not None:
            with contextlib.suppress(Exception):
                await egress_adapter.disconnect()
        session_state = ws_server.active_sessions.get(session_id)
        idle_watcher = session_state.get("idle_watcher") if session_state else None
        if idle_watcher is not None and idle_watcher is not asyncio.current_task():
            idle_watcher.cancel()
        ws_server.unregister_session(session_id)
