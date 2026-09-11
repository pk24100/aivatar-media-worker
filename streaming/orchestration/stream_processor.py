# Orchestrates the full streaming session: LiveKit connection, FlashHead engine, audio/video publishers, and ingestion handlers.
import asyncio
import contextlib
import logging
import os
import time
from queue import Queue
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from streaming.inference.flashhead import FlashHeadStreamingEngine
from streaming.core.upscale import get_output_size
from streaming.transport.adapters.factory import EgressAdapterFactory
from streaming.core.audio_bus import normalize_channels, resample_audio
from streaming.core.session import CanonicalAudioFrame, MediaEgressAdapter
from streaming.protocol.messages import AudioReadyMessage, UtteranceEndedMessage
from streaming.transport.websocket_server import ws_server
from streaming.orchestration.state_manager import StreamStateManager
from streaming.orchestration.idle_video import IdleVideoLoop

_logger = logging.getLogger("stream_processor")


def _short_url(value: str) -> str:
    """Remove query strings and fragments, which can contain long credentials."""
    try:
        parsed = urlsplit(value)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        return value.split("?", 1)[0].split("#", 1)[0]


def _decode_raw_pcm(audio_bytes: bytes, num_channels: int) -> np.ndarray:
    """Decode the WebSocket ingestion contract: signed 16-bit little-endian PCM."""
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


async def _drain_utterance(engine, session_id: str, session_state: dict) -> int:
    """Finish queued speech without ending the persistent media session."""
    had_pending_media = bool(engine.pending_audio) or not engine.frame_queue.empty()
    session_state["is_draining"] = True
    drain_started_at = time.monotonic()
    drain_cycles = 0
    _logger.info("UTTERANCE_DRAIN_STARTED session=%s", session_id)
    # Bounded drain: slice_realtime is 24/25=0.96s. Frame/audio waits are capped
    # at 2x slice_realtime (~1.92s, never 10s) so UtteranceEnded is never blocked
    # beyond ~2s by a stuck consumer. Log+break mirrors batched drain.
    try:
        _slice_realtime = float(engine.slice_len) / float(engine.tgt_fps)
    except Exception:
        _slice_realtime = 24.0 / 25.0
    _drain_timeout = 2.0 * _slice_realtime

    try:
        while True:
            has_pending_audio = await asyncio.to_thread(engine.flush)
            drain_cycles += 1
            if not has_pending_audio:
                break
            await asyncio.sleep(engine.next_slice_delay())

        _frame_wait_start = time.monotonic()
        while not engine.frame_queue.empty():
            if (time.monotonic() - _frame_wait_start) > _drain_timeout:
                try:
                    _qsize = engine.frame_queue.qsize()
                except Exception:
                    _qsize = -1
                _logger.warning(
                    "UTTERANCE_DRAIN_FRAME_WAIT_TIMEOUT session=%s qsize=%s",
                    session_id, _qsize,
                )
                break
            await asyncio.sleep(0.05)

        _audio_wait_start = time.monotonic()
        while not engine.audio_queue.empty():
            if (time.monotonic() - _audio_wait_start) > _drain_timeout:
                try:
                    _aqsize = engine.audio_queue.qsize()
                except Exception:
                    _aqsize = -1
                _logger.warning(
                    "UTTERANCE_DRAIN_AUDIO_WAIT_TIMEOUT session=%s qsize=%s",
                    session_id, _aqsize,
                )
                break
            await asyncio.sleep(0.05)

        # AudioSource can retain one slice after its matching frames leave the queue.
        if had_pending_media:
            await asyncio.sleep(engine.slice_len / float(engine.tgt_fps))
    finally:
        session_state["is_draining"] = False
        session_state["last_activity_time"] = time.monotonic()

    _logger.info(
        "UTTERANCE_DRAIN_COMPLETED session=%s cycles=%d elapsed=%.1fms",
        session_id,
        drain_cycles,
        (time.monotonic() - drain_started_at) * 1000,
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


def _cancel_engine_audio(engine) -> None:
    engine.pending_audio.clear()
    engine.audio_context.clear()
    engine.audio_context.extend([0.0] * engine.cached_audio_samples)
    engine._last_slice_time = None
    while not engine.frame_queue.empty():
        try:
            engine.frame_queue.get_nowait()
        except Exception:
            break
    while not engine.audio_queue.empty():
        try:
            engine.audio_queue.get_nowait()
        except Exception:
            break


async def _audio_publish_loop(engine, egress_adapter, session_id: str | None = None):
    """Publish generated audio to the selected egress sink."""
    consecutive_failures = 0
    max_consecutive_failures = 10
    while True:
        audio_chunk = await asyncio.to_thread(engine.audio_queue.get)
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
                    "AudioPublisher exceeded max consecutive failures (%d). "
                    "Aborting audio publish loop.",
                    max_consecutive_failures,
                )
                break



async def run_streaming_session(
    room_name: str,
    room_token: str,
    room_url: str,
    egress_type: str | None = None,
    pipeline: any = None,
    model_pool: any = None,
    model_ready_waiter: any = None,
    source_image: str = None,
    session_id: str = None,
    ingestion_token: str = None,
    idle_video_url: str = None,
    idle_video_key: str = None,
    room_metadata: dict | None = None,
):
    return await _run_streaming_session_native(
        room_name, room_token, room_url, egress_type, pipeline, model_pool, model_ready_waiter, source_image,
        session_id, ingestion_token, idle_video_url, idle_video_key,
        room_metadata=room_metadata)

async def _run_streaming_session_native(
    room_name: str,
    room_token: str,
    room_url: str,
    egress_type: str | None = None,
    pipeline: any = None,
    model_pool: any = None,
    model_ready_waiter: any = None,
    source_image: str = None,
    session_id: str = None,
    ingestion_token: str = None,
    idle_video_url: str = None,
    idle_video_key: str = None,
    room_metadata: dict | None = None,
):
    """
    Run a streaming session with FlashHead Lite model.
    
    Args:
        room_name: LiveKit room name
        room_token: Transport token for authentication
        room_url: Transport room URL or endpoint
        egress_type: Explicit transport type, or None for URL detection
        pipeline: Optional pre-acquired FlashHeadPipeline instance
        model_pool: Pool used after idle publishing when no pipeline is supplied
        model_ready_waiter: Awaitable that gates GPU model initialization
        source_image: URL or path to the avatar/source image
        session_id: Unique session identifier
        ingestion_token: Token for websocket ingestion
        idle_video_url: URL for idle video loop
        idle_video_key: Immutable idle asset key for the default snapshot cache
    """
    import time

    # Enable Rust FFI debug logs (ICE, DTLS, etc.) — checked at callback time
    os.environ.setdefault("LIVEKIT_RTC_DEBUG", "false")
    # Rust FFI log level — set to DEBUG for ICE/DTLS diagnosis, INFO for production.
    # DEBUG confirmed in logs7.txt that UDP to LiveKit's IP range (161.115.180.x)
    # is blocked (error 101/ENETUNREACH) and connection succeeds via TCP port 7881.
    _rtc_debug = os.environ.get("LIVEKIT_RTC_DEBUG", "false").strip().lower() in ("true", "1")
    for _lk_name in ("livekit", "livekit.rtc", "livekit.api"):
        logging.getLogger(_lk_name).setLevel(logging.DEBUG if _rtc_debug else logging.INFO)

    # --- Threading context diagnostic (Modal runtime) ---
    import threading as _th
    _loop = asyncio.get_event_loop()
    _logger.info(
        "[NATIVE-DIAG] threading context: thread_id=%s thread_name=%s event_loop_id=%s "
        "MODAL_RUNTIME=%s",
        _th.get_ident(), _th.current_thread().name, id(_loop),
        os.environ.get("MODAL_RUNTIME", "unset"),
    )

    # Capture every observable event for root-cause diagnosis
    _diag_events = []
    _diag_t0 = time.monotonic()

    def _diag(ev: str, **kwargs):
        entry = {"event": ev, "t": round(time.monotonic() - _diag_t0, 3), **kwargs}
        _diag_events.append(entry)
        _logger.info("[NATIVE-DIAG] %s", entry)

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

    # We can publish a target-sized idle track before touching a GPU pipeline.
    # This lets viewers see the same LiveKit track while model warmup and avatar
    # preparation complete in the background.
    from flash_head.inference import get_infer_params

    infer_params = get_infer_params()
    output_width, output_height = get_output_size()
    num_channels = int(os.getenv("AUDIO_CHANNELS", "1"))
    chunk_timeout = float(os.getenv("AUDIO_SUBSCRIBE_TIMEOUT", "15"))
    idle_timeout_ms = int(os.getenv("IDLE_TIMEOUT_MS", "500"))
    live_frame_queue = Queue()
    engine = None
    publish_task = None
    audio_publish_task = None
    egress_adapter = None
    input_adapter = None
    audio_bus = None
    audio_ready_task = None
    rtc_metrics_task = None
    _prep_state = None
    try:
        # Build the egress adapter and start the LiveKit connection immediately
        # so ICE/DTLS negotiation overlaps with model acquisition, idle video
        # loading and avatar preparation.
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

        # Shared holder so the finally block can release a pipeline that was
        # acquired even if _prepare_model fails after acquisition.
        _prep_state = {"pipeline": pipeline, "engine": None, "sample_rate": None}

        # Load a default snapshot-cached clip when available. Custom clips are
        # intentionally fetched per session and decoded outside the aiohttp loop.
        async def _setup_idle():
            idle_loop = None
            if idle_video_key or idle_video_url:
                from utils.default_idle_video_cache import default_idle_video_cache

                cached_idle_bytes = default_idle_video_cache.get_bytes(idle_video_key)
                idle_loop = await asyncio.to_thread(
                    IdleVideoLoop,
                    idle_video_url,
                    video_bytes=cached_idle_bytes,
                )
                idle_loop.normalize(output_width, output_height)
            if idle_loop is None or not idle_loop.is_valid():
                idle_loop = await asyncio.to_thread(
                    IdleVideoLoop.fallback_from_source_image,
                    source_image,
                    output_width,
                    output_height,
                )
                if idle_loop.is_valid():
                    _logger.info("IDLE_FALLBACK_PUBLISHED session=%s", session_id)
            return idle_loop

        # Acquire the pipeline and prepare the avatar while LiveKit connects.
        async def _prepare_model():
            if model_ready_waiter is not None:
                await model_ready_waiter()
            if _prep_state["pipeline"] is None:
                if model_pool is None:
                    raise RuntimeError("pipeline or model_pool is required for streaming")
                _prep_state["pipeline"] = await model_pool.acquire()
            _pipeline = _prep_state["pipeline"]

            if not source_image:
                raise ValueError("source_image is required for FlashHead streaming")
            _engine = FlashHeadStreamingEngine(
                pipeline=_pipeline,
                avatar_image_path=source_image,
                auto_prepare_avatar=False,
                frame_queue=live_frame_queue,
            )
            _prep_state["engine"] = _engine
            _prep_state["sample_rate"] = _engine.sample_rate

            prep_t0 = time.monotonic()
            await asyncio.to_thread(_engine.prepare_avatar, source_image)
            _logger.info(
                "Prepared avatar for session=%s source=%s in %.1f ms",
                session_id,
                source_image,
                (time.monotonic() - prep_t0) * 1000,
            )
            return _engine, _prep_state["sample_rate"]

        idle_task = asyncio.create_task(_setup_idle(), name=f"idle-setup-{session_id}")
        model_task = asyncio.create_task(_prepare_model(), name=f"model-prep-{session_id}")

        try:
            idle_loop, (engine, sample_rate), _ = await asyncio.gather(
                idle_task, model_task, connect_task
            )
        except Exception:
            # Cancel any unfinished background tasks so a failed connect doesn’t
            # leave a partially-prepared engine or idle download running.
            for t in (connect_task, idle_task, model_task):
                if t is not None and not t.done():
                    t.cancel()
            with contextlib.suppress(Exception):
                cancel = getattr(egress_adapter, "cancel_probe_early", None)
                if callable(cancel):
                    cancel()
            await asyncio.gather(
                connect_task, idle_task, model_task, return_exceptions=True
            )
            raise

        # Expose acquired pipeline to the function scope used by finally.
        pipeline = _prep_state["pipeline"]
        sample_rate = _prep_state["sample_rate"]

        state_manager = StreamStateManager(
            live_frame_queue=live_frame_queue,
            idle_video=idle_loop,
            idle_timeout_ms=idle_timeout_ms
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

        # Republish the ingested audio to LiveKit so subscribers hear speech in
        # sync with the lip-synced video. Without this, the viewer sees the
        # avatar's mouth move but hears nothing -- the input audio is consumed
        # by FlashHead and never reaches the room.
        egress_adapter.sample_rate = sample_rate
        egress_adapter.channels = num_channels

        audio_publish_task = asyncio.create_task(
            _audio_publish_loop(engine, egress_adapter, session_id)
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
            next_slice_delay = await asyncio.to_thread(engine.next_live_slice_delay)
            if next_slice_delay is None:
                audio_event = await audio_queue.get()
            elif next_slice_delay <= 0:
                await asyncio.to_thread(engine.process_pending_audio)
                continue
            else:
                try:
                    audio_event = await asyncio.wait_for(
                        audio_queue.get(),
                        timeout=next_slice_delay,
                    )
                except asyncio.TimeoutError:
                    await asyncio.to_thread(engine.process_pending_audio)
                    continue
            if audio_event is None:
                session_ended = True
                break
            if not isinstance(audio_event, CanonicalAudioFrame):
                continue
            if audio_event.is_cancelled:
                _cancel_engine_audio(engine)
                session_state["is_draining"] = False
                continue
            if audio_event.is_end_of_utterance:
                cycles = await _drain_utterance(engine, session_id, session_state)
                ended_seq = audio_event.protocol_sequence_number if audio_event.protocol_sequence_number is not None else audio_event.sequence_number
                await ws_server.send_message(
                    session_id,
                    UtteranceEndedMessage(ended_seq, cycles),
                )
                continue

            normalized = normalize_channels(
                audio_event.pcm,
                audio_event.channels,
                1,
            ).reshape(-1)
            audio_array = resample_audio(
                normalized,
                audio_event.sample_rate,
                engine.sample_rate,
            )
            await asyncio.to_thread(engine.run_chunk, audio_array)

        if session_ended:
            await _drain_utterance(engine, session_id, session_state)
    finally:
        if rtc_metrics_task is not None:
            rtc_metrics_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await rtc_metrics_task

        # Enqueue the audio sentinel BEFORE closing the engine so the
        # _audio_publish_loop can drain any remaining audio and then exit
        # cleanly.  Without this, the task blocks forever on queue.get().
        _engine = _prep_state.get("engine") if _prep_state is not None else engine
        if _engine is not None:
            _engine.audio_queue.put_nowait(None)

        # Wait for the audio publish task to finish (it will exit after
        # consuming the None sentinel).
        if audio_publish_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await audio_publish_task

        if _engine is not None:
            await asyncio.to_thread(_engine.close)
        _pipeline = _prep_state.get("pipeline") if _prep_state is not None else pipeline
        if _pipeline is not None and model_pool is not None:
            model_pool.release(_pipeline)
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
        from streaming.transport.websocket_server import ws_server
        session_state = ws_server.active_sessions.get(session_id)
        idle_watcher = session_state.get("idle_watcher") if session_state else None
        if idle_watcher is not None and idle_watcher is not asyncio.current_task():
            idle_watcher.cancel()
        ws_server.unregister_session(session_id)
            

