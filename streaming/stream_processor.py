# Orchestrates the full streaming session: LiveKit connection, FlashHead engine, audio/video publishers, and ingestion handlers.
import asyncio
import contextlib
import logging
import os
import time
from queue import Queue
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from livekit import rtc

from streaming.flashhead_streaming import FlashHeadStreamingEngine
from streaming.livekit.video_publisher import VideoPublisher
from streaming.livekit.audio_publisher import AudioPublisher
from streaming.websocket_server import ws_server
from streaming.sip_handler import SipAudioSubscriber
from streaming.state_manager import StreamStateManager
from streaming.idle_video import IdleVideoLoop

_logger = logging.getLogger("stream_processor")
RTC_STATS_INTERVAL_SECONDS = float(os.getenv("LIVEKIT_RTC_STATS_INTERVAL_SECONDS", "0"))


def _short_url(value: str) -> str:
    """Remove query strings and fragments, which can contain long credentials."""
    try:
        parsed = urlsplit(value)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        return value.split("?", 1)[0].split("#", 1)[0]


def _short_stat_id(value: str) -> str:
    return str(value)[-8:] if value else "unknown"


def _rtc_stat_id(entry) -> str:
    for field in (
        "codec",
        "inbound_rtp",
        "outbound_rtp",
        "remote_inbound_rtp",
        "remote_outbound_rtp",
        "media_source",
        "media_playout",
        "peer_connection",
        "data_channel",
        "transport",
        "candidate_pair",
        "local_candidate",
        "remote_candidate",
        "certificate",
        "stream",
        "track",
    ):
        if entry.HasField(field):
            return getattr(entry, field).rtc.id
    return ""


async def _publisher_rtc_metrics_loop(room, room_name: str, session_id: str) -> None:
    """Log publisher transport health without emitting a record for every frame."""
    previous_frames = {}
    previous_bytes = {}

    while True:
        await asyncio.sleep(RTC_STATS_INTERVAL_SECONDS)
        if room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
            return

        try:
            publisher_stats = (await room.get_rtc_stats()).publisher_stats
        except Exception as exc:
            _logger.warning(
                "LIVEKIT_PUBLISHER_RTC_FAILED session=%s room=%s errorType=%s error=%s",
                session_id,
                room_name,
                type(exc).__name__,
                exc,
            )
            continue

        entries_by_id = {
            _rtc_stat_id(entry): entry
            for entry in publisher_stats
            if _rtc_stat_id(entry)
        }
        outbound = []
        transports = []
        candidates = []

        for entry in publisher_stats:
            if entry.HasField("outbound_rtp"):
                payload = entry.outbound_rtp
                stats = payload.outbound
                stats_key = payload.rtc.id or f"outbound-{len(outbound)}"
                previous = previous_frames.get(stats_key)
                previous_frames[stats_key] = stats.frames_sent
                frame_delta = None if previous is None else stats.frames_sent - previous
                outbound.append(
                    "stream=%d framesSent=%d framesDelta=%s framesEncoded=%d fps=%.1f nack=%d pli=%d"
                    % (
                        len(outbound),
                        stats.frames_sent,
                        "initial" if frame_delta is None else frame_delta,
                        stats.frames_encoded,
                        stats.frames_per_second,
                        stats.nack_count,
                        stats.pli_count,
                    )
                )
            elif entry.HasField("transport"):
                payload = entry.transport
                stats = payload.transport
                stats_key = payload.rtc.id or f"transport-{len(transports)}"
                previous = previous_bytes.get(stats_key)
                previous_bytes[stats_key] = stats.bytes_sent
                byte_delta = None if previous is None else stats.bytes_sent - previous
                transports.append(
                    "transport=%d ice=%s dtls=%s packetsSent=%d bytesSent=%d bytesDelta=%s selectedPair=%s"
                    % (
                        len(transports),
                        stats.ice_state,
                        stats.dtls_state,
                        stats.packets_sent,
                        stats.bytes_sent,
                        "initial" if byte_delta is None else byte_delta,
                        _short_stat_id(stats.selected_candidate_pair_id),
                    )
                )
                pair_entry = entries_by_id.get(stats.selected_candidate_pair_id)
                if pair_entry is None or not pair_entry.HasField("candidate_pair"):
                    continue
                pair = pair_entry.candidate_pair.candidate_pair
                local_entry = entries_by_id.get(pair.local_candidate_id)
                remote_entry = entries_by_id.get(pair.remote_candidate_id)
                local = local_entry.local_candidate.candidate if local_entry and local_entry.HasField("local_candidate") else None
                remote = remote_entry.remote_candidate.candidate if remote_entry and remote_entry.HasField("remote_candidate") else None
                candidates.append(
                    "selectedPair=%s state=%s local=%s/%s remote=%s/%s rttMs=%.1f"
                    % (
                        _short_stat_id(stats.selected_candidate_pair_id),
                        pair.state,
                        local.protocol if local else "unknown",
                        local.candidate_type if local else "unknown",
                        remote.protocol if remote else "unknown",
                        remote.candidate_type if remote else "unknown",
                        pair.current_round_trip_time * 1000,
                    )
                )
            elif entry.HasField("candidate_pair"):
                pair = entry.candidate_pair.candidate_pair
                candidates.append(
                    "pair=%d state=%s rttMs=%.1f outgoingBitrate=%d"
                    % (
                        len(candidates),
                        pair.state,
                        pair.current_round_trip_time * 1000,
                        pair.available_outgoing_bitrate,
                    )
                )
            elif entry.HasField("local_candidate"):
                candidate = entry.local_candidate.candidate
                candidates.append(
                    "local=%s/%s relay=%s"
                    % (
                        candidate.protocol,
                        candidate.candidate_type,
                        candidate.relay_protocol or "none",
                    )
                )
            elif entry.HasField("remote_candidate"):
                candidate = entry.remote_candidate.candidate
                candidates.append(
                    "remote=%s/%s relay=%s"
                    % (
                        candidate.protocol,
                        candidate.candidate_type,
                        candidate.relay_protocol or "none",
                    )
                )

        if not outbound and not transports:
            continue

        _logger.info(
            "LIVEKIT_PUBLISHER_RTC session=%s room=%s outbound=[%s] transport=[%s] candidate=[%s]",
            session_id,
            room_name,
            "; ".join(outbound) or "none",
            "; ".join(transports) or "none",
            "; ".join(candidates) or "none",
        )


def _log_rtc_metrics_task_failure(task, session_id: str, room_name: str) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        _logger.error(
            "LIVEKIT_PUBLISHER_RTC_TASK_FAILED session=%s room=%s errorType=%s error=%s",
            session_id,
            room_name,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )


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


async def _drain_utterance(engine, session_id: str, session_state: dict) -> None:
    """Finish queued speech without ending the persistent media session."""
    had_pending_media = bool(engine.pending_audio) or not engine.frame_queue.empty()
    session_state["is_draining"] = True
    drain_started_at = time.monotonic()
    drain_cycles = 0
    _logger.info("UTTERANCE_DRAIN_STARTED session=%s", session_id)

    try:
        while True:
            has_pending_audio = await asyncio.to_thread(engine.flush)
            drain_cycles += 1
            if not has_pending_audio:
                break
            await asyncio.sleep(engine.next_slice_delay())

        while not engine.frame_queue.empty():
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


async def _audio_publish_loop(engine, audio_publisher):
    """
    Consumes audio slices from engine.audio_queue and pushes them to
    LiveKit exactly when the matching video frames are ready.  This
    keeps audio and video lip-synced: the viewer hears the audio at
    the same moment the avatar's mouth moves.
    """
    consecutive_failures = 0
    max_consecutive_failures = 10
    while True:
        audio_chunk = await asyncio.to_thread(engine.audio_queue.get)
        if audio_chunk is None:
            # Sentinel — stream is over.
            break
        try:
            await audio_publisher.push_audio(audio_chunk)
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
    livekit_token: str,
    livekit_url: str,
    pipeline: any = None,
    model_pool: any = None,
    model_ready_waiter: any = None,
    source_image: str = None,
    ingestion_method: str = "websocket",
    session_id: str = None,
    ingestion_token: str = None,
    idle_video_url: str = None,
    idle_video_key: str = None,
    preconnected_room: any = None,
):
    return await _run_streaming_session_native(
        room_name, livekit_token, livekit_url, pipeline, model_pool, model_ready_waiter, source_image,
        ingestion_method, session_id, ingestion_token, idle_video_url, idle_video_key, preconnected_room)

async def _run_streaming_session_native(
    room_name: str,
    livekit_token: str,
    livekit_url: str,
    pipeline: any = None,
    model_pool: any = None,
    model_ready_waiter: any = None,
    source_image: str = None,
    ingestion_method: str = "websocket",
    session_id: str = None,
    ingestion_token: str = None,
    idle_video_url: str = None,
    idle_video_key: str = None,
    preconnected_room: any = None,
):
    """
    Run a streaming session with FlashHead Lite model.
    
    Args:
        room_name: LiveKit room name
        livekit_token: LiveKit token for authentication
        livekit_url: LiveKit server URL
        pipeline: Optional pre-acquired FlashHeadPipeline instance
        model_pool: Pool used after idle publishing when no pipeline is supplied
        model_ready_waiter: Awaitable that gates GPU model initialization
        source_image: URL or path to the avatar/source image
        ingestion_method: Audio ingestion method (websocket, sip)
        session_id: Unique session identifier
        ingestion_token: Token for websocket ingestion
        idle_video_url: URL for idle video loop
        idle_video_key: Immutable idle asset key for the default snapshot cache
        preconnected_room: Already-connected LiveKit room claimed from prewarm pool
    """
    import time

    # Enable Rust FFI debug logs (ICE, DTLS, etc.) — checked at callback time
    os.environ.setdefault("LIVEKIT_RTC_DEBUG", "false")
    # Rust FFI log level — set to DEBUG for ICE/DTLS diagnosis, INFO for production.
    # DEBUG confirmed in logs7.txt that UDP to LiveKit's IP range (161.115.180.x)
    # is blocked (error 101/ENETUNREACH) and connection succeeds via TCP port 7881.
    for _lk_name in ("livekit", "livekit.rtc", "livekit.api"):
        logging.getLogger(_lk_name).setLevel(logging.INFO)

    # --- Threading context diagnostic ---
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

    _logger.info("Connecting to LiveKit room=%s url=%s session=%s", room_name, _short_url(livekit_url), session_id)

    max_retries = 3
    last_error = None
    room = preconnected_room

    def _bind_room_diagnostics(target_room):
        @target_room.on("connected")
        def on_connected(*_args):
            _diag("connected", state=str(target_room.connection_state))

        @target_room.on("disconnected")
        def on_disconnected(*args):
            _diag(
                "disconnected",
                state=str(target_room.connection_state),
                reason=str(args[0]) if args else None,
            )

        @target_room.on("connection_state_changed")
        def on_connection_state_changed(state):
            _diag("connection_state_changed", state=str(state))

        @target_room.on("connection_quality_changed")
        def on_quality_changed(participant, quality):
            _diag(
                "quality_changed",
                participant=participant.identity,
                quality=str(quality),
            )

        @target_room.on("reconnecting")
        def on_reconnecting(*_args):
            _diag("reconnecting")

        @target_room.on("reconnected")
        def on_reconnected(*_args):
            _diag("reconnected", state=str(target_room.connection_state))

        @target_room.on("track_published")
        def on_track_published(pub, participant):
            _diag("track_published", sid=_pub_sid(pub), participant=participant.identity)

        @target_room.on("track_unpublished")
        def on_track_unpublished(pub, participant):
            _diag("track_unpublished", sid=_pub_sid(pub), participant=participant.identity)

        @target_room.on("track_subscribed")
        def on_track_subscribed(track, pub, participant):
            _diag("track_subscribed", sid=_pub_sid(pub), kind=track.kind, participant=participant.identity)

        @target_room.on("track_subscription_failed")
        def on_track_subscription_failed(track_sid, participant):
            _diag("track_subscription_failed", sid=track_sid, participant=participant.identity)

        @target_room.on("participant_connected")
        def on_participant_connected(participant):
            _diag("participant_connected", identity=participant.identity)

        @target_room.on("participant_disconnected")
        def on_participant_disconnected(participant):
            _diag("participant_disconnected", identity=participant.identity)

        @target_room.on("local_track_published")
        def on_local_track_published(pub, *_args):
            _diag("local_track_published", sid=_pub_sid(pub), kind=pub.kind)

    if room is not None:
        _bind_room_diagnostics(room)

        _logger.info(
            "Reusing pre-connected LiveKit room=%s session=%s connection_state=%s local_participant_identity=%s",
            room_name,
            session_id,
            room.connection_state,
            room.local_participant.identity if room.local_participant else "N/A",
        )
    else:
        for attempt in range(max_retries):
            # Create a FRESH rtc.Room() for each retry attempt.
            # Reusing the same Room object after a failed connect corrupts the
            # Rust FFI server's internal state, causing "timed out waiting for
            # ReadyForRoomEventRequest" panics on subsequent sessions.
            if room is not None:
                try:
                    if room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
                        await room.disconnect()
                except Exception:
                    pass
                room = None

            room = rtc.Room()
            _bind_room_diagnostics(room)

            _logger.info(
                "[NATIVE-DIAG] room.connect() attempt %d/%d starting | RoomOptions: auto_subscribe=True single_peer_connection=True connect_timeout=45.0 | url=%s",
                attempt + 1, max_retries, _short_url(livekit_url),
            )
            conn_t0 = time.monotonic()
            try:
                await room.connect(
                    livekit_url,
                    livekit_token,
                    options=rtc.RoomOptions(
                        auto_subscribe=True,
                        single_peer_connection=True,
                        connect_timeout=45.0,
                    ),
                )
                conn_ms = round((time.monotonic() - conn_t0) * 1000, 1)
                _logger.info(
                    "Successfully connected to LiveKit room=%s (attempt %d, %.1f ms). events=%d",
                    room_name, attempt + 1, conn_ms, len(_diag_events)
                )
                _logger.info(
                    "[NATIVE-DIAG] post-connect state: connection_state=%s local_participant_identity=%s",
                    room.connection_state,
                    room.local_participant.identity if room.local_participant else "N/A",
                )
                break
            except Exception as exc:
                conn_ms = round((time.monotonic() - conn_t0) * 1000, 1)
                last_error = exc
                _logger.warning(
                    "LiveKit connect failed (attempt %d/%d, %.1f ms): %s. "
                    "connection_state=%s diag_events=%s",
                    attempt + 1, max_retries, conn_ms, exc,
                    room.connection_state if room else "N/A",
                    _diag_events,
                    exc_info=True,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # exponential backoff: 1s, 2s
                else:
                    raise ConnectionError(
                        f"Failed to connect to LiveKit after {max_retries} attempts. "
                        f"Final state={room.connection_state if room else 'N/A'} "
                        f"Events={_diag_events}"
                    ) from last_error

    # We can publish a target-sized idle track before touching a GPU pipeline.
    # This lets viewers see the same LiveKit track while model warmup and avatar
    # preparation complete in the background.
    from flash_head.inference import get_infer_params

    infer_params = get_infer_params()
    num_channels = int(os.getenv("AUDIO_CHANNELS", "1"))
    chunk_timeout = float(os.getenv("AUDIO_SUBSCRIBE_TIMEOUT", "15"))
    idle_timeout_ms = int(os.getenv("IDLE_TIMEOUT_MS", "500"))
    live_frame_queue = Queue()
    engine = None
    publish_task = None
    audio_publish_task = None
    audio_publisher = None
    rtc_metrics_task = None
    try:
        # Load a default snapshot-cached clip when available. Custom clips are
        # intentionally fetched per session and decoded outside the aiohttp loop.
        idle_loop = None
        if idle_video_key or idle_video_url:
            from utils.default_idle_video_cache import default_idle_video_cache

            cached_idle_bytes = default_idle_video_cache.get_bytes(idle_video_key)
            idle_loop = await asyncio.to_thread(
                IdleVideoLoop,
                idle_video_url,
                video_bytes=cached_idle_bytes,
            )
            idle_loop.normalize(infer_params["width"], infer_params["height"])
        if idle_loop is None or not idle_loop.is_valid():
            idle_loop = await asyncio.to_thread(
                IdleVideoLoop.fallback_from_source_image,
                source_image,
                infer_params["width"],
                infer_params["height"],
            )
            if idle_loop.is_valid():
                _logger.info("IDLE_FALLBACK_PUBLISHED session=%s", session_id)

        state_manager = StreamStateManager(
            live_frame_queue=live_frame_queue,
            idle_video=idle_loop,
            idle_timeout_ms=idle_timeout_ms
        )

        fps = infer_params["tgt_fps"]
        publisher = VideoPublisher(room, fps=fps)
        publish_task = asyncio.create_task(publisher.publish_from_state_manager(state_manager))

        if idle_loop and idle_loop.is_valid():
            try:
                await asyncio.wait_for(publisher.first_frame_published.wait(), timeout=15)
                _logger.info("IDLE_TRACK_PUBLISHED session=%s room=%s", session_id, room_name)
            except asyncio.TimeoutError:
                raise RuntimeError("Timed out publishing the idle video track")

        if model_ready_waiter is not None:
            await model_ready_waiter()
        if pipeline is None:
            if model_pool is None:
                raise RuntimeError("pipeline or model_pool is required for streaming")
            pipeline = await model_pool.acquire()

        if not source_image:
            raise ValueError("source_image is required for FlashHead streaming")
        engine = FlashHeadStreamingEngine(
            pipeline=pipeline,
            avatar_image_path=source_image,
            auto_prepare_avatar=False,
            frame_queue=live_frame_queue,
        )
        sample_rate = engine.sample_rate

        prep_t0 = time.monotonic()
        await asyncio.to_thread(engine.prepare_avatar, source_image)
        _logger.info(
            "Prepared avatar for session=%s source=%s in %.1f ms",
            session_id,
            source_image,
            (time.monotonic() - prep_t0) * 1000,
        )

        # Republish the ingested audio to LiveKit so subscribers hear speech in
        # sync with the lip-synced video. Without this, the viewer sees the
        # avatar's mouth move but hears nothing -- the input audio is consumed
        # by FlashHead and never reaches the room.
        audio_publisher = AudioPublisher(
            room=room,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        # For websocket and SIP ingestion the audio must be re-published to
        # the LiveKit room.  We pull from engine.audio_queue (which is fed
        # in lock-step with video frames) so A-V stays synchronised.
        if ingestion_method in ("websocket", "sip"):
            audio_publish_task = asyncio.create_task(
                _audio_publish_loop(engine, audio_publisher)
            )

        if RTC_STATS_INTERVAL_SECONDS > 0:
            rtc_metrics_task = asyncio.create_task(
                _publisher_rtc_metrics_loop(room, room_name, session_id),
                name=f"rtc-metrics-{session_id}",
            )
            rtc_metrics_task.add_done_callback(
                lambda task: _log_rtc_metrics_task_failure(task, session_id, room_name)
            )

        if ingestion_method == "websocket":
            from streaming.websocket_server import ws_server
            
            # Register the session with the WS server
            audio_queue = ws_server.register_session(session_id, ingestion_token)
            session_state = ws_server.active_sessions[session_id]
            session_ended = False
            
            while True:
                # Binary messages are PCM; dictionaries are handler control events.
                next_slice_delay = await asyncio.to_thread(engine.next_live_slice_delay)
                if next_slice_delay is None:
                    audio_event = await audio_queue.get()
                elif next_slice_delay <= 0:
                    # Gnani can pause between packets while a complete slice is already
                    # buffered. Generate at the real-time deadline instead of repeating
                    # the previous video frame until another packet arrives.
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

                if isinstance(audio_event, dict):
                    control_type = audio_event.get("type")
                    if control_type == "end_utterance":
                        await _drain_utterance(engine, session_id, session_state)
                    elif control_type == "end_session":
                        session_ended = True
                        break
                    continue
                      
                # WebSocket payloads are raw PCM, not self-contained audio files.
                # Trying container decoders per 100 ms payload produces libav errors
                # and adds avoidable work before the known PCM fallback.
                audio_array = _decode_raw_pcm(audio_event, num_channels)

                # Audio is NOT pushed here any more — it is enqueued inside
                # engine.run_chunk → _process_available_audio and consumed
                # by _audio_publish_loop, keeping A-V synchronised.
                await asyncio.to_thread(engine.run_chunk, audio_array)
                    
        elif ingestion_method == "sip":
            from streaming.sip_handler import SipAudioSubscriber
            
            # SIP ingestion is similar to LiveKit, but the audio comes from a specific SIP participant track
            audio_queue = asyncio.Queue()
            sip_subscriber = SipAudioSubscriber(
                room=room, 
                audio_queue=audio_queue,
                sample_rate=sample_rate, 
                num_channels=num_channels
            )
            sip_subscriber.bind()
            
            ready = await sip_subscriber.wait_until_ready(timeout=chunk_timeout)
            if not ready:
                raise RuntimeError("Timed out waiting for SIP audio track.")
                
            while True:
                audio_array = await audio_queue.get()
                if audio_array is None:
                    break
                # Audio is NOT pushed here any more — same as websocket branch.
                await asyncio.to_thread(engine.run_chunk, audio_array)
                    
        else:
            raise ValueError(f"Unsupported ingestion_method: {ingestion_method}")

        if ingestion_method == "websocket":
            # An explicit session end or remote disconnect closes this pipeline.
            if session_ended:
                await _drain_utterance(engine, session_id, session_state)
        else:
            await _drain_utterance(engine, session_id, {"is_draining": False})
    finally:
        if rtc_metrics_task is not None:
            rtc_metrics_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await rtc_metrics_task

        # Enqueue the audio sentinel BEFORE closing the engine so the
        # _audio_publish_loop can drain any remaining audio and then exit
        # cleanly.  Without this, the task blocks forever on queue.get().
        if engine is not None:
            engine.audio_queue.put_nowait(None)

        # Wait for the audio publish task to finish (it will exit after
        # consuming the None sentinel).
        if audio_publish_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await audio_publish_task

        if engine is not None:
            await asyncio.to_thread(engine.close)
        if pipeline is not None and model_pool is not None:
            model_pool.release(pipeline)
        if publish_task is not None:
            publish_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await publish_task
        if audio_publisher is not None:
            with contextlib.suppress(Exception):
                await audio_publisher.aclose()
        if ingestion_method == "websocket":
            from streaming.websocket_server import ws_server
            session_state = ws_server.active_sessions.get(session_id)
            idle_watcher = session_state.get("idle_watcher") if session_state else None
            if idle_watcher is not None and idle_watcher is not asyncio.current_task():
                idle_watcher.cancel()
            ws_server.unregister_session(session_id)
            
        # livekit-rtc 1.x: Room exposes `connection_state` (enum) -- the old
        # `room.connected` boolean was removed. See:
        # https://docs.livekit.io/reference/python/livekit/rtc/room.html
        if room is not None and room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
            await room.disconnect()

