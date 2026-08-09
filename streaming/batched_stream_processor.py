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

from livekit import rtc

from streaming.batched_engine import ACTIVE
from streaming.livekit.video_publisher import VideoPublisher
from streaming.livekit.audio_publisher import AudioPublisher
from streaming.websocket_server import ws_server
from streaming.state_manager import StreamStateManager
from streaming.idle_video import IdleVideoLoop

_logger = logging.getLogger("batched_stream_processor")
RTC_STATS_INTERVAL_SECONDS = float(os.getenv("LIVEKIT_RTC_STATS_INTERVAL_SECONDS", "0"))


def _short_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        return value.split("?", 1)[0].split("#", 1)[0]


def _short_stat_id(value: str) -> str:
    return str(value)[-8:] if value else "unknown"


def _rtc_stat_id(entry) -> str:
    for field in (
        "codec", "inbound_rtp", "outbound_rtp", "remote_inbound_rtp",
        "remote_outbound_rtp", "media_source", "media_playout",
        "peer_connection", "data_channel", "transport", "candidate_pair",
        "local_candidate", "remote_candidate", "certificate", "stream", "track",
    ):
        if entry.HasField(field):
            return getattr(entry, field).rtc.id
    return ""


async def _publisher_rtc_metrics_loop(room, room_name: str, session_id: str) -> None:
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
                session_id, room_name, type(exc).__name__, exc,
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
                        len(outbound), stats.frames_sent,
                        "initial" if frame_delta is None else frame_delta,
                        stats.frames_encoded, stats.frames_per_second,
                        stats.nack_count, stats.pli_count,
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
                        len(transports), stats.ice_state, stats.dtls_state,
                        stats.packets_sent, stats.bytes_sent,
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
                    % (len(candidates), pair.state, pair.current_round_trip_time * 1000,
                       pair.available_outgoing_bitrate)
                )
            elif entry.HasField("local_candidate"):
                candidate = entry.local_candidate.candidate
                candidates.append(
                    "local=%s/%s relay=%s"
                    % (candidate.protocol, candidate.candidate_type, candidate.relay_protocol or "none")
                )
            elif entry.HasField("remote_candidate"):
                candidate = entry.remote_candidate.candidate
                candidates.append(
                    "remote=%s/%s relay=%s"
                    % (candidate.protocol, candidate.candidate_type, candidate.relay_protocol or "none")
                )

        if not outbound and not transports:
            continue

        _logger.info(
            "LIVEKIT_PUBLISHER_RTC session=%s room=%s outbound=[%s] transport=[%s] candidate=[%s]",
            session_id, room_name,
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
            session_id, room_name, type(error).__name__, error,
            exc_info=(type(error), error, error.__traceback__),
        )


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


async def _audio_publish_loop(session, audio_publisher):
    """Consume audio slices from the session's audio_queue and push to LiveKit."""
    consecutive_failures = 0
    max_consecutive_failures = 10
    while True:
        audio_chunk = await asyncio.to_thread(session.audio_queue.get)
        if audio_chunk is None:
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
                    "AudioPublisher exceeded max consecutive failures (%d). Aborting.",
                    max_consecutive_failures,
                )
                break


async def _drain_session(engine, session_id, session_state, keep_alive=False):
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


async def run_batched_streaming_session(
    room_name: str,
    livekit_token: str,
    livekit_url: str,
    engine: any = None,
    source_image: str = None,
    ingestion_method: str = "websocket",
    session_id: str = None,
    ingestion_token: str = None,
    idle_video_url: str = None,
    idle_video_key: str = None,
    preconnected_room: any = None,
    seed: int = 42,
):
    """
    Run a streaming session with batched inference via BatchedStreamingEngine.

    Args:
        room_name: LiveKit room name
        livekit_token: LiveKit token for authentication
        livekit_url: LiveKit server URL
        engine: Shared BatchedStreamingEngine instance
        source_image: URL or path to the avatar/source image
        ingestion_method: Audio ingestion method (websocket, sip)
        session_id: Unique session identifier
        ingestion_token: Token for websocket ingestion
        idle_video_url: URL for idle video loop
        idle_video_key: Immutable idle asset key for the default snapshot cache
        preconnected_room: Already-connected LiveKit room claimed from prewarm pool
        seed: Random seed for avatar generation
    """
    import time

    os.environ.setdefault("LIVEKIT_RTC_DEBUG", "false")
    for _lk_name in ("livekit", "livekit.rtc", "livekit.api"):
        logging.getLogger(_lk_name).setLevel(logging.INFO)

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
            _diag("disconnected", state=str(target_room.connection_state),
                  reason=str(args[0]) if args else None)

        @target_room.on("connection_state_changed")
        def on_connection_state_changed(state):
            _diag("connection_state_changed", state=str(state))

        @target_room.on("connection_quality_changed")
        def on_quality_changed(participant, quality):
            _diag("quality_changed", participant=participant.identity, quality=str(quality))

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
            "Reusing pre-connected LiveKit room=%s session=%s connection_state=%s",
            room_name, session_id, room.connection_state,
        )
    else:
        for attempt in range(max_retries):
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
                "[BATCHED-DIAG] room.connect() attempt %d/%d | url=%s",
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
                    "Connected to LiveKit room=%s (attempt %d, %.1f ms). events=%d",
                    room_name, attempt + 1, conn_ms, len(_diag_events),
                )
                break
            except Exception as exc:
                conn_ms = round((time.monotonic() - conn_t0) * 1000, 1)
                last_error = exc
                _logger.warning(
                    "LiveKit connect failed (attempt %d/%d, %.1f ms): %s. state=%s",
                    attempt + 1, max_retries, conn_ms, exc,
                    room.connection_state if room else "N/A",
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise ConnectionError(
                        f"Failed to connect to LiveKit after {max_retries} attempts. "
                        f"Final state={room.connection_state if room else 'N/A'}"
                    ) from last_error

    from flash_head.inference import get_infer_params

    infer_params = get_infer_params()
    num_channels = int(os.getenv("AUDIO_CHANNELS", "1"))
    idle_timeout_ms = int(os.getenv("IDLE_TIMEOUT_MS", "500"))

    # The session's frame_queue will be created by BatchedSession inside the engine.
    # We need to get it after registration. Use a placeholder that we'll replace.
    live_frame_queue = Queue()  # temporary, replaced after engine registration
    publish_task = None
    audio_publish_task = None
    audio_publisher = None
    rtc_metrics_task = None
    batched_session = None

    try:
        # Set up idle video
        idle_loop = None
        if idle_video_key or idle_video_url:
            from utils.default_idle_video_cache import default_idle_video_cache
            cached_idle_bytes = default_idle_video_cache.get_bytes(idle_video_key)
            idle_loop = await asyncio.to_thread(
                IdleVideoLoop, idle_video_url, video_bytes=cached_idle_bytes,
            )
            idle_loop.normalize(infer_params["width"], infer_params["height"])
        if idle_loop is None or not idle_loop.is_valid():
            idle_loop = await asyncio.to_thread(
                IdleVideoLoop.fallback_from_source_image,
                source_image, infer_params["width"], infer_params["height"],
            )
            if idle_loop.is_valid():
                _logger.info("IDLE_FALLBACK_PUBLISHED session=%s", session_id)

        # Register session with the engine (this creates the BatchedSession
        # and prepares the avatar on the pipeline)
        _logger.info("Registering session %s with BatchedStreamingEngine", session_id)
        prep_t0 = time.monotonic()
        batched_session = await asyncio.to_thread(
            engine.add_session, session_id, source_image, seed, infer_params,
        )
        prep_ms = round((time.monotonic() - prep_t0) * 1000, 1)
        _logger.info(
            "Session %s registered with engine in %.1f ms (avatar prepared)",
            session_id, prep_ms,
        )

        # Now use the session's actual frame_queue for the state manager
        live_frame_queue = batched_session.frame_queue

        state_manager = StreamStateManager(
            live_frame_queue=live_frame_queue,
            idle_video=idle_loop,
            idle_timeout_ms=idle_timeout_ms,
            engine=engine,
        )

        fps = infer_params["tgt_fps"]
        publisher = VideoPublisher(room, fps=fps, session_id=session_id)
        publish_task = asyncio.create_task(publisher.publish_from_state_manager(state_manager))

        if idle_loop and idle_loop.is_valid():
            try:
                await asyncio.wait_for(publisher.first_frame_published.wait(), timeout=15)
                _logger.info("IDLE_TRACK_PUBLISHED session=%s room=%s", session_id, room_name)
            except asyncio.TimeoutError:
                raise RuntimeError("Timed out publishing the idle video track")

        sample_rate = infer_params["sample_rate"]
        audio_publisher = AudioPublisher(
            room=room,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        if ingestion_method in ("websocket", "sip"):
            audio_publish_task = asyncio.create_task(
                _audio_publish_loop(batched_session, audio_publisher)
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
            audio_queue = ws_server.register_session(session_id, ingestion_token)
            session_state = ws_server.active_sessions[session_id]
            session_ended = False

            while True:
                audio_event = await audio_queue.get()
                if audio_event is None:
                    session_ended = True
                    break

                if isinstance(audio_event, dict):
                    control_type = audio_event.get("type")
                    if control_type == "end_utterance":
                        await _drain_session(engine, session_id, session_state, keep_alive=True)
                    elif control_type == "end_session":
                        session_ended = True
                        break
                    continue

                audio_array = _decode_raw_pcm(audio_event, num_channels)
                await asyncio.to_thread(engine.feed_audio, session_id, audio_array)

        elif ingestion_method == "sip":
            from streaming.sip_handler import SipAudioSubscriber

            audio_queue = asyncio.Queue()
            sip_subscriber = SipAudioSubscriber(
                room=room,
                audio_queue=audio_queue,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )
            sip_subscriber.bind()

            ready = await sip_subscriber.wait_until_ready(timeout=float(os.getenv("AUDIO_SUBSCRIBE_TIMEOUT", "15")))
            if not ready:
                raise RuntimeError("Timed out waiting for SIP audio track.")

            while True:
                audio_array = await audio_queue.get()
                if audio_array is None:
                    break
                await asyncio.to_thread(engine.feed_audio, session_id, audio_array)

        else:
            raise ValueError(f"Unsupported ingestion_method: {ingestion_method}")

        if ingestion_method == "websocket":
            if session_ended:
                await _drain_session(engine, session_id, session_state)
        else:
            await _drain_session(engine, session_id, {"is_draining": False})

    finally:
        if rtc_metrics_task is not None:
            rtc_metrics_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await rtc_metrics_task

        # Signal audio publish loop to stop
        if batched_session is not None:
            batched_session.audio_queue.put_nowait(None)

        if audio_publish_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await audio_publish_task

        # Remove session from engine if still present
        if engine is not None:
            engine.remove_session(session_id)

        if publish_task is not None:
            publish_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await publish_task
        if audio_publisher is not None:
            with contextlib.suppress(Exception):
                await audio_publisher.aclose()
        if ingestion_method == "websocket":
            session_state = ws_server.active_sessions.get(session_id)
            idle_watcher = session_state.get("idle_watcher") if session_state else None
            if idle_watcher is not None and idle_watcher is not asyncio.current_task():
                idle_watcher.cancel()
            ws_server.unregister_session(session_id)

        if room is not None and room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
            await room.disconnect()
