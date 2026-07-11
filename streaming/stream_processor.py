# Orchestrates the full streaming session: LiveKit connection, FlashHead engine, audio/video publishers, and ingestion handlers.
import asyncio
import contextlib
import logging
import os
import time

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
    pipeline: any,
    source_image: str = None,
    ingestion_method: str = "websocket",
    session_id: str = None,
    ingestion_token: str = None,
    idle_video_url: str = None,
    preconnected_room: any = None,
):
    return await _run_streaming_session_native(
        room_name, livekit_token, livekit_url, pipeline, source_image, 
        ingestion_method, session_id, ingestion_token, idle_video_url, preconnected_room)

async def _run_streaming_session_native(
    room_name: str,
    livekit_token: str,
    livekit_url: str,
    pipeline: any,
    source_image: str = None,
    ingestion_method: str = "websocket",
    session_id: str = None,
    ingestion_token: str = None,
    idle_video_url: str = None,
    preconnected_room: any = None,
):
    """
    Run a streaming session with FlashHead Lite model.
    
    Args:
        room_name: LiveKit room name
        livekit_token: LiveKit token for authentication
        livekit_url: LiveKit server URL
        pipeline: FlashHeadPipeline instance from the model pool
        source_image: URL or path to the avatar/source image
        ingestion_method: Audio ingestion method (websocket, sip)
        session_id: Unique session identifier
        ingestion_token: Token for websocket ingestion
        idle_video_url: URL for idle video loop
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

    # --- Network diagnostics (non-blocking, max 2s total) ---
    # UDP to LiveKit's IP range is confirmed blocked (error 101/ENETUNREACH)
    # from logs7.txt ICE DEBUG logs. These tests are kept as quick sanity
    # checks with short timeouts so they don't delay room.connect().
    import socket as _sock
    import struct as _struct
    import urllib.parse as _urlparse

    # 1. STUN sanity check (UDP to Google STUN — tests general UDP egress)
    try:
        _stun_host = "stun.l.google.com"
        _stun_port = 19302
        _stun_req = _struct.pack("!HHI12s", 0x0001, 0, 0x2112A442, b"\x00" * 12)
        _stun_sock = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        _stun_sock.setblocking(False)
        _stun_sock.sendto(_stun_req, (_stun_host, _stun_port))
        _loop = asyncio.get_event_loop()
        _stun_resp, _stun_addr = await asyncio.wait_for(
            _loop.sock_recvfrom(_stun_sock, 1024), timeout=1.0
        )
        _stun_sock.close()
        _logger.info("[NATIVE-DIAG] STUN sanity check (Google): SUCCESS (response from %s, %d bytes)", _stun_addr, len(_stun_resp))
    except Exception as _stun_exc:
        _logger.warning("[NATIVE-DIAG] STUN sanity check (Google): FAILED — %s", _stun_exc)
        try:
            _stun_sock.close()
        except Exception:
            pass

    # 2. Resolve LiveKit server and test TCP connectivity (quick, 1s timeout)
    try:
        _lk_parsed = _urlparse.urlparse(livekit_url)
        _lk_host = _lk_parsed.hostname
        _lk_port = _lk_parsed.port or (443 if _lk_parsed.scheme == "wss" else 80)
        _lk_ips = _sock.getaddrinfo(_lk_host, None, _sock.AF_INET)
        _lk_ip = _lk_ips[0][4][0] if _lk_ips else None
        _logger.info("[NATIVE-DIAG] LiveKit server: host=%s resolved_ip=%s port=%s", _lk_host, _lk_ip, _lk_port)

        # 2a. TCP connectivity test (quick, 1s timeout)
        try:
            _tcp_sock = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
            _tcp_sock.setblocking(False)
            _tcp_t0 = time.monotonic()
            await asyncio.wait_for(
                asyncio.get_event_loop().sock_connect(_tcp_sock, (_lk_host, _lk_port)),
                timeout=1.0,
            )
            _tcp_ms = round((time.monotonic() - _tcp_t0) * 1000, 1)
            _logger.info("[NATIVE-DIAG] TCP to LiveKit %s:%d: SUCCESS in %.1fms", _lk_host, _lk_port, _tcp_ms)
            _tcp_sock.close()
        except Exception as _tcp_exc:
            _logger.warning("[NATIVE-DIAG] TCP to LiveKit %s:%d: FAILED — %s", _lk_host, _lk_port, _tcp_exc)
            try:
                _tcp_sock.close()
            except Exception:
                pass

        # 2b. UDP tests — fire-and-forget (don't block room.connect())
        # UDP to LiveKit's IP range (161.115.180.x) is confirmed blocked
        # via error 101 in ICE DEBUG logs. Running these as background tasks
        # so they don't add 6s delay to the connection.
        if _lk_ip:
            async def _bg_udp_test(ip, port, label):
                try:
                    _udp_sock = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
                    _udp_sock.setblocking(False)
                    _stun_req = _struct.pack("!HHI12s", 0x0001, 0, 0x2112A442, b"\x00" * 12)
                    _udp_sock.sendto(_stun_req, (ip, port))
                    _udp_t0 = time.monotonic()
                    _udp_resp, _udp_addr = await asyncio.wait_for(
                        asyncio.get_event_loop().sock_recvfrom(_udp_sock, 1024), timeout=3.0
                    )
                    _udp_ms = round((time.monotonic() - _udp_t0) * 1000, 1)
                    _logger.info("[NATIVE-DIAG] UDP/STUN to LiveKit %s:%d: SUCCESS in %.1fms", ip, port, _udp_ms)
                except Exception as _udp_exc:
                    _logger.warning("[NATIVE-DIAG] UDP/STUN to LiveKit %s:%d: FAILED — %s", ip, port, _udp_exc)
                finally:
                    try:
                        _udp_sock.close()
                    except Exception:
                        pass
            asyncio.create_task(_bg_udp_test(_lk_ip, 3478, "TURN"))
            asyncio.create_task(_bg_udp_test(_lk_ip, 443, "media"))
    except Exception as _resolve_exc:
        _logger.warning("[NATIVE-DIAG] LiveKit server resolution failed: %s", _resolve_exc)

    # Capture every observable event for root-cause diagnosis
    _diag_events = []
    _diag_t0 = time.monotonic()

    def _diag(ev: str, **kwargs):
        entry = {"event": ev, "t": round(time.monotonic() - _diag_t0, 3), **kwargs}
        _diag_events.append(entry)
        _logger.info("[NATIVE-DIAG] %s", entry)

    def _pub_sid(pub):
        return getattr(pub, "sid", getattr(pub, "track_sid", "unknown"))

    _logger.info("Connecting to LiveKit room=%s url=%s session=%s", room_name, livekit_url, session_id)

    max_retries = 3
    last_error = None
    room = preconnected_room
    if room is not None:
        @room.on("connection_quality_changed")
        def on_quality_changed(quality):
            _diag("quality_changed", quality=str(quality))

        @room.on("participant_connected")
        def on_participant_connected(participant):
            _diag("participant_connected", identity=participant.identity)

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant):
            _diag("participant_disconnected", identity=participant.identity)

        @room.on("local_track_published")
        def on_local_track_published(pub):
            _diag("local_track_published", sid=pub.sid, kind=pub.kind)

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

            @room.on("connected")
            def on_connected():
                _diag("connected")

            @room.on("disconnected")
            def on_disconnected():
                _diag("disconnected", state=str(room.connection_state))

            @room.on("connection_quality_changed")
            def on_quality_changed(quality):
                _diag("quality_changed", quality=str(quality))

            @room.on("reconnecting")
            def on_reconnecting():
                _diag("reconnecting")

            @room.on("reconnected")
            def on_reconnected():
                _diag("reconnected")

            @room.on("track_published")
            def on_track_published(pub, participant):
                _diag("track_published", sid=_pub_sid(pub))

            @room.on("track_unpublished")
            def on_track_unpublished(pub, participant):
                _diag("track_unpublished", sid=_pub_sid(pub))

            @room.on("track_subscribed")
            def on_track_subscribed(track, pub, participant):
                _diag("track_subscribed", sid=_pub_sid(pub), kind=track.kind)

            @room.on("track_subscription_failed")
            def on_track_subscription_failed(track_sid, participant):
                _diag("track_subscription_failed", sid=track_sid)

            @room.on("participant_connected")
            def on_participant_connected(participant):
                _diag("participant_connected", identity=participant.identity)

            @room.on("participant_disconnected")
            def on_participant_disconnected(participant):
                _diag("participant_disconnected", identity=participant.identity)

            @room.on("local_track_published")
            def on_local_track_published(pub):
                _diag("local_track_published", sid=pub.sid, kind=pub.kind)

            _logger.info(
                "[NATIVE-DIAG] room.connect() attempt %d/%d starting | RoomOptions: auto_subscribe=True single_peer_connection=True connect_timeout=45.0 | url=%s",
                attempt + 1, max_retries, livekit_url,
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

    # We will read sample_rate from the engine's model params instead of env
    num_channels = int(os.getenv("AUDIO_CHANNELS", "1"))
    chunk_timeout = float(os.getenv("AUDIO_SUBSCRIBE_TIMEOUT", "15"))
    idle_timeout_ms = int(os.getenv("IDLE_TIMEOUT_MS", "500"))

    engine = None
    publish_task = None
    audio_publish_task = None
    audio_publisher = None
    try:
        # Create FlashHead engine with pipeline and avatar image
        if not source_image:
            raise ValueError("source_image is required for FlashHead streaming")

        engine = FlashHeadStreamingEngine(
            pipeline=pipeline,
            avatar_image_path=source_image,
            auto_prepare_avatar=False,
        )
        prep_t0 = time.monotonic()
        await asyncio.to_thread(engine.prepare_avatar, source_image)
        _logger.info(
            "Prepared avatar for session=%s source=%s in %.1f ms",
            session_id,
            source_image,
            (time.monotonic() - prep_t0) * 1000,
        )

        sample_rate = engine.sample_rate

        # Initialize idle video loop if URL provided
        idle_loop = None
        if idle_video_url:
            idle_loop = IdleVideoLoop(idle_video_url)

        state_manager = StreamStateManager(
            live_frame_queue=engine.frame_queue,
            idle_video=idle_loop,
            idle_timeout_ms=idle_timeout_ms
        )

        fps = engine.tgt_fps
        publisher = VideoPublisher(room, fps=fps)
        publish_task = asyncio.create_task(publisher.publish_from_state_manager(state_manager))

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

