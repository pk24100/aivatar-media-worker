import asyncio
import contextlib
import logging
import os

from livekit import rtc

from streaming.flashhead_streaming import FlashHeadStreamingEngine
from streaming.video_publisher import VideoPublisher
from streaming.audio_publisher import AudioPublisher
from streaming.audio_subscriber import AudioSubscriber
from streaming.websocket_server import ws_server
from streaming.sip_handler import SipAudioSubscriber
from streaming.state_manager import StreamStateManager
from streaming.idle_video import IdleVideoLoop

_logger = logging.getLogger("stream_processor")


async def _audio_publish_loop(engine, audio_publisher):
    """
    Consumes audio slices from engine.audio_queue and pushes them to
    LiveKit exactly when the matching video frames are ready.  This
    keeps audio and video lip-synced: the viewer hears the audio at
    the same moment the avatar's mouth moves.
    """
    while True:
        audio_chunk = await asyncio.to_thread(engine.audio_queue.get)
        if audio_chunk is None:
            # Sentinel — stream is over.
            break
        try:
            await audio_publisher.push_audio(audio_chunk)
        except Exception as exc:
            _logger.warning("AudioPublisher push failed: %s", exc)


async def run_streaming_session(
    room_name: str,
    livekit_token: str,
    livekit_url: str,
    pipeline: any,
    source_image: str = None,
    ingestion_method: str = "livekit",
    session_id: str = None,
    ingestion_token: str = None,
    idle_video_url: str = None
):
    """
    Run a streaming session with FlashHead Lite model.
    
    Args:
        room_name: LiveKit room name
        livekit_token: LiveKit token for authentication
        livekit_url: LiveKit server URL
        pipeline: FlashHeadPipeline instance from the model pool
        source_image: URL or path to the avatar/source image
        ingestion_method: Audio ingestion method (livekit, websocket, sip)
        session_id: Unique session identifier
        ingestion_token: Token for websocket ingestion
        idle_video_url: URL for idle video loop
    """
    room = rtc.Room()
    await room.connect(
        livekit_url,
        livekit_token,
        options=rtc.RoomOptions(auto_subscribe=True),
    )

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
            avatar_image_path=source_image
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
            import numpy as np
            import librosa
            import io
            import soundfile as sf
            
            # Register the session with the WS server
            audio_queue = ws_server.register_session(session_id, ingestion_token)
            
            while True:
                # Wait for raw audio bytes from the websocket
                audio_bytes = await audio_queue.get()
                if audio_bytes is None:
                    # End of stream signaled
                    break
                    
                # Robust audio decoding and resampling to match model expectations
                try:
                    # sf.read handles WAV, FLAC, OGG, etc. and gives float32
                    audio_array, sr_in = sf.read(io.BytesIO(audio_bytes))
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=1) # mix down to mono
                    if sr_in != sample_rate:
                        audio_array = librosa.resample(audio_array, orig_sr=sr_in, target_sr=sample_rate)
                except Exception:
                    # Fallback if raw PCM is sent without headers
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

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
            # Default LiveKit ingestion (Client frontend)
            subscriber = AudioSubscriber(room, sample_rate=sample_rate, num_channels=num_channels)
            subscriber.bind()

            ready = await subscriber.wait_until_ready(timeout=chunk_timeout)
            if not ready:
                raise RuntimeError("Timed out waiting for LiveKit audio track.")

            while True:
                audio = await subscriber.read()
                if audio is None:
                    break
                # For native LiveKit ingestion the publisher track already exists
                # in the room, so re-publishing would duplicate. Skip push_audio
                # in this branch -- subscribers can already hear the source.
                await asyncio.to_thread(engine.run_chunk, audio)

        await asyncio.to_thread(engine.flush)
        while not engine.frame_queue.empty():
            await asyncio.sleep(0.05)
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
            ws_server.unregister_session(session_id)
            
        # livekit-rtc 1.x: Room exposes `connection_state` (enum) -- the old
        # `room.connected` boolean was removed. See:
        # https://docs.livekit.io/reference/python/livekit/rtc/room.html
        if room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
            await room.disconnect()
