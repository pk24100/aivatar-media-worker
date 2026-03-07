import asyncio
import os
from typing import Optional, Tuple

from livekit import rtc

from streaming.audio_chunker import AudioChunker
from streaming.audio_subscriber import AudioSubscriber
from streaming.ditto_streaming import DittoStreamingEngine
from streaming.video_publisher import VideoPublisher


def _parse_chunksize(value: Optional[str]) -> Tuple[int, int, int]:
    if not value:
        return (3, 5, 2)
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("DITTO_CHUNKSIZE must have 3 comma-separated ints, e.g. 3,5,2")
    return tuple(parts)


async def run_streaming_session(
    room_name: str,
    livekit_token: str,
    livekit_url: str,
    model_root: str,
    source_image: str,
    model_instance: Optional[object] = None,
    ingestion_method: str = "livekit",
    session_id: str = "",
    ingestion_token: str = "",
):
    room = rtc.Room()
    await room.connect(
        livekit_url,
        livekit_token,
        options=rtc.RoomOptions(auto_subscribe=True),
    )

    sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    num_channels = int(os.getenv("AUDIO_CHANNELS", "1"))
    chunk_timeout = float(os.getenv("AUDIO_SUBSCRIBE_TIMEOUT", "15"))
    chunksize = _parse_chunksize(os.getenv("DITTO_CHUNKSIZE"))

    engine = DittoStreamingEngine(
        model_root=model_root,
        source_path=source_image,
        chunksize=chunksize,
        model_instance=model_instance,
    )
    chunker = AudioChunker(sample_rate=sample_rate, chunksize=chunksize)

    fps = int(os.getenv("LIVEKIT_PUBLISH_FPS", "25"))
    publisher = VideoPublisher(room, fps=fps)
    publish_task = asyncio.create_task(publisher.publish_from_queue(engine.frame_queue))

    try:
        if ingestion_method == "websocket":
            from streaming.websocket_server import ws_server
            import numpy as np
            
            # Register the session with the WS server
            audio_queue = ws_server.register_session(session_id, ingestion_token)
            
            while True:
                # Wait for raw audio bytes from the websocket
                audio_bytes = await audio_queue.get()
                if audio_bytes is None:
                    # End of stream signaled
                    break
                    
                # Convert bytes to numpy array (assuming raw 16kHz PCM int16 for now)
                # In a real app, you'd likely want to handle different encodings (e.g. mp3/wav)
                # using librosa or soundfile in memory, but assuming PCM for raw performance
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Make it 2D (time, channels)
                if len(audio_array.shape) == 1:
                    audio_array = audio_array[:, np.newaxis]
                    
                for chunk in chunker.add_samples(audio_array):
                    await asyncio.to_thread(engine.run_chunk, chunk)
                    
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
                for chunk in chunker.add_samples(audio_array):
                    await asyncio.to_thread(engine.run_chunk, chunk)
                    
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
                for chunk in chunker.add_samples(audio):
                    await asyncio.to_thread(engine.run_chunk, chunk)

        # Flush remaining chunks
        for chunk in chunker.flush():
            await asyncio.to_thread(engine.run_chunk, chunk)

        await asyncio.to_thread(engine.close)
        await publish_task
    finally:
        if ingestion_method == "websocket":
            from streaming.websocket_server import ws_server
            ws_server.unregister_session(session_id)
            
        if room.connected:
            await room.disconnect()
