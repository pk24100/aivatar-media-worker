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

    subscriber = AudioSubscriber(room, sample_rate=sample_rate, num_channels=num_channels)
    subscriber.bind()

    ready = await subscriber.wait_until_ready(timeout=chunk_timeout)
    if not ready:
        await room.disconnect()
        raise RuntimeError("Timed out waiting for LiveKit audio track.")

    engine = DittoStreamingEngine(
        model_root=model_root,
        source_path=source_image,
        chunksize=chunksize,
    )
    chunker = AudioChunker(sample_rate=sample_rate, chunksize=chunksize)

    fps = int(os.getenv("LIVEKIT_PUBLISH_FPS", "25"))
    publisher = VideoPublisher(room, fps=fps)
    publish_task = asyncio.create_task(publisher.publish_from_queue(engine.frame_queue))

    try:
        while True:
            audio = await subscriber.read()
            if audio is None:
                break
            for chunk in chunker.add_samples(audio):
                await asyncio.to_thread(engine.run_chunk, chunk)

        for chunk in chunker.flush():
            await asyncio.to_thread(engine.run_chunk, chunk)

        await asyncio.to_thread(engine.close)
        await publish_task
    finally:
        if room.connected:
            await room.disconnect()
