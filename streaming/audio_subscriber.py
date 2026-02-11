import asyncio
from typing import Optional

import numpy as np
from livekit import rtc


class AudioSubscriber:
    def __init__(
        self,
        room: rtc.Room,
        sample_rate: int = 16000,
        num_channels: int = 1,
        queue_size: int = 200,
    ):
        self.room = room
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.queue = asyncio.Queue(maxsize=queue_size)
        self._ready = asyncio.Event()
        self._audio_task = None
        self._track_sid = None

    def bind(self):
        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if self._audio_task is not None:
                return
            self._track_sid = publication.sid
            audio_stream = rtc.AudioStream(
                track,
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
            )
            self._audio_task = asyncio.create_task(self._consume(audio_stream))
            self._ready.set()

        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = publication.track
                if track and track.kind == rtc.TrackKind.KIND_AUDIO:
                    on_track_subscribed(track, publication, participant)
                    return

    async def wait_until_ready(self, timeout: Optional[float] = None):
        if timeout is None:
            await self._ready.wait()
            return True
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def read(self, timeout: Optional[float] = None):
        if timeout is None:
            return await self.queue.get()
        return await asyncio.wait_for(self.queue.get(), timeout=timeout)

    async def _consume(self, audio_stream: rtc.AudioStream):
        try:
            async for frame_event in audio_stream:
                frame = frame_event.frame
                data = np.frombuffer(frame.data, dtype=np.int16)
                if self.num_channels > 1:
                    data = data.reshape(-1, self.num_channels).mean(axis=1)
                audio = data.astype(np.float32) / 32768.0
                if self.queue.full():
                    _ = self.queue.get_nowait()
                await self.queue.put(audio)
        finally:
            await self.queue.put(None)
