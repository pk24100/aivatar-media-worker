"""LiveKit audio input adapter for remote participant tracks.

Parked for MVP - WebSocket PCM is the sole ingestion path. This adapter is
retained for future LiveKit track-based ingestion scenarios.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from streaming.core.audio_bus import CanonicalAudioBus
from streaming.core.session import MediaInputAdapter

logger = logging.getLogger("LiveKitInputAdapter")


class LiveKitInputAdapter(MediaInputAdapter):
    def __init__(self, bus: CanonicalAudioBus, *, room=None, participant_filter=None):
        self.bus = bus
        self.room = room
        self.participant_filter = participant_filter
        self._ready = asyncio.Event()
        self._audio_task: asyncio.Task | None = None
        self._connected = False
        self._track_sid: str | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self, session_config: dict) -> None:
        self.room = session_config.get("room", self.room)
        if self.room is None:
            raise ValueError("room is required for LiveKit input")
        bus_config = {
            key: value
            for key, value in session_config.items()
            if key in {"audio_encoding", "sample_rate", "channels", "avatar_id", "session_id", "metadata"}
        }
        await self.bus.connect(bus_config)
        self._connected = True
        self.bind()

    async def disconnect(self) -> None:
        self._connected = False
        if self._audio_task is not None:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None

    async def start_utterance(self) -> None:
        await self.bus.start_utterance()

    async def push_audio(self, pcm: np.ndarray, sample_rate: int, channels: int) -> None:
        await self.bus.push_audio(pcm, sample_rate, channels)

    async def end_utterance(self) -> None:
        await self.bus.end_utterance()

    async def cancel_utterance(self) -> None:
        await self.bus.cancel_utterance()

    async def on_interrupted(self) -> None:
        await self.cancel_utterance()

    def bind(self) -> None:
        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if getattr(track, "kind", None) != self._audio_kind():
                return
            if self.participant_filter and not self.participant_filter(participant):
                return
            if self._audio_task is not None:
                return
            self._track_sid = getattr(publication, "sid", None)
            from livekit import rtc

            audio_stream = rtc.AudioStream(track)
            self._audio_task = asyncio.create_task(self._consume(audio_stream))
            self._ready.set()

        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = publication.track
                if track and getattr(track, "kind", None) == self._audio_kind():
                    on_track_subscribed(track, publication, participant)
                    return

    async def wait_until_ready(self, timeout: float | None = None) -> bool:
        try:
            if timeout is None:
                await self._ready.wait()
            else:
                await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _consume(self, audio_stream) -> None:
        try:
            async for frame_event in audio_stream:
                frame = frame_event.frame
                pcm = np.frombuffer(frame.data, dtype=np.int16)
                await self.push_audio(
                    pcm,
                    int(getattr(frame, "sample_rate", self.bus.canonical_sample_rate)),
                    int(getattr(frame, "num_channels", 1)),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("LiveKit audio stream failed")
        finally:
            if self._connected:
                await self.bus.end_utterance()

    @staticmethod
    def _audio_kind():
        from livekit import rtc

        return rtc.TrackKind.KIND_AUDIO
