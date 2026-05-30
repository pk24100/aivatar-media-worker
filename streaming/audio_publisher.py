"""LiveKit audio republisher.

The worker receives PCM/float audio chunks over its WebSocket ingestion and
feeds them to the FlashHead engine to drive lip-sync. Without this module those
audio bytes are *consumed* and never reach the viewer in the LiveKit room, so
the viewer sees the avatar's mouth move but hears nothing.

`AudioPublisher.push_audio` tees the same audio chunk into a LiveKit
`AudioSource`/`LocalAudioTrack` so a subscriber gets video AND speech.

Notes on pacing:
- `livekit.rtc.AudioSource.capture_frame` internally rate-limits to wall-clock
  realtime (its internal queue length is bounded by sample_rate). Calling it
  from inside the existing audio-ingest loop is safe; we just convert to
  int16 PCM before pushing.
- The first call lazily publishes the audio track. We don't publish at
  __init__ to keep the surface compatible with VideoPublisher's lazy track
  publish (so both tracks appear when the first piece of media is ready).
"""
import asyncio
import logging
from typing import Optional

import numpy as np
from livekit import rtc

logger = logging.getLogger("AudioPublisher")


class AudioPublisher:
    def __init__(
        self,
        room: rtc.Room,
        sample_rate: int = 16000,
        num_channels: int = 1,
        track_name: str = "aivatar-audio",
    ):
        self.room = room
        self.sample_rate = int(sample_rate)
        self.num_channels = int(num_channels)
        self.track_name = track_name
        self.audio_source: Optional[rtc.AudioSource] = None
        self.track: Optional[rtc.LocalAudioTrack] = None
        self._publish_lock = asyncio.Lock()

    async def _ensure_track(self) -> None:
        if self.track is not None:
            return
        async with self._publish_lock:
            if self.track is not None:
                return
            self.audio_source = rtc.AudioSource(self.sample_rate, self.num_channels)
            self.track = rtc.LocalAudioTrack.create_audio_track(
                self.track_name, self.audio_source
            )
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await self.room.local_participant.publish_track(self.track, options)
            logger.info(
                "AudioPublisher track published (%d Hz, %d ch)",
                self.sample_rate,
                self.num_channels,
            )

    async def push_audio(self, audio_array: np.ndarray) -> None:
        """Push a chunk of audio to the LiveKit room.

        Accepts float32 in [-1.0, 1.0] (the same shape FlashHead expects) or
        int16 PCM. Conversion to int16 is done here so callers don't have to
        duplicate the buffer.
        """
        if audio_array is None or len(audio_array) == 0:
            return

        await self._ensure_track()

        arr = np.asarray(audio_array)
        if arr.ndim > 1:
            # mix down to mono
            arr = arr.mean(axis=1)

        if arr.dtype == np.int16:
            pcm = arr
        else:
            # float32/float64 in [-1, 1]
            pcm = np.clip(arr.astype(np.float32), -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16)

        samples_per_channel = len(pcm) // self.num_channels
        if samples_per_channel == 0:
            return

        frame = rtc.AudioFrame.create(
            self.sample_rate, self.num_channels, samples_per_channel
        )
        # Copy PCM into the preallocated buffer
        buf = np.frombuffer(frame.data, dtype=np.int16)
        np.copyto(buf, pcm[: len(buf)])

        # capture_frame is async; it returns once there's room in the queue --
        # effectively rate-limits us to realtime which is what we want.
        await self.audio_source.capture_frame(frame)

    async def aclose(self) -> None:
        try:
            if self.audio_source is not None:
                await self.audio_source.aclose()
        except Exception as exc:
            logger.debug("AudioPublisher source close failed: %s", exc)
        self.audio_source = None
        self.track = None
