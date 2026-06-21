"""
Audio publisher using aiortc for Modal deployments.

Drop-in replacement for AudioPublisher that uses aiortc's AudioStreamTrack
instead of livekit-rtc's AudioSource/LocalAudioTrack. Same public API.
"""

import asyncio
import fractions
import logging
from typing import Optional

import numpy as np
from av import AudioFrame
from aiortc import AudioStreamTrack

_logger = logging.getLogger("AiortcAudioPublisher")

# Time base for 48 kHz audio clock (aiortc default)
_AUDIO_TIME_BASE = fractions.Fraction(1, 48000)


class AiortcAudioStreamTrack(AudioStreamTrack):
    """
    Custom audio track that serves PCM frames via recv().

    Audio chunks are pushed from the engine thread via push_audio().
    The aiortc transport calls recv() to pull audio for encoding/sending.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 16000, num_channels: int = 1):
        super().__init__()
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._pts = 0
        # aiortc expects 960 samples per frame at 48kHz (20ms Opus frames).
        # We accumulate incoming audio and repackage to match.
        self._buffer = np.array([], dtype=np.int16)
        self._output_rate = 48000
        self._frame_size = 960  # samples per channel per frame (20ms at 48kHz)

    async def push_audio(self, audio_array: np.ndarray) -> None:
        """Push a chunk of audio (float32 [-1,1] or int16) into the track."""
        if audio_array is None or len(audio_array) == 0:
            return

        arr = np.asarray(audio_array)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        if arr.dtype == np.int16:
            pcm = arr
        else:
            pcm = np.clip(arr.astype(np.float32), -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16)

        # Resample if needed (simple linear resampling)
        if self._sample_rate != self._output_rate:
            ratio = self._output_rate / self._sample_rate
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm.astype(np.float32))
            pcm = pcm.astype(np.int16)

        # Accumulate in buffer
        self._buffer = np.concatenate([self._buffer, pcm])

        # Enqueue complete frames
        while len(self._buffer) >= self._frame_size:
            chunk = self._buffer[:self._frame_size]
            self._buffer = self._buffer[self._frame_size:]
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    async def recv(self) -> AudioFrame:
        """Called by aiortc when it needs the next audio frame to encode."""
        chunk = await self._queue.get()

        # Create av.AudioFrame — s16 layout (signed 16-bit PCM)
        frame = AudioFrame(format="s16", layout="mono", samples=len(chunk))
        frame.sample_rate = self._output_rate
        frame.pts = self._pts
        frame.time_base = _AUDIO_TIME_BASE
        self._pts += len(chunk)

        # Copy PCM data into the frame
        frame.planes[0].update(chunk.tobytes())

        return frame


class AiortcAudioPublisher:
    """
    Publish audio to a LiveKit room via aiortc.

    Same public API as AudioPublisher for compatibility.
    """

    def __init__(
        self,
        client,  # AiortcLiveKitClient
        sample_rate: int = 16000,
        num_channels: int = 1,
        track_name: str = "aivatar-audio",
    ):
        self.client = client
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.track_name = track_name
        self.audio_track = AiortcAudioStreamTrack(
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._published = False

    async def push_audio(self, audio_array: np.ndarray) -> None:
        """Push a chunk of audio to the LiveKit room.

        Same API as AudioPublisher.push_audio.
        """
        await self.audio_track.push_audio(audio_array)

    async def aclose(self) -> None:
        """Close the audio track and release resources."""
        try:
            if self.audio_track:
                self.audio_track.stop()
        except Exception as exc:
            _logger.debug("AiortcAudioPublisher close failed: %s", exc)
