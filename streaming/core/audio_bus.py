"""Canonical audio normalization and lifecycle queue."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np

from streaming.core.session import CanonicalAudioFrame, SessionConfig

try:
    from scipy.signal import resample
except ImportError:
    resample = None


CANONICAL_SAMPLE_RATE = 48_000
CANONICAL_CHANNELS = 1


@dataclass(slots=True)
class _BusState:
    utterance_active: bool = False
    connected: bool = False
    next_sequence: int = 0


def _as_float32(pcm: np.ndarray) -> np.ndarray:
    array = np.asarray(pcm)
    if array.dtype == np.int16:
        return array.astype(np.float32) / 32768.0
    if array.dtype == np.int32:
        return array.astype(np.float32) / 2147483648.0
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        return array.astype(np.float32) / scale
    return np.asarray(array, dtype=np.float32)


def normalize_channels(pcm: np.ndarray, channels: int, target_channels: int = CANONICAL_CHANNELS) -> np.ndarray:
    array = _as_float32(pcm)
    if channels < 1:
        raise ValueError("channels must be positive")
    if array.ndim == 1:
        if array.size % channels:
            raise ValueError("audio payload is not aligned to the channel count")
        array = array.reshape(-1, channels)
    elif array.ndim != 2:
        raise ValueError("audio must be a one- or two-dimensional array")
    if array.shape[1] != channels:
        raise ValueError("audio shape does not match channels")

    if target_channels == channels:
        return array
    if target_channels == 1:
        return array.mean(axis=1, keepdims=True)
    if target_channels == 2 and channels == 1:
        return np.repeat(array, 2, axis=1)
    raise ValueError(f"cannot convert {channels} channels to {target_channels}")


def resample_audio(pcm: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    array = np.asarray(pcm, dtype=np.float32)
    if source_rate == target_rate or array.shape[0] == 0:
        return array
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    target_length = max(1, round(array.shape[0] * target_rate / source_rate))
    if resample is not None:
        return np.asarray(resample(array, target_length, axis=0), dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, array.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, 1.0, target_length, endpoint=False)
    if array.ndim == 1:
        return np.interp(target_positions, source_positions, array).astype(np.float32)
    return np.stack(
        [np.interp(target_positions, source_positions, array[:, idx]) for idx in range(array.shape[1])],
        axis=1,
    ).astype(np.float32)


class CanonicalAudioBus:
    def __init__(
        self,
        *,
        canonical_sample_rate: int = CANONICAL_SAMPLE_RATE,
        canonical_channels: int = CANONICAL_CHANNELS,
        max_queue_size: int = 0,
    ):
        if canonical_sample_rate <= 0:
            raise ValueError("canonical_sample_rate must be positive")
        if canonical_channels not in (1, 2):
            raise ValueError("canonical_channels must be 1 or 2")
        self.canonical_sample_rate = canonical_sample_rate
        self.canonical_channels = canonical_channels
        self.queue: asyncio.Queue[CanonicalAudioFrame | None] = asyncio.Queue(maxsize=max_queue_size)
        self._state = _BusState()
        self._config: SessionConfig | None = None
        self._last_sequence = -1

    @property
    def is_connected(self) -> bool:
        return self._state.connected

    @property
    def last_sequence_number(self) -> int:
        return self._last_sequence

    async def connect(self, session_config: SessionConfig | dict | None = None) -> None:
        if isinstance(session_config, dict):
            self._config = SessionConfig(**session_config)
        else:
            self._config = session_config
        self._state.connected = True

    async def disconnect(self) -> None:
        self._state.connected = False
        await self.cancel_utterance()
        await self.close()

    async def start_utterance(self) -> None:
        if not self._state.connected:
            raise RuntimeError("audio bus is not connected")
        self._state.utterance_active = True

    async def push_audio(
        self,
        pcm: np.ndarray,
        sample_rate: int,
        channels: int,
        sequence_number: int | None = None,
        timestamp: float | None = None,
    ) -> CanonicalAudioFrame:
        if not self._state.connected:
            raise RuntimeError("audio bus is not connected")
        normalized = normalize_channels(pcm, channels, self.canonical_channels)
        normalized = resample_audio(normalized, sample_rate, self.canonical_sample_rate)
        if self.canonical_channels == 1:
            normalized = normalized.reshape(-1)
        sequence = self._next_sequence(sequence_number)
        frame = CanonicalAudioFrame(
            pcm=np.ascontiguousarray(normalized, dtype=np.float32),
            sample_rate=self.canonical_sample_rate,
            channels=self.canonical_channels,
            sequence_number=sequence,
            timestamp=time.time() if timestamp is None else timestamp,
            protocol_sequence_number=sequence_number,
        )
        await self.queue.put(frame)
        self._state.utterance_active = True
        return frame

    async def end_utterance(self, sequence_number: int | None = None) -> CanonicalAudioFrame:
        if not self._state.connected:
            raise RuntimeError("audio bus is not connected")
        sequence = self._next_sequence(sequence_number)
        frame = CanonicalAudioFrame(
            pcm=np.empty((0,), dtype=np.float32),
            sample_rate=self.canonical_sample_rate,
            channels=self.canonical_channels,
            sequence_number=sequence,
            timestamp=time.time(),
            is_end_of_utterance=True,
            protocol_sequence_number=sequence_number,
        )
        await self.queue.put(frame)
        self._state.utterance_active = False
        return frame

    async def cancel_utterance(self, sequence_number: int | None = None) -> CanonicalAudioFrame:
        self._clear_queue()
        if not self._state.connected:
            raise RuntimeError("audio bus is not connected")
        sequence = self._next_sequence(sequence_number)
        frame = CanonicalAudioFrame(
            pcm=np.empty((0,), dtype=np.float32),
            sample_rate=self.canonical_sample_rate,
            channels=self.canonical_channels,
            sequence_number=sequence,
            timestamp=time.time(),
            is_cancelled=True,
            protocol_sequence_number=sequence_number,
        )
        await self.queue.put(frame)
        self._state.utterance_active = False
        return frame

    async def end_session(self, sequence_number: int | None = None) -> None:
        if not self._state.connected:
            raise RuntimeError("audio bus is not connected")
        if self._state.utterance_active:
            await self.end_utterance(sequence_number)
        await self.queue.put(None)
        self._state.connected = False

    async def on_interrupted(self) -> None:
        await self.cancel_utterance()

    async def close(self) -> None:
        self._clear_queue()
        await self.queue.put(None)

    def _next_sequence(self, supplied: int | None) -> int:
        if supplied is None:
            sequence = self._state.next_sequence
        else:
            if supplied < 0:
                raise ValueError("sequence_number must be non-negative")
            sequence = max(supplied, self._state.next_sequence)
        self._state.next_sequence = sequence + 1
        self._last_sequence = sequence
        return sequence

    def _clear_queue(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
