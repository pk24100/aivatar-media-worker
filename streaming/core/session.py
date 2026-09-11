"""Transport-neutral media session contracts and canonical media data types."""

from abc import ABC, abstractmethod
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CanonicalAudioFrame:
    pcm: np.ndarray
    sample_rate: int
    channels: int
    sequence_number: int
    timestamp: float
    is_end_of_utterance: bool = False
    is_cancelled: bool = False
    protocol_sequence_number: int | None = None


@dataclass(slots=True)
class SessionConfig:
    audio_encoding: str = "pcm_s16le"
    sample_rate: int = 48_000
    channels: int = 1
    avatar_id: str = ""
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MediaInputAdapter(ABC):
    @abstractmethod
    async def connect(self, session_config: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def start_utterance(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def push_audio(self, pcm: np.ndarray, sample_rate: int, channels: int) -> None:
        raise NotImplementedError

    @abstractmethod
    async def end_utterance(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def cancel_utterance(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def on_interrupted(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError


class MediaEgressAdapter(ABC):
    @abstractmethod
    async def connect(self, room_config: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def publish_video_frame(self, frame: np.ndarray, pts: float) -> None:
        raise NotImplementedError

    @abstractmethod
    async def publish_audio_chunk(self, pcm: np.ndarray, sample_rate: int) -> None:
        raise NotImplementedError

    @abstractmethod
    async def wait_for_ready(self) -> None:
        raise NotImplementedError

    def start_video(self, state_manager) -> asyncio.Task:
        """Start the shared state-manager video loop for simple transports."""
        video_task = getattr(self, "video_task", None)
        if video_task is not None:
            return video_task
        self.video_task = asyncio.create_task(
            self._publish_video_from_state_manager(state_manager),
            name=f"video-egress-{getattr(self, 'session_id', '')}",
        )
        return self.video_task

    async def _publish_video_from_state_manager(self, state_manager) -> None:
        fps = max(float(getattr(self, "fps", 25)), 1.0)
        frame_interval = 1.0 / fps
        while self.is_connected:
            frame = await asyncio.to_thread(state_manager.get_next_frame)
            if frame is not None:
                await self.publish_video_frame(frame, time.monotonic())
            await asyncio.sleep(frame_interval)

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError
