"""
Publishes video frames to a LiveKit room via aiortc.

Video publisher using aiortc for Modal deployments.

Drop-in replacement for VideoPublisher that uses aiortc's VideoStreamTrack
instead of livekit-rtc's VideoSource/LocalVideoTrack. Same public API.
"""

import asyncio
import logging
import fractions
from typing import Optional

import numpy as np
from av import VideoFrame
from aiortc import VideoStreamTrack

_logger = logging.getLogger("AiortcVideoPublisher")

# Time base for 90 kHz RTP clock
_VIDEO_TIME_BASE = fractions.Fraction(1, 90000)


class AiortcVideoStreamTrack(VideoStreamTrack):
    """
    Custom video track that serves numpy frames via recv().

    Frames are pushed from the FlashHead engine thread via push_frame().
    The aiortc transport calls recv() to pull frames for encoding/sending.
    """

    kind = "video"

    def __init__(self, fps: int = 25):
        super().__init__()
        self._fps = fps
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._frame_count = 0
        self._start_time: Optional[float] = None

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Push a numpy frame (H, W, 3 or H, W, 4) into the track.

        Thread-safe: can be called from the FlashHead engine thread.
        Drops oldest frame if queue is full (to avoid backpressure stalls).
        """
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    async def recv(self) -> VideoFrame:
        """
        Called by aiortc when it needs the next frame to encode and send.

        Blocks until a frame is available, then converts numpy → av.VideoFrame.
        """
        frame_np = await self._queue.get()

        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()

        # Convert numpy to av.VideoFrame
        if frame_np.shape[2] == 4:
            # RGBA → RGB
            frame_np = frame_np[:, :, :3]

        # Create VideoFrame from numpy (RGB24 format)
        av_frame = VideoFrame.from_ndarray(frame_np, format="rgb24")

        # Set timestamps
        self._frame_count += 1
        av_frame.pts = int(self._frame_count * 90000 / self._fps)
        av_frame.time_base = _VIDEO_TIME_BASE

        return av_frame


class AiortcVideoPublisher:
    """
    Publish video frames to a LiveKit room via aiortc.

    Same public API as VideoPublisher for compatibility.
    """

    def __init__(
        self,
        client,  # AiortcLiveKitClient
        fps: int = 25,
        max_bitrate: int = 3_000_000,
        track_name: str = "aivatar-video",
    ):
        self.client = client
        self.fps = fps
        self.max_bitrate = max_bitrate
        self.track_name = track_name
        self.video_track = AiortcVideoStreamTrack(fps=fps)
        self._published = False

    async def _ensure_published(self) -> None:
        """Ensure the video track is published to the room."""
        if self._published:
            return
        # Publishing is deferred — the track is added to the client when
        # publish_tracks is called from stream_processor.
        self._published = True

    async def publish_from_queue(self, frame_queue) -> None:
        """Publish frames from a raw queue at the target FPS."""
        frame_interval = 1.0 / float(self.fps)
        while True:
            frame = await asyncio.to_thread(frame_queue.get)
            if frame is None:
                break
            # Drop stale frames
            while frame_queue.qsize() > 0:
                try:
                    next_frame = frame_queue.get_nowait()
                    if next_frame is None:
                        frame = None
                        break
                    frame = next_frame
                except Exception:
                    break
            if frame is None:
                break
            self.video_track.push_frame(frame)
            await asyncio.sleep(frame_interval)

    async def publish_from_state_manager(self, state_manager) -> None:
        """
        Publish frames continuously at the target FPS.
        Pulls frames from the state manager which handles Live/Idle transitions.
        """
        frame_interval = 1.0 / float(self.fps)
        while True:
            frame = await asyncio.to_thread(state_manager.get_next_frame)
            if frame is None:
                await asyncio.sleep(frame_interval)
                continue
            self.video_track.push_frame(frame)
            await asyncio.sleep(frame_interval)

    async def _cleanup(self) -> None:
        """Stop the video track."""
        if self.video_track:
            self.video_track.stop()
