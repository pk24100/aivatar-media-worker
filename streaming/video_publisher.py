import asyncio
import numpy as np
from livekit import rtc


class VideoPublisher:
    def __init__(
        self,
        room: rtc.Room,
        fps: int = 25,
        max_bitrate: int = 3_000_000,
        track_name: str = "aivatar-video",
    ):
        self.room = room
        self.fps = fps
        self.max_bitrate = max_bitrate
        self.track_name = track_name
        self.video_source = None
        self.track = None

    async def publish_from_queue(self, frame_queue):
        frame_interval = 1.0 / float(self.fps)
        while True:
            frame = await asyncio.to_thread(frame_queue.get)
            if frame is None:
                break
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
            await self._ensure_track(frame)
            await self._send_frame(frame)
            await asyncio.sleep(frame_interval)
        await self._cleanup()

    async def _ensure_track(self, frame: np.ndarray):
        if self.track is not None:
            return
        height, width = frame.shape[0], frame.shape[1]
        self.video_source = rtc.VideoSource(width, height)
        self.track = rtc.LocalVideoTrack.create_video_track(self.track_name, self.video_source)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=False,
            video_encoding=rtc.VideoEncoding(
                max_framerate=int(self.fps),
                max_bitrate=self.max_bitrate,
            ),
            video_codec=rtc.VideoCodec.H264,
        )
        await self.room.local_participant.publish_track(self.track, options)

    async def _send_frame(self, frame: np.ndarray):
        if frame.shape[2] == 3:
            alpha = np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=np.uint8)
            frame = np.concatenate([frame, alpha], axis=2)
        video_frame = rtc.VideoFrame(
            frame.shape[1],
            frame.shape[0],
            rtc.VideoBufferType.RGBA,
            frame.tobytes(),
        )
        self.video_source.capture_frame(video_frame)

    async def _cleanup(self):
        if self.room and self.room.connected:
            await self.room.disconnect()
