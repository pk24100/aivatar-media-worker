import asyncio
import numpy as np
from livekit import rtc


# Publish video frames to a LiveKit room.
class VideoPublisher:
    # Initialize publisher with room, fps, and bitrate settings.
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

    # Publish frames from a raw queue at the target FPS.
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

    async def publish_from_state_manager(self, state_manager):
        """
        Publishes frames continuously at the target FPS.
        Pulls frames from the state manager which handles Live/Idle transitions.

        Pacing rule: emit AT MOST one frame per 1/fps interval. If the inference
        pipeline produces frames in bursts and the live queue grows behind us, we
        drop the older frames and keep only the freshest one for the next tick --
        we never burst-send queued frames back-to-back (that was the legacy
        behaviour and caused the avatar to talk far faster than the audio).
        """
        frame_interval = 1.0 / float(self.fps)
        while True:
            # get_next_frame handles Live/Idle selection. We always pull exactly
            # one frame per tick so the output cadence stays locked at self.fps.
            frame = await asyncio.to_thread(state_manager.get_next_frame)

            if frame is None:
                # Neither live nor idle frames available right now; keep the
                # publish loop ticking at the target rate.
                await asyncio.sleep(frame_interval)
                continue

            await self._ensure_track(frame)
            await self._send_frame(frame)
            await asyncio.sleep(frame_interval)

        # Unreachable unless the task is cancelled, but here for safety.
        await self._cleanup()

    # Lazily create and publish the local video track.
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

    # Convert and send a frame to the LiveKit video source.
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

    # Disconnect from the room if still connected.
    async def _cleanup(self):
        # livekit-rtc 1.x removed the `room.connected` boolean; check the
        # `connection_state` enum instead. See:
        # https://docs.livekit.io/reference/python/livekit/rtc/room.html
        if self.room and self.room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
            await self.room.disconnect()
