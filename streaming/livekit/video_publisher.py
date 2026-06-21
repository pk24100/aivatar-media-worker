# Publish avatar video frames to a LiveKit room.
import asyncio
import logging
import time
import numpy as np
from livekit import rtc

_logger = logging.getLogger("VideoPublisher")


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
        self._width = 0
        self._height = 0

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

        The engine is paced to real-time in _process_available_audio (one slice
        per slice_len/tgt_fps seconds), so the frame_queue stays small and we
        simply consume at tgt_fps.  If the queue does grow temporarily (e.g.
        during initial buffering), we skip to the freshest frame to avoid
        playing video faster than the audio.
        """
        frame_interval = 1.0 / float(self.fps)
        _none_count = 0
        _first_frame_logged = False
        while True:
            _cycle_start = time.monotonic()

            frame = await asyncio.to_thread(state_manager.get_next_frame)

            if frame is None:
                _none_count += 1
                if _none_count % 50 == 1:
                    _logger.info(
                        "[VP-DIAG] No frame yet (None count=%d, ~%.1fs waiting). Track not created.",
                        _none_count, _none_count * frame_interval,
                    )
                _elapsed = time.monotonic() - _cycle_start
                await asyncio.sleep(max(0, frame_interval - _elapsed))
                continue

            _none_count = 0
            if not _first_frame_logged:
                _logger.info(
                    "[VP-DIAG] First non-None frame received after %d None ticks (~%.1fs). Creating track...",
                    _none_count, _none_count * frame_interval,
                )
                _first_frame_logged = True

            await self._ensure_track(frame)
            await self._send_frame(frame)

            # Real-time pacing: sleep only the remaining time to maintain
            # target FPS.  asyncio.sleep(frame_interval) alone sleeps for
            # AT LEAST frame_interval, but to_thread + _send_frame add
            # ~8-10ms overhead, dropping effective FPS to ~21 and causing
            # slow-motion video.
            _elapsed = time.monotonic() - _cycle_start
            await asyncio.sleep(max(0, frame_interval - _elapsed))

        # Unreachable unless the task is cancelled, but here for safety.
        await self._cleanup()

    # Lazily create and publish the local video track.
    async def _ensure_track(self, frame: np.ndarray):
        if self.track is not None:
            return
        self._height, self._width = int(frame.shape[0]), int(frame.shape[1])
        _logger.info("[VP-DIAG] Creating video track %dx%d", self._width, self._height)
        self.video_source = rtc.VideoSource(self._width, self._height)
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
        _logger.info("[VP-DIAG] Calling publish_track()...")
        _pub_t0 = time.monotonic()
        await self.room.local_participant.publish_track(self.track, options)
        _pub_ms = round((time.monotonic() - _pub_t0) * 1000, 1)
        _logger.info("[VP-DIAG] publish_track() completed in %.1f ms", _pub_ms)

    # Convert and send a frame to the LiveKit video source.
    async def _send_frame(self, frame: np.ndarray):
        if frame is None or frame.ndim < 2:
            return
        if self.video_source is None:
            return
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            _logger.warning(
                "Frame size mismatch: got %dx%d, expected %dx%d. Skipping.",
                frame.shape[1], frame.shape[0], self._width, self._height,
            )
            return

        # Convert RGB (H,W,3) to RGBA (H,W,4) and send to LiveKit.
        # The SDK's native C++ layer handles RGBA→I420 conversion for H264.
        # This matches the official LiveKit python-sdks publisher example.
        rgba = np.ascontiguousarray(
            np.concatenate(
                [frame, np.full((*frame.shape[:2], 1), 255, dtype=np.uint8)],
                axis=2,
            )
        )
        video_frame = rtc.VideoFrame(
            self._width,
            self._height,
            rtc.VideoBufferType.RGBA,
            rgba.tobytes(),
        )
        self.video_source.capture_frame(video_frame)

    # Disconnect from the room if still connected.
    async def _cleanup(self):
        # livekit-rtc 1.x removed the `room.connected` boolean; check the
        # `connection_state` enum instead. See:
        # https://docs.livekit.io/reference/python/livekit/rtc/room.html
        if self.room and self.room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
            await self.room.disconnect()
