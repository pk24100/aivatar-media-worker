# Publish avatar video frames to a LiveKit room.
import asyncio
import logging
import os
import time
import numpy as np
from livekit import rtc

_logger = logging.getLogger("VideoPublisher")


def _resolve_video_codec():
    """Resolve video codec and encoder backend from AIVATAR_VIDEO_CODEC env var.

    Supported values:
      - "vp8"    (default): VP8 software encoding via libvpx. Bypasses NVENC entirely.
      - "h264_sw": H264 software encoding via OpenH264. Bypasses NVENC.
      - "h264_hw": H264 hardware encoding via NVENC. Fails on Blackwell GPUs
                   with older LiveKit SDK NVENC wrappers.

    Returns (video_codec_enum, encoder_backend_value_or_None).
    """
    choice = os.environ.get("AIVATAR_VIDEO_CODEC", "vp8").lower().strip()

    if choice == "h264_hw":
        return rtc.VideoCodec.H264, _get_encoder_backend("hardware")
    elif choice == "h264_sw":
        return rtc.VideoCodec.H264, _get_encoder_backend("software")
    else:
        if choice != "vp8":
            _logger.warning(
                "AIVATAR_VIDEO_CODEC='%s' not recognized, defaulting to 'vp8'", choice,
            )
        return rtc.VideoCodec.VP8, None


def _get_encoder_backend(mode: str):
    """Get VideoEncoderBackend enum value for the given mode, or None if unavailable."""
    backend_enum = getattr(rtc, "VideoEncoderBackend", None)
    if backend_enum is None:
        _logger.warning(
            "VideoEncoderBackend not available in installed livekit-rtc version; "
            "AIVATAR_VIDEO_CODEC encoder backend selection will be ignored.",
        )
        return None
    attr = {
        "software": "ENCODER_BACKEND_SOFTWARE",
        "hardware": "ENCODER_BACKEND_HARDWARE",
        "nvenc": "ENCODER_BACKEND_NVENC",
    }.get(mode, "ENCODER_BACKEND_HARDWARE")
    return getattr(backend_enum, attr, None)


# Publish video frames to a LiveKit room.
class VideoPublisher:
    # Initialize publisher with room, fps, and bitrate settings.
    def __init__(
        self,
        room: rtc.Room,
        fps: int = 25,
        max_bitrate: int = 3_000_000,
        track_name: str = "aivatar-video",
        session_id: str = "",
    ):
        self.room = room
        self.fps = fps
        self.max_bitrate = max_bitrate
        self.track_name = track_name
        self.session_id = session_id
        self.video_source = None
        self.track = None
        self.first_frame_published = asyncio.Event()
        self._width = 0
        self._height = 0
        self._last_capture_at = None
        self._metrics_started_at = time.monotonic()
        self._metrics_frames = 0
        self._metrics_repeated_live_frames = 0
        self._metrics_idle_frames = 0
        self._metrics_max_gap_ms = 0.0
        self._metrics_max_capture_ms = 0.0

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
        next_frame_deadline = time.monotonic()
        _none_count = 0
        _first_frame_logged = False
        while True:
            frame = await asyncio.to_thread(state_manager.get_next_frame)

            if frame is None:
                _none_count += 1
                if _none_count % 50 == 1:
                    _logger.info(
                        "[VP-DIAG] No frame yet (None count=%d, ~%.1fs waiting). Track not created.",
                        _none_count, _none_count * frame_interval,
                    )
                next_frame_deadline += frame_interval
                await asyncio.sleep(max(0, next_frame_deadline - time.monotonic()))
                continue

            _none_count = 0
            if not _first_frame_logged:
                _logger.info(
                    "[VP-DIAG] First non-None frame received after %d None ticks (~%.1fs). Creating track...",
                    _none_count, _none_count * frame_interval,
                )
                _first_frame_logged = True

            await self._ensure_track(frame)
            await self._send_frame(frame, getattr(state_manager, "last_frame_source", "unknown"))

            # Advance from a fixed deadline so timer jitter cannot accumulate
            # into visible A/V drift during a long tail drain.
            next_frame_deadline += frame_interval
            await asyncio.sleep(max(0, next_frame_deadline - time.monotonic()))

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

        video_codec, encoder_backend = _resolve_video_codec()
        codec_name = "VP8" if video_codec == rtc.VideoCodec.VP8 else "H264"
        encoder_name = "auto"
        if encoder_backend is not None:
            be = getattr(rtc, "VideoEncoderBackend", None)
            if be is not None:
                if encoder_backend == be.ENCODER_BACKEND_SOFTWARE:
                    encoder_name = "software"
                elif encoder_backend == be.ENCODER_BACKEND_HARDWARE:
                    encoder_name = "hardware"
                elif encoder_backend == be.ENCODER_BACKEND_NVENC:
                    encoder_name = "nvenc"

        options_kwargs = dict(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=False,
            video_encoding=rtc.VideoEncoding(
                max_framerate=int(self.fps),
                max_bitrate=self.max_bitrate,
            ),
            video_codec=video_codec,
        )
        if encoder_backend is not None:
            options_kwargs["video_encoder"] = encoder_backend

        _logger.info(
            "[VP-DIAG] Codec=%s encoder=%s (AIVATAR_VIDEO_CODEC=%s)",
            codec_name, encoder_name, os.environ.get("AIVATAR_VIDEO_CODEC", "vp8"),
        )
        options = rtc.TrackPublishOptions(**options_kwargs)
        _logger.info("[VP-DIAG] Calling publish_track()...")
        _pub_t0 = time.monotonic()
        await self.room.local_participant.publish_track(self.track, options)
        self.first_frame_published.set()
        _pub_ms = round((time.monotonic() - _pub_t0) * 1000, 1)
        _logger.info("[VP-DIAG] publish_track() completed in %.1f ms", _pub_ms)

    # Convert and send a frame to the LiveKit video source.
    async def _send_frame(self, frame: np.ndarray, frame_source: str = "live"):
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
        now = time.monotonic()
        if self._last_capture_at is not None:
            gap_ms = (now - self._last_capture_at) * 1000
            self._metrics_max_gap_ms = max(self._metrics_max_gap_ms, gap_ms)
            if gap_ms > max(80.0, 2_000.0 / self.fps):
                _logger.warning(
                    "VIDEO_PUBLISH_GAP gapMs=%.1f targetFps=%d frameSource=%s",
                    gap_ms,
                    self.fps,
                    frame_source,
                )
        self._last_capture_at = now

        capture_started_at = time.monotonic()
        self.video_source.capture_frame(video_frame)
        capture_ms = (time.monotonic() - capture_started_at) * 1000
        self._metrics_frames += 1
        self._metrics_max_capture_ms = max(self._metrics_max_capture_ms, capture_ms)
        if frame_source == "repeated_live":
            self._metrics_repeated_live_frames += 1
        elif frame_source in ("idle", "transition_to_idle", "transition_to_live"):
            self._metrics_idle_frames += 1

        elapsed = time.monotonic() - self._metrics_started_at
        if elapsed >= 5.0:
            live_frames = self._metrics_frames - self._metrics_repeated_live_frames - self._metrics_idle_frames
            desync_pct = 0.0
            if self._metrics_frames > 0:
                desync_pct = (self._metrics_repeated_live_frames + self._metrics_idle_frames) / self._metrics_frames * 100.0
            _logger.info(
                "VIDEO_PUBLISH_METRICS session=%s windowMs=%.0f frames=%d effectiveFps=%.1f "
                "liveFrames=%d repeatedLiveFrames=%d idleFrames=%d "
                "desyncPct=%.1f maxGapMs=%.1f maxCaptureMs=%.1f",
                self.session_id,
                elapsed * 1000,
                self._metrics_frames,
                self._metrics_frames / elapsed,
                live_frames,
                self._metrics_repeated_live_frames,
                self._metrics_idle_frames,
                desync_pct,
                self._metrics_max_gap_ms,
                self._metrics_max_capture_ms,
            )
            self._metrics_started_at = time.monotonic()
            self._metrics_frames = 0
            self._metrics_repeated_live_frames = 0
            self._metrics_idle_frames = 0
            self._metrics_max_gap_ms = 0.0
            self._metrics_max_capture_ms = 0.0

    # Disconnect from the room if still connected.
    async def _cleanup(self):
        # livekit-rtc 1.x removed the `room.connected` boolean; check the
        # `connection_state` enum instead. See:
        # https://docs.livekit.io/reference/python/livekit/rtc/room.html
        if self.room and self.room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
            await self.room.disconnect()
