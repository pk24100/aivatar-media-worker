import asyncio
import os

import imageio.v2 as imageio
import numpy as np
from livekit import rtc

DEFAULT_FPS = 25
DEFAULT_MAX_BITRATE = 3_000_000


def _get_publish_fps(metadata):
    env_fps = os.getenv("LIVEKIT_PUBLISH_FPS")
    if env_fps:
        try:
            return float(env_fps)
        except ValueError:
            pass
    return float(metadata.get("fps") or DEFAULT_FPS)


async def _publish_video(mp4_path, livekit_token, livekit_url):
    reader = imageio.get_reader(mp4_path, format="ffmpeg")
    metadata = reader.get_meta_data()
    fps = _get_publish_fps(metadata)
    frame_interval = 1.0 / fps

    first_frame = reader.get_next_data()
    height, width = first_frame.shape[0], first_frame.shape[1]

    room = rtc.Room()
    await room.connect(livekit_url, livekit_token)

    video_source = rtc.VideoSource(width, height)
    track = rtc.LocalVideoTrack.create_video_track("aivatar-video", video_source)
    options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        simulcast=False,
        video_encoding=rtc.VideoEncoding(
            max_framerate=int(fps),
            max_bitrate=DEFAULT_MAX_BITRATE,
        ),
        video_codec=rtc.VideoCodec.H264,
    )
    await room.local_participant.publish_track(track, options)

    async def send_frame(frame):
        if frame.shape[2] == 3:
            alpha = np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=np.uint8)
            frame = np.concatenate([frame, alpha], axis=2)
        video_frame = rtc.VideoFrame(
            width,
            height,
            rtc.VideoBufferType.RGBA,
            frame.tobytes(),
        )
        video_source.capture_frame(video_frame)

    try:
        await send_frame(first_frame)
        await asyncio.sleep(frame_interval)

        for frame in reader:
            await send_frame(frame)
            await asyncio.sleep(frame_interval)
    finally:
        reader.close()
        await room.disconnect()


def publish_video_to_livekit(mp4_path, room_name, livekit_token, livekit_url):
    """
    Publish video frames to LiveKit via the server-side Python SDK.
    Requires LIVEKIT_URL (livekit_url) and a valid livekit_token.
    """
    print(f"Publishing video to LiveKit room via SDK: {room_name}")
    asyncio.run(_publish_video(mp4_path, livekit_token, livekit_url))
    return True
