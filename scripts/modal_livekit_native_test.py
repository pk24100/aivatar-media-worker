"""
Quick Modal validation test: does the official livekit Python SDK connect
from a Modal container without STATE_MISMATCH?

Usage:
    cd aivatar-media-worker
    modal run scripts/modal_livekit_native_test.py

This creates a throw-away test room, connects with livekit.rtc.Room,
optionally publishes one blank frame, then disconnects and reports state.
"""

import os
import modal

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")
    .apt_install("git", "git-lfs", "ffmpeg", "libsndfile1", "wget", "ca-certificates")
    .pip_install("ninja")
    .run_commands("pip install flash-attn --no-build-isolation || true")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App("aivatar-livekit-native-test", image=image)


@app.function(
    gpu="L4",
    timeout=120,
    secrets=[
        modal.Secret.from_name("livekit-secret"),
    ],
)
def test_livekit_native() -> str:
    """
    Connect to LiveKit using the official livekit.rtc SDK from inside a
    Modal container. Returns a human-readable status string.
    """
    import asyncio
    from livekit import rtc
    from livekit.api import LiveKitAPI, AccessToken, VideoGrants

    livekit_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not livekit_url:
        return "FAIL: LIVEKIT_URL not set"
    if not api_key or not api_secret:
        return "FAIL: LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set"

    room_name = "modal-native-test"
    identity = "modal-test-worker"

    # Generate a short-lived token
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name("Modal Test")
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    ).to_jwt()

    print(f"[test] LiveKit URL: {livekit_url}")
    print(f"[test] Room:        {room_name}")
    print(f"[test] Token (head): {token[:40]}...")

    async def _run():
        room = rtc.Room()
        print("[test] Connecting with rtc.Room.connect ...")
        try:
            await room.connect(
                livekit_url,
                token,
                options=rtc.RoomOptions(auto_subscribe=True),
            )
            print(f"[test] Room connected. connection_state={room.connection_state}")

            # Try publishing a blank video track to verify full negotiate path
            source = rtc.VideoSource(320, 240)
            track = rtc.LocalVideoTrack.create_video_track("test-video", source)
            publication = await room.local_participant.publish_track(track)
            print(f"[test] Published blank video track: sid={publication.sid}")

            # Briefly let ICE settle
            await asyncio.sleep(2)

            await room.disconnect()
            print("[test] Disconnected cleanly.")
            return "SUCCESS: native livekit.rtc connected + published from Modal container"
        except Exception as exc:
            return f"FAIL: {type(exc).__name__}: {exc}"

    return asyncio.run(_run())


@app.local_entrypoint()
def main():
    result = test_livekit_native.remote()
    print("=" * 60)
    print("RESULT:", result)
    print("=" * 60)
