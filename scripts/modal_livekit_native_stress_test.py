"""
Stress-test the official livekit.rtc SDK on Modal with production-identical
VideoPublisher + AudioPublisher loops. Runs for 60 seconds to surface any
UDP/DTLS/ICE stability issues under sustained load.

Usage:
    cd aivatar-media-worker
    modal run scripts/modal_livekit_native_stress_test.py

This mirrors the production stream_processor.py native path exactly:
- Video: H264, 512x512, 25 FPS, RGBA frames, lazy track publish
- Audio: 16 kHz mono, int16 PCM, lazy track publish
- Continuous loop for 60 s (no model loading)
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

app = modal.App("aivatar-livekit-native-stress", image=image)


@app.function(
    gpu="L4",
    timeout=300,
    secrets=[
        modal.Secret.from_name("livekit-secret"),
    ],
)
def stress_test_livekit_native() -> str:
    """
    Mirror the production streaming loop using the official livekit.rtc SDK.
    """
    import asyncio
    import time
    import traceback
    import numpy as np
    import logging
    from livekit import rtc
    from livekit.api import AccessToken, VideoGrants

    # Verbose logging so any failure is fully diagnosable in Modal logs
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _logger = logging.getLogger("stress_test")

    def _log(msg, *args):
        formatted = msg % args if args else msg
        print(formatted, flush=True)
        _logger.info(formatted)

    livekit_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not livekit_url:
        return "FAIL: LIVEKIT_URL not set"
    if not api_key or not api_secret:
        return "FAIL: LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set"

    room_name = "modal-stress-test"
    identity = "modal-stress-worker"

    # Production-identical parameters from FlashHeadStreamingEngine infer_params
    FPS = 25
    SAMPLE_RATE = 16000
    NUM_CHANNELS = 1
    WIDTH = 512
    HEIGHT = 512
    MAX_BITRATE = 3_000_000
    TRACK_NAME_VIDEO = "aivatar-video"
    TRACK_NAME_AUDIO = "aivatar-audio"
    DURATION_SECONDS = 60

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name("Modal Stress Test")
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    ).to_jwt()

    print(f"[stress] LiveKit URL: {livekit_url}")
    print(f"[stress] Room:        {room_name}")
    print(f"[stress] Duration:    {DURATION_SECONDS}s @ {FPS} FPS")
    print(f"[stress] Video:       {WIDTH}x{HEIGHT} H264")
    print(f"[stress] Audio:       {SAMPLE_RATE} Hz mono")

    # Capture every Room event for post-mortem diagnosis
    room_events = []

    async def _run():
        room = rtc.Room()

        @room.on("connected")
        def on_connected():
            _log("[EVENT] room.on_connected fired")
            room_events.append(("connected", time.monotonic()))

        @room.on("disconnected")
        def on_disconnected():
            _log("[EVENT] room.on_disconnected fired")
            room_events.append(("disconnected", time.monotonic()))

        @room.on("connection_quality_changed")
        def on_quality_changed(quality):
            _log("[EVENT] connection_quality_changed: %s", quality)
            room_events.append(("quality_changed", time.monotonic(), str(quality)))

        @room.on("reconnecting")
        def on_reconnecting():
            _log("[EVENT] room.on_reconnecting fired")
            room_events.append(("reconnecting", time.monotonic()))

        @room.on("reconnected")
        def on_reconnected():
            _log("[EVENT] room.on_reconnected fired")
            room_events.append(("reconnected", time.monotonic()))

        @room.on("track_published")
        def on_track_published(pub, participant):
            _log("[EVENT] track_published: %s by %s", pub.track_sid, participant.identity)
            room_events.append(("track_published", time.monotonic(), pub.track_sid))

        @room.on("track_unpublished")
        def on_track_unpublished(pub, participant):
            _log("[EVENT] track_unpublished: %s by %s", pub.track_sid, participant.identity)
            room_events.append(("track_unpublished", time.monotonic(), pub.track_sid))

        @room.on("track_subscribed")
        def on_track_subscribed(track, pub, participant):
            _log("[EVENT] track_subscribed: %s kind=%s from %s", pub.track_sid, track.kind, participant.identity)
            room_events.append(("track_subscribed", time.monotonic(), pub.track_sid, track.kind))

        @room.on("track_subscription_failed")
        def on_track_subscription_failed(track_sid, participant):
            _log("[EVENT] track_subscription_failed: %s from %s", track_sid, participant.identity)
            room_events.append(("track_subscription_failed", time.monotonic(), track_sid))

        @room.on("participant_connected")
        def on_participant_connected(participant):
            _log("[EVENT] participant_connected: %s", participant.identity)
            room_events.append(("participant_connected", time.monotonic(), participant.identity))

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant):
            _log("[EVENT] participant_disconnected: %s", participant.identity)
            room_events.append(("participant_disconnected", time.monotonic(), participant.identity))

        @room.on("local_track_published")
        def on_local_track_published(pub):
            _log("[EVENT] local_track_published: %s kind=%s", pub.sid, pub.kind)
            room_events.append(("local_track_published", time.monotonic(), pub.sid, pub.kind))

        @room.on("local_track_unpublished")
        def on_local_track_unpublished(pub):
            _log("[EVENT] local_track_unpublished: %s", pub.sid)
            room_events.append(("local_track_unpublished", time.monotonic(), pub.sid))

        _log("[stress] Connecting with rtc.Room.connect ...")

        connect_exc = None
        try:
            await room.connect(
                livekit_url,
                token,
                options=rtc.RoomOptions(auto_subscribe=True),
            )
        except Exception as exc:
            connect_exc = exc
            _log("[stress] CONNECT EXCEPTION: %s\n%s", exc, traceback.format_exc())
            return f"FAIL connect: {type(exc).__name__}: {exc}"

        _log("[stress] Connected. connection_state=%s", room.connection_state)
        conn_start = time.monotonic()

        # ---- Lazy track state (mirrors VideoPublisher + AudioPublisher) ----
        video_source = None
        video_track = None
        audio_source = None
        audio_track = None

        frame_interval = 1.0 / FPS
        audio_samples_per_frame = SAMPLE_RATE // FPS  # 640 samples @ 16kHz/25fps

        frames_sent = 0
        audio_frames_sent = 0
        errors = []

        try:
            while time.monotonic() - conn_start < DURATION_SECONDS:
                loop_start = time.monotonic()
                frame_idx = frames_sent + 1

                # ================== VIDEO (mirrors _ensure_track + _send_frame) ==================
                video_t0 = time.monotonic()
                try:
                    if video_track is None:
                        _log("[stress] Lazy-publishing video track ...")
                        video_source = rtc.VideoSource(WIDTH, HEIGHT)
                        video_track = rtc.LocalVideoTrack.create_video_track(
                            TRACK_NAME_VIDEO, video_source
                        )
                        options = rtc.TrackPublishOptions(
                            source=rtc.TrackSource.SOURCE_CAMERA,
                            simulcast=False,
                            video_encoding=rtc.VideoEncoding(
                                max_framerate=FPS,
                                max_bitrate=MAX_BITRATE,
                            ),
                            video_codec=rtc.VideoCodec.H264,
                        )
                        await room.local_participant.publish_track(video_track, options)
                        _log("[stress] Video track published")

                    # Generate a blank RGBA frame (mimics FlashHead output)
                    rgb = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
                    alpha = np.full((HEIGHT, WIDTH, 1), 255, dtype=np.uint8)
                    frame_rgba = np.concatenate([rgb, alpha], axis=2)

                    video_frame = rtc.VideoFrame(
                        WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, frame_rgba.tobytes()
                    )
                    video_source.capture_frame(video_frame)
                    frames_sent += 1
                except Exception as exc:
                    err_msg = f"video@{frame_idx}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    errors.append(err_msg)
                    _log("[stress] VIDEO ERROR at frame %d: %s", frame_idx, err_msg)

                video_elapsed = time.monotonic() - video_t0

                # ================== AUDIO (mirrors _ensure_track + push_audio) ==================
                audio_t0 = time.monotonic()
                try:
                    if audio_track is None:
                        _log("[stress] Lazy-publishing audio track ...")
                        audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
                        audio_track = rtc.LocalAudioTrack.create_audio_track(
                            TRACK_NAME_AUDIO, audio_source
                        )
                        options = rtc.TrackPublishOptions(
                            source=rtc.TrackSource.SOURCE_MICROPHONE
                        )
                        await room.local_participant.publish_track(audio_track, options)
                        _log("[stress] Audio track published")

                    # Generate silent int16 PCM (mimics audio pipeline)
                    pcm = np.zeros(audio_samples_per_frame, dtype=np.int16)
                    af = rtc.AudioFrame.create(
                        SAMPLE_RATE, NUM_CHANNELS, audio_samples_per_frame
                    )
                    buf = np.frombuffer(af.data, dtype=np.int16)
                    np.copyto(buf, pcm)
                    await audio_source.capture_frame(af)
                    audio_frames_sent += 1
                except Exception as exc:
                    err_msg = f"audio@{frame_idx}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    errors.append(err_msg)
                    _log("[stress] AUDIO ERROR at frame %d: %s", frame_idx, err_msg)

                audio_elapsed = time.monotonic() - audio_t0

                # ================== PACING + PER-FRAME DIAGNOSTICS ==================
                elapsed = time.monotonic() - loop_start
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    _log("[stress] LAG: behind by %.1fms at frame %d (video=%.1fms audio=%.1fms)",
                         -sleep_for * 1000, frame_idx, video_elapsed * 1000, audio_elapsed * 1000)

                # Periodic status every 5 seconds
                if frames_sent % (FPS * 5) == 0:
                    _log("[stress] STATUS frame=%d audio=%d conn_state=%s events=%d errors=%d",
                         frames_sent, audio_frames_sent, room.connection_state, len(room_events), len(errors))

        except Exception as exc:
            err_msg = f"loop: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            errors.append(err_msg)
            _log("[stress] LOOP ERROR: %s", err_msg)

        finally:
            _log("[stress] Disconnecting ...")
            try:
                await room.disconnect()
            except Exception as exc:
                err_msg = f"disconnect: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                errors.append(err_msg)
                _log("[stress] DISCONNECT ERROR: %s", err_msg)

        total_time = time.monotonic() - conn_start
        result = (
            f"COMPLETED in {total_time:.1f}s | "
            f"frames={frames_sent} audio_frames={audio_frames_sent} "
            f"errors={len(errors)} events={len(room_events)}"
        )
        if errors:
            result += f"\nFirst error: {errors[0][:200]}"
        # Dump captured events for post-mortem
        _log("[stress] Event log (%d events):", len(room_events))
        for ev in room_events:
            _log("  %s", str(ev))
        return result

    return asyncio.run(_run())


@app.local_entrypoint()
def main():
    result = stress_test_livekit_native.remote()
    print("=" * 60)
    print("RESULT:", result)
    print("=" * 60)
