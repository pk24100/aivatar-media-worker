#!/bin/bash
set -e

# ===========================================
# Build Docker Image on RunPod Pod (Self-Contained)
# ===========================================
# This script runs INSIDE a RunPod pod that has:
#   - Network volume mounted at /runpod-volume (with models + ditto repo)
#   - Container disk: 50 GB recommended
#
# It generates ALL worker source code inline — no need to upload code.
# Models are copied from the network volume into the Docker build context.
#
# Usage:
#   1. Spin up a RunPod pod (any cheap GPU or CPU)
#      - Attach network volume: aivatar-models
#      - Container disk: 50 GB
#   2. Connect via Web Terminal
#   3. Copy-paste this entire script, or:
#      wget -O build.sh <raw-url> && bash build.sh
# ===========================================

DOCKER_USER="pk24100"
IMAGE_NAME="aivatar-worker"
IMAGE_TAG="latest"
FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}"
BUILD_DIR="/tmp/aivatar-build"

# Optional override if your RunPod pod mounts the network volume somewhere other than /runpod-volume.
# Example: VOLUME_PATH=/workspace ./build_on_runpod.sh
VOLUME_PATH="${VOLUME_PATH:-}"

echo "============================================"
echo "  AiVatar Worker — Docker Build on RunPod"
echo "============================================"

# ── Step 1: Install buildah (daemonless image builder) ──
# RunPod pods are containers — Docker daemon cannot run inside them.
# buildah builds OCI images without a daemon.
echo ""
echo "=== Step 1/6: Installing buildah ==="
if ! command -v buildah &> /dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq buildah git fuse-overlayfs
    # Configure buildah for rootless/container usage
    mkdir -p /etc/containers
    echo '[storage]' > /etc/containers/storage.conf
    echo 'driver = "vfs"' >> /etc/containers/storage.conf
fi
echo "buildah: $(buildah --version)"

_detect_volume_root() {
    if [ -n "${VOLUME_PATH}" ]; then
        if [ -d "${VOLUME_PATH}/models/ditto" ]; then
            echo "${VOLUME_PATH}"
            return 0
        fi
        echo ""
        return 1
    fi

    for p in /workspace /mnt /volume /data /root; do
        if [ -d "${p}/models/ditto" ]; then
            echo "${p}"
            return 0
        fi
    done

    # Last resort: limited-depth search.
    local found
    found="$(find / -maxdepth 5 -type d -path '*/models/ditto' 2>/dev/null | head -n 1)"
    if [ -n "${found}" ]; then
        echo "$(dirname "$(dirname "${found}")")"
        return 0
    fi

    echo ""
    return 1
}

# ── Step 2: Docker Hub login ──
echo ""
echo "=== Step 2/6: Docker Hub Login ==="
echo "Enter your Docker Hub access token when prompted:"
buildah login -u ${DOCKER_USER} docker.io

# ── Step 3: Generate worker source code ──
echo ""
echo "=== Step 3/6: Generating worker source code ==="
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}/streaming ${BUILD_DIR}/utils

# --- Dockerfile ---
cat > ${BUILD_DIR}/Dockerfile << 'DOCKERFILE_EOF'
FROM nvcr.io/nvidia/tensorrt:23.08-py3

RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libsndfile1 wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /app/requirements.txt

COPY handler.py entrypoint.sh /app/
COPY streaming/ /app/streaming/
COPY utils/ /app/utils/
RUN chmod +x /app/entrypoint.sh

# Bake models + ditto repo into the image
COPY models/ditto /app/models/ditto
COPY ditto-talkinghead /app/ditto-talkinghead

CMD ["python3", "-u", "handler.py"]
DOCKERFILE_EOF

# --- requirements.txt ---
cat > ${BUILD_DIR}/requirements.txt << 'REQ_EOF'
--extra-index-url https://download.pytorch.org/whl/cu121
runpod
torch
torchvision
torchaudio
onnxruntime-gpu
librosa
tqdm
filetype
imageio
imageio-ffmpeg
opencv-python-headless
scikit-image
cython
numpy==2.0.1
cuda-python
polygraphy
colored
livekit
soundfile
requests
REQ_EOF

# --- entrypoint.sh ---
cat > ${BUILD_DIR}/entrypoint.sh << 'ENTRY_EOF'
#!/bin/bash
echo "Starting AIVatar serverless worker..."
python3 -u handler.py
ENTRY_EOF

# --- handler.py ---
cat > ${BUILD_DIR}/handler.py << 'HANDLER_EOF'
import asyncio
import os
import tempfile

import runpod
import torch

from streaming.stream_processor import run_streaming_session
from utils.ditto_runner import run_ditto_inference
from utils.livekit_publisher import publish_video_to_livekit

MODEL_ROOT = os.getenv("MODEL_ROOT", "/app/models/ditto")
DITTO_REPO_PATH = os.getenv("DITTO_REPO_PATH", "/app/ditto-talkinghead")


def _verify_models():
    cfg = os.path.join(MODEL_ROOT, "ditto_cfg")
    if not os.path.isdir(cfg):
        raise FileNotFoundError(f"Model config not found at {cfg}")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) >= (8, 9):
            preferred = "ditto_trt_ada"
        else:
            preferred = "ditto_trt_Ampere_Plus"
        preferred_path = os.path.join(MODEL_ROOT, preferred)
        if os.path.isdir(preferred_path) and len(os.listdir(preferred_path)) > 0:
            print(f"Using data_root: {preferred_path}")
            return preferred_path

    for candidate in ["ditto_trt_Ampere_Plus", "ditto_trt_ada", "ditto_trt_custom", "ditto_onnx", "ditto_pytorch"]:
        path = os.path.join(MODEL_ROOT, candidate)
        if os.path.isdir(path) and len(os.listdir(path)) > 0:
            print(f"Using data_root: {path}")
            return path
    raise FileNotFoundError("No Ditto model directory found (checked baked-in /app/models/ditto)")


DATA_ROOT = _verify_models()


def _is_streaming(event):
    if event.get("streaming") is True:
        return True
    if str(event.get("mode", "")).lower() == "streaming":
        return True
    env_flag = os.getenv("AIVATAR_STREAMING", "").strip().lower()
    return env_flag in {"1", "true", "yes"}

def handler(job):
    """
    Called per request by Runpod Serverless.
    job["input"] example:
    {
      "roomName": "room_abc",
      "livekitToken": "...",
      "audioPath": "s3://bucket/audio.wav",
      "sourceImage": "s3://bucket/avatar.png"
    }
    """
    event = job["input"]

    roomName = event.get("roomName")
    livekitToken = event.get("livekitToken")
    audioPath = event.get("audioPath")
    sourceImage = event.get("sourceImage")

    if _is_streaming(event):
        if not roomName or not livekitToken:
            raise ValueError("roomName and livekitToken are required for streaming mode")
        if not sourceImage:
            raise ValueError("sourceImage is required for streaming mode")
        livekit_url = os.getenv("LIVEKIT_URL")
        if not livekit_url:
            raise ValueError("LIVEKIT_URL is required for streaming mode")
        asyncio.run(
            run_streaming_session(
                room_name=roomName,
                livekit_token=livekitToken,
                livekit_url=livekit_url,
                model_root=MODEL_ROOT,
                source_image=sourceImage,
            )
        )
        return {"status": "ok", "mode": "streaming"}

    if not audioPath:
        raise ValueError("audioPath is required for offline inference")
    if not sourceImage:
        raise ValueError("sourceImage is required for offline inference")

    # produce a temporary mp4 path
    out_mp4 = tempfile.mktemp(suffix=".mp4")

    # 1. run Ditto inference and generate an mp4
    run_ditto_inference(
        MODEL_ROOT,
        audioPath,
        sourceImage,
        out_mp4
    )

    # 2. publish the video into the LiveKit room
    publish_video_to_livekit(
        out_mp4,
        roomName,
        livekitToken,
        os.getenv("LIVEKIT_URL")
    )

    return {"status": "ok", "output": out_mp4}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
HANDLER_EOF

# --- streaming/__init__.py ---
cat > ${BUILD_DIR}/streaming/__init__.py << 'INIT_EOF'
# Streaming utilities for LiveKit + Ditto integration.
INIT_EOF

# --- streaming/audio_chunker.py ---
cat > ${BUILD_DIR}/streaming/audio_chunker.py << 'CHUNKER_EOF'
import numpy as np


class AudioChunker:
    def __init__(self, sample_rate=16000, chunksize=(3, 5, 2)):
        self.sample_rate = sample_rate
        self.chunksize = chunksize
        self.chunk_step = int(chunksize[1] * 0.04 * sample_rate)
        self.chunk_total = int(sum(chunksize) * 0.04 * sample_rate) + 80
        self.left_pad = int(chunksize[0] * 0.04 * sample_rate)
        self.buffer = np.zeros((self.left_pad,), dtype=np.float32)
        self.cursor = 0

    def add_samples(self, samples):
        if samples is None or len(samples) == 0:
            return []
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self.buffer = np.concatenate([self.buffer, samples], axis=0)
        chunks = []
        while self.cursor + self.chunk_total <= len(self.buffer):
            chunk = self.buffer[self.cursor : self.cursor + self.chunk_total]
            chunks.append(chunk)
            self.cursor += self.chunk_step
        if self.cursor > self.chunk_total:
            self.buffer = self.buffer[self.cursor :]
            self.cursor = 0
        return chunks

    def flush(self):
        if self.cursor >= len(self.buffer):
            return []
        remaining = len(self.buffer) - self.cursor
        if remaining <= 0:
            return []
        pad = max(0, self.chunk_total - remaining)
        if pad:
            self.buffer = np.concatenate(
                [self.buffer, np.zeros((pad,), dtype=self.buffer.dtype)], axis=0
            )
        chunk = self.buffer[self.cursor : self.cursor + self.chunk_total]
        self.cursor = len(self.buffer)
        return [chunk]
CHUNKER_EOF

# --- streaming/audio_subscriber.py ---
cat > ${BUILD_DIR}/streaming/audio_subscriber.py << 'SUBSCRIBER_EOF'
import asyncio
from typing import Optional

import numpy as np
from livekit import rtc


class AudioSubscriber:
    def __init__(
        self,
        room: rtc.Room,
        sample_rate: int = 16000,
        num_channels: int = 1,
        queue_size: int = 200,
    ):
        self.room = room
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.queue = asyncio.Queue(maxsize=queue_size)
        self._ready = asyncio.Event()
        self._audio_task = None
        self._track_sid = None

    def bind(self):
        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if self._audio_task is not None:
                return
            self._track_sid = publication.sid
            audio_stream = rtc.AudioStream(
                track,
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
            )
            self._audio_task = asyncio.create_task(self._consume(audio_stream))
            self._ready.set()

        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = publication.track
                if track and track.kind == rtc.TrackKind.KIND_AUDIO:
                    on_track_subscribed(track, publication, participant)
                    return

    async def wait_until_ready(self, timeout: Optional[float] = None):
        if timeout is None:
            await self._ready.wait()
            return True
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def read(self, timeout: Optional[float] = None):
        if timeout is None:
            return await self.queue.get()
        return await asyncio.wait_for(self.queue.get(), timeout=timeout)

    async def _consume(self, audio_stream: rtc.AudioStream):
        try:
            async for frame_event in audio_stream:
                frame = frame_event.frame
                data = np.frombuffer(frame.data, dtype=np.int16)
                if self.num_channels > 1:
                    data = data.reshape(-1, self.num_channels).mean(axis=1)
                audio = data.astype(np.float32) / 32768.0
                if self.queue.full():
                    _ = self.queue.get_nowait()
                await self.queue.put(audio)
        finally:
            await self.queue.put(None)
SUBSCRIBER_EOF

# --- streaming/ditto_streaming.py ---
cat > ${BUILD_DIR}/streaming/ditto_streaming.py << 'DITTO_STREAM_EOF'
import os
import queue
import sys
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch


class QueueFrameWriter:
    def __init__(self, _video_path, fps=25, frame_queue=None, max_queue=200, **_kwargs):
        self.fps = fps
        self.frame_queue = frame_queue or queue.Queue(maxsize=max_queue)

    def __call__(self, img, fmt="bgr"):
        frame = img
        if fmt == "bgr":
            frame = frame[..., ::-1]
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def close(self):
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass


class DittoStreamingEngine:
    def __init__(
        self,
        model_root: str,
        source_path: str,
        cfg_path: Optional[str] = None,
        data_root: Optional[str] = None,
        repo_path: Optional[str] = None,
        fps: int = 25,
        queue_size: int = 200,
        chunksize: Tuple[int, int, int] = (3, 5, 2),
    ):
        self.model_root = model_root
        self.source_path = source_path
        self.cfg_path = cfg_path or self._resolve_cfg_path(model_root)
        self.data_root = data_root or self._resolve_data_root(model_root)
        self.repo_path = repo_path or os.getenv("DITTO_REPO_PATH", "/app/ditto-talkinghead")
        self.fps = fps
        self.chunksize = chunksize
        self.frame_queue = queue.Queue(maxsize=queue_size)

        stream_module = self._load_stream_module(self.repo_path)
        stream_module.VideoWriterByImageIO = lambda video_path, fps=25, **kwargs: QueueFrameWriter(
            video_path,
            fps=fps,
            frame_queue=self.frame_queue,
            max_queue=queue_size,
            **kwargs,
        )
        self._stream_module = stream_module
        self._sdk = stream_module.StreamSDK(self.cfg_path, self.data_root)
        output_path = tempfile.mktemp(suffix=".mp4")
        self._sdk.setup(
            source_path,
            output_path,
            online_mode=True,
            N_d=-1,
        )
        self._sdk.setup_Nd(N_d=-1)

    def run_chunk(self, audio_chunk: np.ndarray):
        self._sdk.run_chunk(audio_chunk, chunksize=self.chunksize)

    def close(self):
        self._sdk.close()

    @staticmethod
    def _load_stream_module(repo_path: str):
        if not repo_path:
            raise ValueError("DITTO_REPO_PATH is required to load stream_pipeline_online.py")
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            import stream_pipeline_online as stream_module
        except Exception as exc:
            raise ImportError(
                "Unable to import stream_pipeline_online from Ditto repository. "
                "Ensure DITTO_REPO_PATH points to the ditto-talkinghead repo."
            ) from exc
        return stream_module

    @staticmethod
    def _resolve_cfg_path(model_root: str) -> str:
        online_cfg = os.path.join(model_root, "ditto_cfg", "v0.4_hubert_cfg_trt_online.pkl")
        if os.path.isfile(online_cfg):
            return online_cfg
        fallback_cfg = os.path.join(model_root, "ditto_cfg", "v0.4_hubert_cfg_trt.pkl")
        if os.path.isfile(fallback_cfg):
            return fallback_cfg
        raise FileNotFoundError(
            "Ditto config not found. Expected v0.4_hubert_cfg_trt_online.pkl in ditto_cfg."
        )

    @staticmethod
    def _resolve_data_root(model_root: str) -> str:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) >= (8, 9):
                preferred = "ditto_trt_ada"
            else:
                preferred = "ditto_trt_Ampere_Plus"
            preferred_path = os.path.join(model_root, preferred)
            if os.path.isdir(preferred_path) and len(os.listdir(preferred_path)) > 0:
                return preferred_path

        candidates = [
            "ditto_trt_Ampere_Plus",
            "ditto_trt_ada",
            "ditto_trt_3090",
            "ditto_trt_custom",
            "ditto_onnx",
            "ditto_pytorch",
        ]
        for candidate in candidates:
            path = os.path.join(model_root, candidate)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError("Ditto model directory not found under model_root.")
DITTO_STREAM_EOF

# --- streaming/stream_processor.py ---
cat > ${BUILD_DIR}/streaming/stream_processor.py << 'STREAM_PROC_EOF'
import asyncio
import os
from typing import Optional, Tuple

from livekit import rtc

from streaming.audio_chunker import AudioChunker
from streaming.audio_subscriber import AudioSubscriber
from streaming.ditto_streaming import DittoStreamingEngine
from streaming.video_publisher import VideoPublisher


def _parse_chunksize(value: Optional[str]) -> Tuple[int, int, int]:
    if not value:
        return (3, 5, 2)
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("DITTO_CHUNKSIZE must have 3 comma-separated ints, e.g. 3,5,2")
    return tuple(parts)


async def run_streaming_session(
    room_name: str,
    livekit_token: str,
    livekit_url: str,
    model_root: str,
    source_image: str,
):
    room = rtc.Room()
    await room.connect(
        livekit_url,
        livekit_token,
        options=rtc.RoomOptions(auto_subscribe=True),
    )

    sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    num_channels = int(os.getenv("AUDIO_CHANNELS", "1"))
    chunk_timeout = float(os.getenv("AUDIO_SUBSCRIBE_TIMEOUT", "15"))
    chunksize = _parse_chunksize(os.getenv("DITTO_CHUNKSIZE"))

    subscriber = AudioSubscriber(room, sample_rate=sample_rate, num_channels=num_channels)
    subscriber.bind()

    ready = await subscriber.wait_until_ready(timeout=chunk_timeout)
    if not ready:
        await room.disconnect()
        raise RuntimeError("Timed out waiting for LiveKit audio track.")

    engine = DittoStreamingEngine(
        model_root=model_root,
        source_path=source_image,
        chunksize=chunksize,
    )
    chunker = AudioChunker(sample_rate=sample_rate, chunksize=chunksize)

    fps = int(os.getenv("LIVEKIT_PUBLISH_FPS", "25"))
    publisher = VideoPublisher(room, fps=fps)
    publish_task = asyncio.create_task(publisher.publish_from_queue(engine.frame_queue))

    try:
        while True:
            audio = await subscriber.read()
            if audio is None:
                break
            for chunk in chunker.add_samples(audio):
                await asyncio.to_thread(engine.run_chunk, chunk)

        for chunk in chunker.flush():
            await asyncio.to_thread(engine.run_chunk, chunk)

        await asyncio.to_thread(engine.close)
        await publish_task
    finally:
        if room.connected:
            await room.disconnect()
STREAM_PROC_EOF

# --- streaming/video_publisher.py ---
cat > ${BUILD_DIR}/streaming/video_publisher.py << 'VIDPUB_EOF'
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
VIDPUB_EOF

# --- utils/ditto_runner.py ---
cat > ${BUILD_DIR}/utils/ditto_runner.py << 'RUNNER_EOF'
import os
import subprocess

DITTO_REPO_PATH = os.getenv("DITTO_REPO_PATH", "/app/ditto-talkinghead")

def run_ditto_inference(model_root, audio_path, source_image, output_mp4):
    """
    Calls official Ditto inference script.
    Uses the converted TensorRT engines if available, else ONNX.
    """

    # best practice: pick TRT if exists (fastest)
    trt_path = os.path.join(model_root, "ditto_trt_3090")
    cfg_pkl  = os.path.join(model_root, "ditto_cfg/v0.4_hubert_cfg_trt.pkl")

    # if no TRT dir present, fallback to onnx
    if not os.path.isdir(trt_path) or len(os.listdir(trt_path)) == 0:
        trt_path = os.path.join(model_root, "ditto_onnx")

    # call the official inference script
    inference_script = os.path.join(DITTO_REPO_PATH, "inference.py")
    cmd = [
        "python3", inference_script,
        "--data_root", trt_path,
        "--cfg_pkl", cfg_pkl,
        "--audio_path", audio_path,
        "--source_path", source_image,
        "--output_path", output_mp4
    ]
    print("Running Ditto command:", cmd)
    subprocess.run(cmd, check=True)
    return output_mp4
RUNNER_EOF

# --- utils/livekit_publisher.py ---
cat > ${BUILD_DIR}/utils/livekit_publisher.py << 'LKPUB_EOF'
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
    await room.connect(
        livekit_url,
        livekit_token,
        options=rtc.RoomOptions(auto_subscribe=True),
    )

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
LKPUB_EOF

echo "Worker source code generated: $(find ${BUILD_DIR} -type f | wc -l) files"

# ── Step 4: Copy models + ditto repo from network volume ──
echo ""
echo "=== Step 4/6: Copying models from network volume ==="

VOLUME_ROOT="$(_detect_volume_root)"
if [ -z "${VOLUME_ROOT}" ]; then
    echo "ERROR: Could not find the network volume mount containing models/ditto."
    echo "Run:  ls -la / | head"
    echo "and:  mount | head -n 50"
    echo "Then rerun with: VOLUME_PATH=/path ./build_on_runpod.sh"
    exit 1
fi

echo "Using volume mount: ${VOLUME_ROOT}"

mkdir -p ${BUILD_DIR}/models/ditto
echo "Copying ditto_cfg..."
cp -r "${VOLUME_ROOT}/models/ditto/ditto_cfg" ${BUILD_DIR}/models/ditto/

echo "Copying TRT engines..."
cp -r "${VOLUME_ROOT}/models/ditto/ditto_trt_ada" ${BUILD_DIR}/models/ditto/ 2>/dev/null && echo "  Ada TRT: OK" || echo "  Ada TRT: not found (skipped)"
cp -r "${VOLUME_ROOT}/models/ditto/ditto_trt_Ampere_Plus" ${BUILD_DIR}/models/ditto/ 2>/dev/null && echo "  Ampere TRT: OK" || echo "  Ampere TRT: not found (skipped)"
cp -r "${VOLUME_ROOT}/models/ditto/ditto_onnx" ${BUILD_DIR}/models/ditto/ 2>/dev/null && echo "  ONNX: OK" || echo "  ONNX: not found (skipped)"

echo "Copying ditto-talkinghead repo..."
if [ -d "${VOLUME_ROOT}/ditto-talkinghead" ]; then
    cp -r "${VOLUME_ROOT}/ditto-talkinghead" ${BUILD_DIR}/ditto-talkinghead
    echo "  ditto-talkinghead: OK"
else
    echo "  Not on volume — cloning from GitHub..."
    git clone --depth 1 https://github.com/antgroup/ditto-talkinghead ${BUILD_DIR}/ditto-talkinghead
fi

echo ""
echo "Build context summary:"
du -sh ${BUILD_DIR}
du -sh ${BUILD_DIR}/models/ditto/ 2>/dev/null
ls -la ${BUILD_DIR}/models/ditto/

# ── Step 5: Build image with buildah (daemonless) ──
echo ""
echo "=== Step 5/6: Building image with buildah ==="
echo "This may take 10-20 minutes..."
cd ${BUILD_DIR}
buildah bud -t ${FULL_IMAGE} .

echo ""
echo "Image built successfully."

# ── Step 6: Push to Docker Hub ──
echo ""
echo "=== Step 6/6: Pushing to Docker Hub ==="
buildah push ${FULL_IMAGE} docker://docker.io/${FULL_IMAGE}

echo ""
echo "============================================"
echo "  BUILD & PUSH COMPLETE!"
echo "============================================"
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""
echo "NEXT STEPS:"
echo "  1. Go to RunPod Console -> Serverless -> aivatar-ditto -> Edit Endpoint"
echo "  2. Detach the network volume (Clear all under Network Volumes)"
echo "  3. Enable FlashBoot toggle"
echo "  4. Set Active Workers (min) = 0, Max Workers = 3"
echo "  5. Update template env vars:"
echo "       MODEL_ROOT=/app/models/ditto"
echo "       DITTO_REPO_PATH=/app/ditto-talkinghead"
echo "  6. Save and test!"
echo ""
echo "You can now TERMINATE this pod to stop charges."
