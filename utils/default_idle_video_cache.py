import json
import logging
import os
import pathlib
import threading

import boto3
import requests


logger = logging.getLogger("default_idle_video_cache")
MANIFEST_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "default_avatars.manifest.json"
MAX_IDLE_VIDEO_BYTES = int(os.getenv("IDLE_VIDEO_MAX_BYTES", str(8 * 1024 * 1024)))


class DefaultIdleVideoCache:
    """Snapshot compressed idle bytes for fixed platform default avatars only."""

    def __init__(self, manifest_path: pathlib.Path = MANIFEST_PATH):
        self.manifest_path = pathlib.Path(manifest_path)
        self._lock = threading.Lock()
        self._loaded = False
        self._assets = {}
        self._failed_keys = []

    def preload(self):
        with self._lock:
            if not self._loaded:
                self._loaded = True
                self._load_locked()
            return {
                "manifestPath": str(self.manifest_path),
                "cachedIdleVideoCount": len(self._assets),
                "failedIdleVideoKeys": list(self._failed_keys),
            }

    def get_bytes(self, key: str):
        if not key:
            return None
        self.preload()
        with self._lock:
            return self._assets.get(key)

    def _load_locked(self):
        if not self.manifest_path.is_file():
            return
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[default-idle-cache] Failed to read manifest %s: %s", self.manifest_path, exc)
            return

        for avatar in payload.get("avatars", []):
            idle_video = avatar.get("idleVideo") if isinstance(avatar, dict) else None
            if not isinstance(idle_video, dict):
                continue
            key = str(idle_video.get("key") or "").strip()
            url = str(idle_video.get("url") or "").strip()
            if not key:
                continue
            try:
                self._assets[key] = self._download(url) if url else self._download_from_r2(key)
            except Exception as exc:
                self._failed_keys.append(key)
                logger.warning("[default-idle-cache] Failed to preload key=%s: %s", key, exc)

    @staticmethod
    def _download(url: str) -> bytes:
        chunks = []
        total = 0
        with requests.get(url, timeout=(5, 30), stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IDLE_VIDEO_BYTES:
                    raise ValueError(f"idle video exceeds {MAX_IDLE_VIDEO_BYTES} bytes")
                chunks.append(chunk)
        video_bytes = b"".join(chunks)
        if len(video_bytes) < 12 or b"ftyp" not in video_bytes[:32]:
            raise ValueError("idle video is not an ISO BMFF/MP4 file")
        return video_bytes

    @staticmethod
    def _download_from_r2(key: str) -> bytes:
        endpoint = os.getenv("IDLE_VIDEO_R2_ENDPOINT")
        access_key = os.getenv("IDLE_VIDEO_R2_ACCESS_KEY_ID")
        secret_key = os.getenv("IDLE_VIDEO_R2_SECRET_ACCESS_KEY")
        bucket = os.getenv("DEFAULT_IDLE_VIDEO_R2_BUCKET_NAME", "facemode-idle-videos-default")
        if not endpoint or not access_key or not secret_key:
            raise ValueError("idle-video R2 credentials are not configured")
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )
        response = client.get_object(Bucket=bucket, Key=key)
        video_bytes = response["Body"].read(MAX_IDLE_VIDEO_BYTES + 1)
        if len(video_bytes) > MAX_IDLE_VIDEO_BYTES:
            raise ValueError(f"idle video exceeds {MAX_IDLE_VIDEO_BYTES} bytes")
        if len(video_bytes) < 12 or b"ftyp" not in video_bytes[:32]:
            raise ValueError("idle video is not an ISO BMFF/MP4 file")
        return video_bytes


default_idle_video_cache = DefaultIdleVideoCache()
