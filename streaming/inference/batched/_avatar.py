"""Avatar URL/path resolution and download with SSRF protection."""

import os
import time
import logging
import tempfile
import uuid
from urllib.parse import urlparse

import requests
import boto3

from utils.default_avatar_cache import default_avatar_cache
from utils.ssrf_fetch import validate_url

logger = logging.getLogger("BatchedStreamingEngine")

# --- SSRF prevention (centralized log-only, see utils/ssrf_fetch.py) ---
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

_IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",
}


def _is_valid_image_bytes(data: bytes) -> bool:
    if len(data) < 12:
        return False
    for magic, fmt in _IMAGE_MAGIC_BYTES.items():
        if data.startswith(magic):
            if fmt == "webp" and data[8:12] != b"WEBP":
                continue
            return True
    return False


_AVATAR_DOWNLOAD_MAX_ATTEMPTS = 3
_AVATAR_DOWNLOAD_BASE_DELAY = 1.0

_TRANSIENT_NETWORK_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _resolve_avatar_path(avatar_image_path: str) -> str:
    """Resolve avatar URL/path to a local file, downloading if needed."""
    cached_avatar_path = default_avatar_cache.get_cached_path(avatar_image_path)
    if cached_avatar_path:
        logger.info("Using cached default avatar for %s -> %s", avatar_image_path, cached_avatar_path)
        return cached_avatar_path

    parsed = urlparse(avatar_image_path)
    if parsed.scheme in ("http", "https"):
        try:
            validate_url(avatar_image_path, "image")
        except Exception as exc:
            logger.warning("SSRF_WOULD_BLOCK kind=image url=%s err=%s", avatar_image_path, exc)
            if os.getenv("SSRF_ENFORCE", "0").strip() == "1":
                raise
            # LOG-ONLY: still proceed with existing download below.
        suffix = os.path.splitext(parsed.path)[1] or ".jpg"
        last_error: Exception | None = None
        for attempt in range(1, _AVATAR_DOWNLOAD_MAX_ATTEMPTS + 1):
            try:
                with requests.get(avatar_image_path, timeout=(10, 30), stream=True,
                                  headers={"Referer": "https://facemode.io"}) as response:
                    response.raise_for_status()

                    content_length = int(response.headers.get("Content-Length", 0))
                    if content_length > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"Avatar image exceeds {MAX_DOWNLOAD_BYTES} bytes (got {content_length})"
                        )

                    temp_path = os.path.join(tempfile.gettempdir(), f"avatar_{uuid.uuid4().hex}{suffix}")
                    downloaded = b""
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += chunk
                        if len(downloaded) > MAX_DOWNLOAD_BYTES:
                            raise ValueError(
                                f"Avatar image exceeded {MAX_DOWNLOAD_BYTES} bytes during download"
                            )

                if not _is_valid_image_bytes(downloaded):
                    raise ValueError("Downloaded content is not a valid image (JPEG/PNG/WebP)")

                with open(temp_path, "wb") as handle:
                    handle.write(downloaded)
                return temp_path
            except _TRANSIENT_NETWORK_ERRORS as exc:
                last_error = exc
                if attempt < _AVATAR_DOWNLOAD_MAX_ATTEMPTS:
                    delay = _AVATAR_DOWNLOAD_BASE_DELAY * attempt
                    logger.warning(
                        "Avatar download attempt %d/%d failed: %s. Retrying in %.1fs",
                        attempt, _AVATAR_DOWNLOAD_MAX_ATTEMPTS, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Avatar download failed after %d attempts: %s",
                        _AVATAR_DOWNLOAD_MAX_ATTEMPTS, exc,
                    )
                    raise
        if last_error is not None:
            raise last_error
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 sourceImage: {avatar_image_path}")
        suffix = os.path.splitext(key)[1] or ".png"
        temp_path = os.path.join(tempfile.gettempdir(), f"avatar_{uuid.uuid4().hex}{suffix}")
        boto3.client("s3").download_file(bucket, key, temp_path)
        return temp_path
    if parsed.scheme == "file":
        avatar_image_path = parsed.path
    elif parsed.scheme:
        raise ValueError(f"Unsupported sourceImage scheme: {parsed.scheme}")
    if not os.path.exists(avatar_image_path):
        raise FileNotFoundError(f"Avatar image not found: {avatar_image_path}")
    return avatar_image_path
