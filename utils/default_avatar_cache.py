import io
import json
import logging
import os
import pathlib
import re
import tempfile
import threading
from urllib.parse import urlparse

import requests
from PIL import Image

from utils.ssrf_fetch import validate_url


logger = logging.getLogger("default_avatar_cache")

MAX_DEFAULT_AVATAR_BYTES = 10 * 1024 * 1024
MANIFEST_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "default_avatars.manifest.json"
MATERIALIZED_CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "aivatar-default-avatar-cache"

_FORMAT_SUFFIXES = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


def _safe_avatar_name(avatar_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", avatar_id).strip("._")
    return safe or "default_avatar"


class DefaultAvatarCache:
    def __init__(self, manifest_path: pathlib.Path = MANIFEST_PATH):
        self.manifest_path = pathlib.Path(manifest_path)
        self._lock = threading.Lock()
        self._loaded = False
        self._manifest_found = False
        self._url_to_entry = {}
        self._entry_count = 0
        self._failed_ids = []

    def preload(self):
        with self._lock:
            if not self._loaded:
                self._loaded = True
                self._load_manifest_locked()
            return self._status_locked()

    def get_cached_path(self, source_url: str):
        if not source_url:
            return None
        self.preload()
        with self._lock:
            entry = self._url_to_entry.get(source_url)
            if entry is None:
                return None
            return self._materialize_locked(entry)

    def _status_locked(self):
        return {
            "manifestPath": str(self.manifest_path),
            "manifestFound": self._manifest_found,
            "cachedAvatarCount": self._entry_count,
            "failedAvatarIds": list(self._failed_ids),
        }

    def _load_manifest_locked(self):
        if not self.manifest_path.is_file():
            logger.info(
                "[default-avatar-cache] Manifest not found at %s. Falling back to per-session URL fetch.",
                self.manifest_path,
            )
            return

        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "[default-avatar-cache] Failed to read manifest %s: %s. Falling back to per-session URL fetch.",
                self.manifest_path,
                exc,
            )
            return

        avatars = payload.get("avatars")
        if not isinstance(avatars, list):
            logger.warning(
                "[default-avatar-cache] Manifest %s is missing an 'avatars' list. Falling back to per-session URL fetch.",
                self.manifest_path,
            )
            return

        self._manifest_found = True
        for avatar in avatars:
            if not isinstance(avatar, dict):
                logger.warning("[default-avatar-cache] Skipping non-object avatar manifest entry: %r", avatar)
                continue

            avatar_id = str(avatar.get("id") or "").strip()
            match_urls = avatar.get("matchUrls")
            if not avatar_id or not isinstance(match_urls, list):
                logger.warning(
                    "[default-avatar-cache] Skipping invalid avatar entry id=%r matchUrls=%r",
                    avatar.get("id"),
                    match_urls,
                )
                continue

            normalized_urls = [str(url).strip() for url in match_urls if str(url).strip()]
            if not normalized_urls:
                logger.warning(
                    "[default-avatar-cache] Skipping avatar id=%s because matchUrls is empty",
                    avatar_id,
                )
                continue

            preload_url = normalized_urls[0]
            try:
                avatar_bytes = self._download_avatar_bytes(preload_url)
                suffix = self._detect_suffix(avatar_bytes, preload_url)
                entry = {
                    "id": avatar_id,
                    "bytes": avatar_bytes,
                    "suffix": suffix,
                    "urls": tuple(normalized_urls),
                    "materializedPath": None,
                }
                inserted = False
                for url in normalized_urls:
                    if url in self._url_to_entry:
                        logger.warning(
                            "[default-avatar-cache] Duplicate matchUrl %s for avatar id=%s. Keeping first mapping.",
                            url,
                            avatar_id,
                        )
                        continue
                    self._url_to_entry[url] = entry
                    inserted = True
                if inserted:
                    self._entry_count += 1
                else:
                    logger.warning(
                        "[default-avatar-cache] Avatar id=%s had no unique matchUrls after dedupe.",
                        avatar_id,
                    )
            except Exception as exc:
                self._failed_ids.append(avatar_id)
                logger.warning(
                    "[default-avatar-cache] Failed to preload avatar id=%s from %s: %s. Matching requests will fall back to runtime fetch.",
                    avatar_id,
                    preload_url,
                    exc,
                )

        logger.info(
            "[default-avatar-cache] Preloaded %d avatar(s) from %s. failed=%d",
            self._entry_count,
            self.manifest_path,
            len(self._failed_ids),
        )

    def _download_avatar_bytes(self, url: str) -> bytes:
        try:
            validate_url(url, "image")
        except Exception as exc:
            logger.warning("SSRF_WOULD_BLOCK kind=image url=%s err=%s", url, exc)
            if os.getenv("SSRF_ENFORCE", "0").strip() == "1":
                raise
            # LOG-ONLY: still proceed with existing download below.
        chunks = []
        total = 0
        with requests.get(url, timeout=(10, 30), stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DEFAULT_AVATAR_BYTES:
                    raise ValueError(
                        f"default avatar exceeds {MAX_DEFAULT_AVATAR_BYTES} bytes while downloading"
                    )
                chunks.append(chunk)
        avatar_bytes = b"".join(chunks)
        if not avatar_bytes:
            raise ValueError("default avatar download returned empty content")
        self._detect_suffix(avatar_bytes, url)
        return avatar_bytes

    def _detect_suffix(self, avatar_bytes: bytes, preload_url: str) -> str:
        try:
            with Image.open(io.BytesIO(avatar_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(avatar_bytes)) as image:
                image_format = (image.format or "").upper()
        except Exception as exc:
            raise ValueError(f"downloaded content is not a valid image: {exc}") from exc

        suffix = _FORMAT_SUFFIXES.get(image_format)
        if suffix:
            return suffix

        parsed = urlparse(preload_url)
        fallback_suffix = pathlib.Path(parsed.path).suffix.lower()
        if fallback_suffix:
            return fallback_suffix
        return ".img"

    def _materialize_locked(self, entry):
        cached_path = entry.get("materializedPath")
        if cached_path and pathlib.Path(cached_path).is_file():
            return cached_path

        MATERIALIZED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_avatar_name(entry['id'])}{entry['suffix']}"
        path = MATERIALIZED_CACHE_DIR / filename
        path.write_bytes(entry["bytes"])
        entry["materializedPath"] = str(path)
        return entry["materializedPath"]


default_avatar_cache = DefaultAvatarCache()
