"""Centralized SSRF validation helper - LOG-ONLY mode by default.

Kinds:
  image    -> ALLOWED_IMAGE_DOMAINS (comma-separated, empty=open)
  video    -> ALLOWED_IDLE_VIDEO_DOMAINS (comma-separated, empty=open)
  callback -> ALLOWED_CALLBACK_HOSTS (comma-separated, empty=open)

Rules:
- Hostname-only allowlist matching (exact, lowercased). Query string
  (?X-Amz-..., ?dl=1) is preserved and never stripped.
- Allow http when NODE_ENV != production. In production require https.
- DNS resolution via socket.getaddrinfo + ipaddress private checks.
- Bare /tmp or filesystem paths are always allowed OUTSIDE this helper
  (callers only invoke validate_url for http/https; s3://, file:// and
  local paths keep existing boto3/filesystem handling).
- No file:// or s3:// handling inside helper (raises for non-http/https).
- SSRF_ENFORCE=0/1 gate (default 0 = log warning + allow, 1 = raise).

LOG-ONLY wiring: callers wrap validate_url in try/except, log
SSRF_WOULD_BLOCK on failure, but still proceed with existing
requests.get/put/post. Do NOT change redirects, do NOT strip query,
do NOT force https unconditionally.

To flip to enforce later:
  1. Set SSRF_ENFORCE=1 in env.
  2. Callers already re-raise when SSRF_ENFORCE==1 (see call-site comments),
     so helper raising will block the request. No code change needed.
     Alternatively remove the log-only try/except swallow and let
     validate_url propagate directly.
"""

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse


logger = logging.getLogger("ssrf_fetch")

_KIND_TO_ENV = {
    "image": "ALLOWED_IMAGE_DOMAINS",
    "video": "ALLOWED_IDLE_VIDEO_DOMAINS",
    "callback": "ALLOWED_CALLBACK_HOSTS",
}


def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except ValueError:
        return True


def _parse_allowlist(env_name: str) -> set:
    raw = os.getenv(env_name, "")
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


def is_ssrf_enforce() -> bool:
    return os.getenv("SSRF_ENFORCE", "0").strip() == "1"


def _validate_host(host: str) -> None:
    if not host:
        raise ValueError("URL is missing hostname")
    try:
        resolved = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname: {host}") from exc
    if not resolved:
        raise ValueError(f"Cannot resolve hostname: {host}")
    for _family, _type, _proto, _canon, sockaddr in resolved:
        ip_str = sockaddr[0]
        if _is_private_ip(ip_str):
            raise ValueError(
                f"URL resolves to private/internal IP {ip_str}. "
                f"SSRF prevention: rejecting."
            )


def _validate_url_strict(url: str, kind: str) -> None:
    if kind not in _KIND_TO_ENV:
        raise ValueError(f"Unknown SSRF kind '{kind}'. Expected one of {sorted(_KIND_TO_ENV)}")
    if not url or not isinstance(url, str):
        raise ValueError("URL is empty")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported URL scheme '{parsed.scheme}'. "
            f"SSRF helper only handles http/https (callers keep s3/file handling)."
        )
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL is missing hostname")
    env_name = _KIND_TO_ENV[kind]
    allowed = _parse_allowlist(env_name)
    if allowed and hostname not in allowed:
        raise ValueError(
            f"URL domain '{hostname}' not in allowlist for kind='{kind}'. "
            f"Allowed: {sorted(allowed)}"
        )
    if os.getenv("NODE_ENV", "").lower() == "production" and parsed.scheme != "https":
        raise ValueError("Only HTTPS URLs are allowed in production")
    # Query (?X-Amz-..., ?dl=1) is preserved - hostname-only matching above.
    # Redirects are untouched (callers keep default requests behavior).
    _validate_host(hostname)


def validate_url(url: str, kind: str) -> bool:
    """Validate URL for kind in image|video|callback with SSRF_ENFORCE gate.

    SSRF_ENFORCE=1 -> raise on failure.
    Default (0/unset) -> log warning with SSRF_WOULD_BLOCK + allow (return True).

    Callers in LOG-ONLY mode should still wrap in try/except logging
    SSRF_WOULD_BLOCK and proceeding, plus re-raise when SSRF_ENFORCE==1,
    so flipping to enforce later is env-only.
    """
    try:
        _validate_url_strict(url, kind)
    except Exception as exc:
        if is_ssrf_enforce():
            raise
        try:
            from utils.errors import sanitize_url_for_logging
            _safe_url = sanitize_url_for_logging(url)
        except Exception:
            _safe_url = "[Redacted-URL]"
        logger.warning("SSRF_WOULD_BLOCK kind=%s url=%s err=%s", kind, _safe_url, exc)
        return True
    return True
