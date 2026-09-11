"""WebSocket security helpers: audio rate limiting, IP throttling, one-time JWTs.

State (_used_jti, _ip_connections) and sibling helpers are read via
`handler.*` at call time so handler-level monkey-patching keeps working.
"""

import time

import handler


class AudioRateLimiter:
    """Token-bucket rate limiter for audio messages (Fix 8)."""

    def __init__(self, rate_per_sec, burst, max_bytes_per_sec):
        self.rate = float(rate_per_sec)
        self.burst = float(burst)
        self.tokens = float(burst)
        self.last_refill = time.monotonic()
        self.byte_window_start = time.monotonic()
        self.bytes_in_window = 0
        self.max_bytes_per_sec = max_bytes_per_sec

    def allow(self, msg_size: int) -> tuple:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens < 1.0:
            return False, "message rate exceeded"
        self.tokens -= 1.0
        window_elapsed = now - self.byte_window_start
        if window_elapsed >= 1.0:
            self.byte_window_start = now
            self.bytes_in_window = 0
        self.bytes_in_window += msg_size
        if self.bytes_in_window > self.max_bytes_per_sec:
            return False, "byte rate exceeded"
        return True, ""


def _check_ip_limit(ip: str, max_conn: int, window_sec: int = 60) -> bool:
    """Check if an IP has exceeded the connection limit (Fix 15)."""
    now = time.time()
    handler._ip_connections[ip] = [t for t in handler._ip_connections[ip] if now - t < window_sec]
    if len(handler._ip_connections[ip]) >= max_conn:
        return False
    handler._ip_connections[ip].append(now)
    return True


def _prune_used_jti() -> None:
    now = time.time()
    for key, expiry in list(handler._used_jti.items()):
        if expiry < now:
            del handler._used_jti[key]


def _jti_is_available(claims: dict) -> bool:
    """Check one-time JWT availability without consuming it before the owner claim."""
    jti = claims.get("jti")
    if not jti:
        return True
    handler._prune_used_jti()
    return jti not in handler._used_jti


def _consume_jti(claims: dict) -> bool:
    """Consume a validated JTI only after the backend has granted ownership."""
    jti = claims.get("jti")
    if not jti:
        return True
    handler._prune_used_jti()
    if jti in handler._used_jti:
        return False
    handler._used_jti[jti] = claims.get("exp", time.time() + 300)
    return True


def _check_jti(claims: dict) -> bool:
    """Compatibility wrapper for callers outside the WebSocket claim path."""
    return handler._jti_is_available(claims) and handler._consume_jti(claims)
