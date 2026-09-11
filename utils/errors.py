"""Central error types + production logging helpers for the media worker.

Additive only: no existing error codes, WS messages, HTTP statuses, or
control flow are changed here. Import these helpers and add one-line
log calls inside existing except blocks.

Sentry is wired via init_sentry() (DSN-gated no-op when SENTRY_DSN is
unset). Call sites use report_error() and never change for Sentry.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from urllib.parse import urlsplit, urlunsplit

_CODE_RE = None


def _valid_code(code: str) -> bool:
    import re

    global _CODE_RE
    if _CODE_RE is None:
        _CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,64}$")
    return bool(_CODE_RE.match(code))


class AppError(Exception):
    """Base application error with a stable machine-readable code."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status_code: int = 500):
        super().__init__(message)
        self.code = code if isinstance(code, str) and _valid_code(code) else "INTERNAL_ERROR"
        self.status_code = status_code


class NotFoundError(AppError):
    def __init__(self, resource: str, sid: str):
        super().__init__(f"{resource} not found: {sid}", "NOT_FOUND", 404)


class ValidationError(AppError):
    def __init__(self, message: str, details=None):
        super().__init__(message, "VALIDATION_ERROR", 400)
        self.details = details or []


def sanitize_url_for_logging(raw_url) -> str:
    """Keep scheme/host/path only. Never log query or fragment (tokens, signatures)."""
    if not isinstance(raw_url, str) or not raw_url:
        return "[Missing-URL]"
    try:
        parts = urlsplit(raw_url)
        if parts.scheme and parts.netloc:
            return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", "", ""))
        path = raw_url.split("?", 1)[0].split("#", 1)[0] or "/"
        return path
    except Exception:
        return "[Invalid-URL]"


_SECRET_KEY_RE = None


def _is_secret_key(key) -> bool:
    import re

    global _SECRET_KEY_RE
    if _SECRET_KEY_RE is None:
        _SECRET_KEY_RE = re.compile(
            r"(token|secret|passwd|password|pwd|credential|bearer|signature|api[_-]?key|auth|jwt|livekit|cookie)",
            re.IGNORECASE,
        )
    if not isinstance(key, str):
        return False
    if key in {"session_id", "sessionId", "requestId", "event", "source"}:
        return False
    if key.endswith("Id") or key.endswith("_id"):
        return False
    return bool(_SECRET_KEY_RE.search(key))


def _redact_value(key, value):
    try:
        if _is_secret_key(key):
            return "[Redacted]"
        if isinstance(value, str):
            if "url" in str(key).lower():
                return sanitize_url_for_logging(value)
            if len(value) > 500:
                return value[:500] + "...[truncated]"
        return value
    except Exception:
        return "[Redacted]"


def _before_send(event, hint):
    """Scrub secrets from Sentry events. Never drops the event on scrub failure."""
    try:
        request = event.get("request") or {}
        url = request.get("url")
        if isinstance(url, str):
            request["url"] = sanitize_url_for_logging(url)
        request.pop("cookies", None)
        request.pop("query_string", None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            for k in list(headers):
                if _is_secret_key(str(k)):
                    headers[k] = "[Redacted]"
        event.pop("user", None)
        extra = event.get("extra")
        if isinstance(extra, dict):
            for k in list(extra):
                extra[k] = _redact_value(k, extra[k])
        contexts = event.get("contexts")
        if isinstance(contexts, dict):
            for _cv in contexts.values():
                if isinstance(_cv, dict):
                    for k in list(_cv):
                        _cv[k] = _redact_value(k, _cv[k])
        crumbs = (event.get("breadcrumbs") or {}).get("values") or []
        for crumb in crumbs:
            data = crumb.get("data") if isinstance(crumb, dict) else None
            if isinstance(data, dict):
                for k in list(data):
                    data[k] = _redact_value(k, data[k])
    except Exception:
        pass
    return event


_SENTRY_READY = False


def is_sentry_enabled() -> bool:
    return _SENTRY_READY


def init_sentry() -> bool:
    """Initialize Sentry once when SENTRY_DSN is set. Never throws."""
    global _SENTRY_READY
    if _SENTRY_READY:
        return True
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
            traces_sample_rate=0.0,
            send_default_pii=False,
            before_send=_before_send,
        )
        _SENTRY_READY = True
        try:
            logging.getLogger("worker.errors").info("SENTRY_INITIALIZED")
        except Exception:
            pass
        return True
    except Exception:
        return False


def _capture(exc, context: dict) -> None:
    if exc is None or not _SENTRY_READY:
        return None
    try:
        import sentry_sdk

        with sentry_sdk.isolation_scope() as scope:
            for k, v in (context or {}).items():
                try:
                    scope.set_extra(str(k)[:64], _redact_value(k, v))
                except Exception:
                    pass
        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
    return None


def report_error(exc, context: dict | None = None) -> None:
    """Log + future-Sentry hook. Never throws. Does not change control flow."""
    try:
        logger = logging.getLogger("worker.errors")
        ctx = dict(context or {})
        session = ctx.get("session_id") or ctx.get("session") or "-"
        code = getattr(exc, "code", None) if exc is not None else None
        logger.error(
            "ERROR_REPORTED session=%s code=%s type=%s",
            session,
            code if isinstance(code, str) else "-",
            type(exc).__name__ if exc is not None else "Unknown",
            exc_info=exc if isinstance(exc, BaseException) else None,
        )
        try:
            _capture(exc, ctx)
        except Exception:
            pass
    except Exception:
        pass


def log_exception(logger, exc, message: str, session_id: str = "-", level: str = "error", **extra) -> None:
    """One-line never-throw log with traceback. Preserves original handling.

    Usage: except Exception as exc: log_exception(logger, exc, "WS_HANDLER_ERROR", session_id)
    """
    try:
        fn = getattr(logger, level, None) or logger.error
        if extra:
            safe = {str(k)[:64]: v for k, v in extra.items()}
            fn("%s session=%s err=%s extra=%s", message, session_id, type(exc).__name__, safe, exc_info=exc)
        else:
            fn("%s session=%s err=%s", message, session_id, type(exc).__name__, exc_info=exc)
    except Exception:
        pass
    try:
        report_error(exc, {"session_id": session_id, "event": message})
    except Exception:
        pass


def make_task_guard(logger, session_id: str = "-", event: str = "background_task"):
    """Done-callback that logs task failures. Additive: keep existing callbacks."""

    def _guard(task: asyncio.Task) -> None:
        try:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                log_exception(logger, exc, event, session_id)
        except Exception:
            pass

    return _guard


_hooks_installed = False


def install_error_hooks() -> bool:
    """Global last-resort logging. Never changes handling, only adds logs."""
    global _hooks_installed
    if _hooks_installed:
        return True
    _hooks_installed = True
    logger = logging.getLogger("worker.errors")

    def _sys_hook(exc_type, exc_value, tb):
        try:
            logger.error("UNCAUGHT_EXCEPTION type=%s", getattr(exc_type, "__name__", "Unknown"), exc_info=(exc_type, exc_value, tb))
            _capture(exc_value, {"source": "sys.excepthook"})
        except Exception:
            pass

    def _thread_hook(args):
        try:
            logger.error(
                "UNCAUGHT_THREAD_EXCEPTION thread=%s type=%s",
                getattr(args, "thread", None) and args.thread.name,
                getattr(getattr(args, "exc_type", None), "__name__", "Unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            _capture(getattr(args, "exc_value", None), {"source": "threading.excepthook"})
        except Exception:
            pass

    try:
        sys.excepthook = _sys_hook
    except Exception:
        pass
    try:
        threading.excepthook = _thread_hook
    except Exception:
        pass
    try:
        loop = asyncio.get_event_loop()
        handler = loop.get_exception_handler()

        def _asyncio_handler(loop, context):
            try:
                exc = context.get("exception")
                logger.error("UNHANDLED_ASYNCIO_ERROR msg=%s", str(context.get("message", ""))[:200], exc_info=exc if isinstance(exc, BaseException) else None)
                _capture(exc, {"source": "asyncio", "message": str(context.get("message", ""))[:200]})
            except Exception:
                pass
            try:
                if handler is not None:
                    handler(loop, context)
            except Exception:
                pass

        try:
            loop.set_exception_handler(_asyncio_handler)
        except Exception:
            pass
    except Exception:
        pass
    return True
