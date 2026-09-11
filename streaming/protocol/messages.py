"""Canonical WebSocket media protocol."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any


logger = logging.getLogger(__name__)

AUDIO_ENCODINGS = frozenset({"pcm_s16le", "pcm_f32le", "base64_json"})
MIN_INPUT_SAMPLE_RATE = 8_000
MAX_INPUT_SAMPLE_RATE = 48_000
PREFERRED_SAMPLE_RATES = frozenset(
    {8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000}
)
SUPPORTED_CHANNELS = frozenset({1, 2})
MIN_UTTERANCE_SILENCE_MS = 100
MAX_UTTERANCE_SILENCE_MS = 5_000
ERROR_CODES = frozenset(
    {
        "INVALID_FORMAT",
        "NOT_NEGOTIATED",
        "SESSION_NOT_FOUND",
        "RATE_LIMITED",
        "INTERNAL_ERROR",
        "AVATAR_NOT_READY",
    }
)


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, fatal: bool = False):
        self.code = code
        self.message = message
        self.fatal = fatal
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ProtocolMessage:
    type: str


@dataclass(frozen=True, slots=True)
class StartMessage(ProtocolMessage):
    session_id: str
    audio_encoding: str
    sample_rate: int
    channels: int
    avatar_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    utterance_silence_ms: int | None = None

    def __init__(
        self,
        session_id: str,
        audio_encoding: str,
        sample_rate: int,
        channels: int,
        avatar_id: str = "",
        metadata: dict[str, Any] | None = None,
        utterance_silence_ms: int | None = None,
    ):
        object.__setattr__(self, "type", "start")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "audio_encoding", audio_encoding)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "avatar_id", avatar_id)
        object.__setattr__(self, "metadata", metadata or {})
        object.__setattr__(self, "utterance_silence_ms", utterance_silence_ms)


@dataclass(frozen=True, slots=True)
class BinaryAudioMessage(ProtocolMessage):
    data: bytes

    def __init__(self, data: bytes):
        object.__setattr__(self, "type", "binary_audio")
        object.__setattr__(self, "data", bytes(data))


@dataclass(frozen=True, slots=True)
class AudioMessage(ProtocolMessage):
    data: str
    seq: int

    def __init__(self, data: str, seq: int):
        object.__setattr__(self, "type", "audio")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "seq", seq)


@dataclass(frozen=True, slots=True)
class SequencedMessage(ProtocolMessage):
    seq: int


@dataclass(frozen=True, slots=True)
class StartUtteranceMessage(SequencedMessage):
    def __init__(self, seq: int):
        object.__setattr__(self, "type", "start_utterance")
        object.__setattr__(self, "seq", seq)


@dataclass(frozen=True, slots=True)
class EndUtteranceMessage(SequencedMessage):
    def __init__(self, seq: int):
        object.__setattr__(self, "type", "end_utterance")
        object.__setattr__(self, "seq", seq)


@dataclass(frozen=True, slots=True)
class CancelUtteranceMessage(SequencedMessage):
    def __init__(self, seq: int):
        object.__setattr__(self, "type", "cancel_utterance")
        object.__setattr__(self, "seq", seq)


@dataclass(frozen=True, slots=True)
class EndSessionMessage(SequencedMessage):
    def __init__(self, seq: int):
        object.__setattr__(self, "type", "end_session")
        object.__setattr__(self, "seq", seq)


@dataclass(frozen=True, slots=True)
class PingMessage(ProtocolMessage):
    ts: int

    def __init__(self, ts: int):
        object.__setattr__(self, "type", "ping")
        object.__setattr__(self, "ts", ts)


@dataclass(frozen=True, slots=True)
class StartedMessage(ProtocolMessage):
    session_id: str
    server_sample_rate: int
    server_channels: int

    def __init__(self, session_id: str, server_sample_rate: int = 48_000, server_channels: int = 1):
        object.__setattr__(self, "type", "started")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "server_sample_rate", server_sample_rate)
        object.__setattr__(self, "server_channels", server_channels)


@dataclass(frozen=True, slots=True)
class AudioReadyMessage(ProtocolMessage):
    seq: int

    def __init__(self, seq: int):
        object.__setattr__(self, "type", "audio_ready")
        object.__setattr__(self, "seq", seq)


@dataclass(frozen=True, slots=True)
class UtteranceEndedMessage(ProtocolMessage):
    seq: int
    cycles: int

    def __init__(self, seq: int, cycles: int):
        object.__setattr__(self, "type", "utterance_ended")
        object.__setattr__(self, "seq", seq)
        object.__setattr__(self, "cycles", cycles)


@dataclass(frozen=True, slots=True)
class EndedMessage(ProtocolMessage):
    session_id: str

    def __init__(self, session_id: str):
        object.__setattr__(self, "type", "ended")
        object.__setattr__(self, "session_id", session_id)


@dataclass(frozen=True, slots=True)
class PongMessage(ProtocolMessage):
    ts: int
    server_ts: int

    def __init__(self, ts: int, server_ts: int):
        object.__setattr__(self, "type", "pong")
        object.__setattr__(self, "ts", ts)
        object.__setattr__(self, "server_ts", server_ts)


@dataclass(frozen=True, slots=True)
class ErrorMessage(ProtocolMessage):
    code: str
    message: str
    fatal: bool = False

    def __init__(self, code: str, message: str, fatal: bool = False):
        object.__setattr__(self, "type", "error")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "fatal", fatal)


def _invalid(message: str, fatal: bool = False) -> ProtocolError:
    return ProtocolError("INVALID_FORMAT", message, fatal=fatal)


def _require_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid("Message must be a JSON object")
    return value


def _require_string(value: Any, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise _invalid(f"{field_name} must be a non-empty string")
    return value


def _require_sequence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _invalid("seq must be a non-negative integer")
    return value


def _require_sample_rate(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid("sample_rate must be an integer")
    if not MIN_INPUT_SAMPLE_RATE <= value <= MAX_INPUT_SAMPLE_RATE:
        raise _invalid(
            f"sample_rate must be between {MIN_INPUT_SAMPLE_RATE} and {MAX_INPUT_SAMPLE_RATE}"
        )
    if value not in PREFERRED_SAMPLE_RATES:
        logger.info("Using uncommon input sample rate: %s", value)
    return value


def _require_utterance_silence_ms(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid("utterance_silence_ms must be an integer")
    if value != 0 and not MIN_UTTERANCE_SILENCE_MS <= value <= MAX_UTTERANCE_SILENCE_MS:
        raise _invalid(
            f"utterance_silence_ms must be 0 or between "
            f"{MIN_UTTERANCE_SILENCE_MS} and {MAX_UTTERANCE_SILENCE_MS}"
        )
    return value


def _parse_text(raw: str) -> ProtocolMessage:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise _invalid("Message is not valid JSON") from exc
    payload = _require_dict(payload)
    message_type = payload.get("type")
    if not isinstance(message_type, str):
        raise _invalid("Message type is required")

    if message_type == "start":
        encoding = payload.get("audio_encoding")
        if encoding not in AUDIO_ENCODINGS:
            raise _invalid(f"audio_encoding must be one of {sorted(AUDIO_ENCODINGS)}")
        sample_rate = _require_sample_rate(payload.get("sample_rate"))
        channels = payload.get("channels")
        if channels not in SUPPORTED_CHANNELS:
            raise _invalid("channels must be 1 or 2")
        session_id = _require_string(payload.get("session_id"), "session_id")
        avatar_id = payload.get("avatar_id", "")
        if not isinstance(avatar_id, str):
            raise _invalid("avatar_id must be a string")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise _invalid("metadata must be an object")
        utterance_silence_ms = None
        if "utterance_silence_ms" in payload:
            utterance_silence_ms = _require_utterance_silence_ms(
                payload["utterance_silence_ms"]
            )
        return StartMessage(
            session_id,
            encoding,
            sample_rate,
            channels,
            avatar_id,
            metadata,
            utterance_silence_ms,
        )

    if message_type == "audio":
        data = _require_string(payload.get("data"), "data")
        _validate_base64(data)
        return AudioMessage(data, _require_sequence(payload.get("seq")))

    sequenced_types = {
        "start_utterance": StartUtteranceMessage,
        "end_utterance": EndUtteranceMessage,
        "cancel_utterance": CancelUtteranceMessage,
        "end_session": EndSessionMessage,
    }
    message_class = sequenced_types.get(message_type)
    if message_class is not None:
        return message_class(_require_sequence(payload.get("seq")))

    if message_type == "ping":
        ts = payload.get("ts")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            raise _invalid("ts must be a number")
        return PingMessage(int(ts))

    raise _invalid(f"Unsupported message type: {message_type}")


def _validate_base64(value: str) -> None:
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _invalid("audio data must be valid base64") from exc


def parse_message(raw: str | bytes) -> ProtocolMessage:
    if isinstance(raw, bytes):
        return BinaryAudioMessage(raw)
    if isinstance(raw, str):
        return _parse_text(raw)
    raise _invalid("WebSocket message must be text or binary")


def serialize_message(msg: ProtocolMessage) -> str | bytes:
    if isinstance(msg, BinaryAudioMessage):
        return msg.data
    if not isinstance(msg, ProtocolMessage):
        raise TypeError("msg must be a ProtocolMessage")
    # Optional fields (e.g. StartMessage.utterance_silence_ms) are omitted when
    # unset so the wire shape is unchanged for existing messages.
    payload = {key: value for key, value in asdict(msg).items() if value is not None}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def error_message(error: ProtocolError) -> ErrorMessage:
    return ErrorMessage(error.code, error.message, error.fatal)
