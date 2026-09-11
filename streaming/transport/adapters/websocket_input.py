"""Canonical WebSocket input adapter."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable

import numpy as np

from streaming.core.audio_bus import CanonicalAudioBus
from streaming.core.session import MediaInputAdapter
from streaming.transport.adapters.provider_profiles import (
    ProviderProfile,
    event_name,
    extract_audio_values,
    get_provider_profile,
    is_end_event,
    is_interruption_event,
)
from streaming.protocol.messages import (
    MAX_UTTERANCE_SILENCE_MS,
    MIN_UTTERANCE_SILENCE_MS,
    AudioMessage,
    BinaryAudioMessage,
    CancelUtteranceMessage,
    EndSessionMessage,
    EndUtteranceMessage,
    ErrorMessage,
    EndedMessage,
    PingMessage,
    PongMessage,
    ProtocolError,
    ProtocolMessage,
    StartMessage,
    StartUtteranceMessage,
    StartedMessage,
    parse_message,
)
from utils.errors import make_task_guard

logger = logging.getLogger(__name__)

DEFAULT_UTTERANCE_SILENCE_MS = 800
UTTERANCE_SILENCE_ENV_VAR = "FACEMODE_UTTERANCE_SILENCE_MS"

_CANONICAL_MESSAGE_TYPES = frozenset({
    "start_utterance",
    "end_utterance",
    "cancel_utterance",
    "end_session",
    "ping",
})


def _resolve_base64_inner_encoding(session_config: dict | None) -> str:
    """Return inner PCM encoding for base64_json payloads.

    Default preserves base64->s16le (e.g. gemini profile audio_encoding ==
    base64_json). Explicit inner f32le is opt-in only via metadata
    inner_encoding (e.g. {"inner_encoding": "pcm_f32le"}).
    """
    try:
        if not isinstance(session_config, dict):
            return "pcm_s16le"
        metadata = session_config.get("metadata")
        inner = None
        if isinstance(metadata, dict):
            inner = metadata.get("inner_encoding") or metadata.get("innerEncoding")
        if inner is None:
            inner = session_config.get("inner_encoding")
        if isinstance(inner, str):
            normalized = inner.strip().lower().replace("-", "_")
            if normalized in ("pcm_f32le", "f32le", "f32", "float32", "float_32le"):
                return "pcm_f32le"
    except Exception:
        pass
    return "pcm_s16le"


def _env_utterance_silence_ms() -> int:
    """Read the process-wide silence gap default.

    Invalid or out-of-range operator values fall back to the default so the
    inferred boundary always lands well below the engine idle-removal cutoff.
    """
    raw = os.environ.get(UTTERANCE_SILENCE_ENV_VAR)
    if raw is None:
        return DEFAULT_UTTERANCE_SILENCE_MS
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_UTTERANCE_SILENCE_MS
    if value == 0 or MIN_UTTERANCE_SILENCE_MS <= value <= MAX_UTTERANCE_SILENCE_MS:
        return value
    return DEFAULT_UTTERANCE_SILENCE_MS


class WebsocketInputAdapter(MediaInputAdapter):
    def __init__(
        self,
        bus: CanonicalAudioBus,
        *,
        expected_session_id: str | None = None,
        on_end_session: Callable[[], Awaitable[None] | None] | None = None,
        provider_profile: str | ProviderProfile | None = None,
    ):
        self.bus = bus
        self.expected_session_id = expected_session_id
        self.on_end_session = on_end_session
        self.provider_profile = (
            get_provider_profile(provider_profile) if provider_profile is not None else None
        )
        self.session_config: dict = {}
        self._connected = False
        self._negotiated = False
        self._utterance_active = False
        self.last_protocol_sequence = -1
        # Utterance boundary control: "inferred" lets the adapter derive
        # boundaries from a silence gap; "explicit" means the client drives
        # start_utterance/end_utterance/cancel_utterance itself. A provider
        # profile defining its own utterance events (deepgram, gemini) never
        # infers. A single explicit client utterance message latches the
        # session to "explicit" permanently (including across reconnects on
        # this adapter) so plugins see zero behavior change.
        self._default_utterance_mode = (
            "explicit"
            if self.provider_profile is not None
            and (
                self.provider_profile.start_utterance_event
                or self.provider_profile.end_utterance_event
                or self.provider_profile.cancel_utterance_event
            )
            else "inferred"
        )
        self._utterance_mode = self._default_utterance_mode
        self._utterance_silence_ms = _env_utterance_silence_ms()
        self._silence_timer_task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_negotiated(self) -> bool:
        return self._negotiated

    async def connect(self, session_config: dict) -> None:
        await self._cancel_silence_timer()
        self.session_config = dict(session_config)
        if self.provider_profile is not None:
            self.session_config.setdefault("audio_encoding", self.provider_profile.audio_encoding)
            self.session_config.setdefault("sample_rate", self.provider_profile.sample_rate)
            self.session_config.setdefault("channels", self.provider_profile.channels)
            self.session_config.setdefault(
                "session_id",
                self.expected_session_id or self.session_config.get("session_id", ""),
            )
        self.last_protocol_sequence = -1
        self._utterance_active = False
        self._utterance_silence_ms = _env_utterance_silence_ms()
        silence_override = self.session_config.pop("utterance_silence_ms", None)
        if silence_override is not None:
            self._apply_utterance_silence_ms(silence_override)
        await self.bus.connect(self.session_config)
        self._connected = True
        self._negotiated = True

    async def disconnect(self) -> None:
        await self._cancel_silence_timer()
        self._connected = False
        self._negotiated = False

    async def start_utterance(self) -> None:
        self._require_negotiated()
        await self.bus.start_utterance()
        self._utterance_active = True

    async def push_audio(self, pcm: np.ndarray, sample_rate: int, channels: int) -> None:
        self._require_negotiated()
        await self.bus.push_audio(pcm, sample_rate, channels)
        self._utterance_active = True

    async def end_utterance(self) -> None:
        self._require_negotiated()
        await self.bus.end_utterance(self._boundary_sequence())
        self._utterance_active = False

    async def cancel_utterance(self) -> None:
        self._require_negotiated()
        await self.bus.cancel_utterance(self._boundary_sequence())
        self._utterance_active = False

    async def on_interrupted(self) -> None:
        await self.cancel_utterance()

    async def handle_message(self, raw: str | bytes) -> list[ProtocolMessage]:
        try:
            if self.provider_profile is not None:
                return await self._handle_provider_message(raw)
            message = parse_message(raw)
            return await self._handle_parsed_message(message)
        except ProtocolError as error:
            if error.fatal:
                await self._cancel_silence_timer()
            return [ErrorMessage(error.code, error.message, error.fatal)]
        except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
            return [ErrorMessage("INVALID_FORMAT", str(error), False)]

    async def _handle_provider_message(self, raw: str | bytes) -> list[ProtocolMessage]:
        profile = self.provider_profile
        if profile is None:
            raise RuntimeError("provider profile is not configured")

        responses: list[ProtocolMessage] = []
        if not self._negotiated:
            await self.connect({
                "audio_encoding": profile.audio_encoding,
                "sample_rate": profile.sample_rate,
                "channels": profile.channels,
                "session_id": self.expected_session_id or "",
                "metadata": {"provider_profile": profile.name},
            })
            responses.append(
                StartedMessage(
                    self.session_config.get("session_id", self.expected_session_id or ""),
                    self.bus.canonical_sample_rate,
                    self.bus.canonical_channels,
                )
            )

        if isinstance(raw, bytes):
            await self._handle_provider_binary(raw)
            return responses
        if not isinstance(raw, str):
            raise ProtocolError("INVALID_FORMAT", "Provider WebSocket message must be text or binary")

        text = raw.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            raise ProtocolError("INVALID_FORMAT", "Provider text message must be valid JSON")

        if isinstance(payload, dict) and payload.get("type") in _CANONICAL_MESSAGE_TYPES:
            message = parse_message(raw)
            canonical_responses = await self._handle_parsed_message(message)
            responses.extend(canonical_responses)
            return responses

        provider_event = event_name(payload, profile)
        if is_interruption_event(payload, profile):
            await self.cancel_utterance()
        if profile.start_utterance_event and provider_event == profile.start_utterance_event:
            await self.start_utterance()
        audio_values = extract_audio_values(payload, profile)
        for value in audio_values:
            if not isinstance(value, str):
                continue
            await self._push_provider_base64(value)
        if is_end_event(payload, profile):
            await self.end_utterance()
        if provider_event in {"Settings", "settings"}:
            self._apply_provider_settings(payload)
        return responses

    async def _handle_provider_binary(self, data: bytes) -> None:
        profile = self.provider_profile
        if profile is None:
            raise RuntimeError("provider profile is not configured")
        payload = _strip_wav_header(data) if profile.strip_wav_header else data
        encoding = profile.audio_encoding
        channels = int(self.session_config.get("channels", profile.channels))
        sample_rate = int(self.session_config.get("sample_rate", profile.sample_rate))
        if encoding == "pcm_s16le":
            alignment = 2 * channels
            if len(payload) % alignment:
                raise ProtocolError("INVALID_FORMAT", "provider PCM payload is not channel aligned")
            pcm = np.frombuffer(payload, dtype="<i2")
        elif encoding == "pcm_f32le":
            alignment = 4 * channels
            if len(payload) % alignment:
                raise ProtocolError("INVALID_FORMAT", "provider float PCM payload is not channel aligned")
            pcm = np.frombuffer(payload, dtype="<f4")
        else:
            raise ProtocolError("INVALID_FORMAT", f"Unsupported provider encoding: {encoding}")
        if self._inference_enabled():
            await self._begin_inferred_utterance()
        await self.push_audio(pcm, sample_rate, channels)
        await self._rearm_silence_timer()

    async def _push_provider_base64(self, value: str) -> None:
        try:
            data = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ProtocolError("INVALID_FORMAT", "provider audio data must be valid base64") from error
        profile = self.provider_profile
        if profile is None:
            raise RuntimeError("provider profile is not configured")
        payload = _strip_wav_header(data) if profile.strip_wav_header else data
        channels = int(self.session_config.get("channels", profile.channels))
        sample_rate = int(self.session_config.get("sample_rate", profile.sample_rate))
        # Default preserves base64->s16le for gemini (audio_encoding==base64_json).
        # Explicit inner f32le only via metadata inner_encoding opt-in, with
        # alignment check mirroring provider base64 2*ch (4*ch for f32).
        inner = _resolve_base64_inner_encoding(self.session_config)
        if inner == "pcm_f32le":
            alignment = 4 * channels
            if len(payload) % alignment:
                raise ProtocolError("INVALID_FORMAT", "provider base64 PCM payload is not channel aligned")
            pcm = np.frombuffer(payload, dtype="<f4")
        else:
            alignment = 2 * channels
            if len(payload) % alignment:
                raise ProtocolError("INVALID_FORMAT", "provider base64 PCM payload is not channel aligned")
            pcm = np.frombuffer(payload, dtype="<i2")
        if self._inference_enabled():
            await self._begin_inferred_utterance()
        await self.push_audio(pcm, sample_rate, channels)
        await self._rearm_silence_timer()

    def _apply_provider_settings(self, payload: dict) -> None:
        settings = payload.get("audio") if isinstance(payload.get("audio"), dict) else payload
        sample_rate = settings.get("sample_rate") or settings.get("sampleRate")
        channels = settings.get("channels")
        if isinstance(sample_rate, int) and sample_rate > 0:
            self.session_config["sample_rate"] = sample_rate
        if isinstance(channels, int) and channels in (1, 2):
            self.session_config["channels"] = channels

    async def _handle_parsed_message(self, message: ProtocolMessage) -> list[ProtocolMessage]:
        if not self._negotiated:
            if not isinstance(message, StartMessage):
                raise ProtocolError("NOT_NEGOTIATED", "The first message must be start", fatal=False)
            if self.expected_session_id and message.session_id != self.expected_session_id:
                raise ProtocolError("SESSION_NOT_FOUND", "session_id does not match the WebSocket session", fatal=True)
            session_config = {
                "audio_encoding": message.audio_encoding,
                "sample_rate": message.sample_rate,
                "channels": message.channels,
                "avatar_id": message.avatar_id,
                "session_id": message.session_id,
                "metadata": message.metadata,
            }
            if message.utterance_silence_ms is not None:
                session_config["utterance_silence_ms"] = message.utterance_silence_ms
            await self.connect(session_config)
            return [StartedMessage(message.session_id, self.bus.canonical_sample_rate, self.bus.canonical_channels)]

        if isinstance(message, BinaryAudioMessage):
            await self._handle_binary_audio(message.data)
            return []
        if isinstance(message, AudioMessage):
            await self._handle_base64_audio(message)
            return []
        if isinstance(message, StartUtteranceMessage):
            await self._latch_explicit_mode()
            self._remember_sequence(message.seq)
            await self.start_utterance()
            return []
        if isinstance(message, EndUtteranceMessage):
            await self._latch_explicit_mode()
            self._remember_sequence(message.seq)
            await self.end_utterance()
            return []
        if isinstance(message, CancelUtteranceMessage):
            await self._latch_explicit_mode()
            self._remember_sequence(message.seq)
            await self.cancel_utterance()
            return []
        if isinstance(message, EndSessionMessage):
            self._remember_sequence(message.seq)
            await self._cancel_silence_timer()
            await self.bus.end_session(self._boundary_sequence())
            self._utterance_active = False
            self._connected = False
            self._negotiated = False
            if self.on_end_session is not None:
                result = self.on_end_session()
                if result is not None:
                    await result
            return [EndedMessage(self.session_config.get("session_id", self.expected_session_id or ""))]
        if isinstance(message, PingMessage):
            return [PongMessage(message.ts, int(time.time() * 1000))]
        if isinstance(message, StartMessage):
            raise ProtocolError("INVALID_FORMAT", "start may only be the first message")
        raise ProtocolError("INVALID_FORMAT", f"Unsupported message type: {message.type}")

    async def _handle_binary_audio(self, data: bytes) -> None:
        encoding = self.session_config.get("audio_encoding")
        if encoding == "base64_json":
            raise ProtocolError("INVALID_FORMAT", "base64_json sessions must send audio JSON messages")
        sample_rate = int(self.session_config["sample_rate"])
        channels = int(self.session_config["channels"])
        if encoding == "pcm_s16le":
            alignment = 2 * channels
            if len(data) % alignment:
                raise ProtocolError("INVALID_FORMAT", "pcm_s16le payload is not channel aligned")
            pcm = np.frombuffer(data, dtype="<i2")
        elif encoding == "pcm_f32le":
            alignment = 4 * channels
            if len(data) % alignment:
                raise ProtocolError("INVALID_FORMAT", "pcm_f32le payload is not channel aligned")
            pcm = np.frombuffer(data, dtype="<f4")
        else:
            raise ProtocolError("INVALID_FORMAT", f"Unsupported negotiated encoding: {encoding}")
        if self._inference_enabled():
            await self._begin_inferred_utterance()
        await self.push_audio(pcm, sample_rate, channels)
        await self._rearm_silence_timer()

    async def _handle_base64_audio(self, message: AudioMessage) -> None:
        if self.session_config.get("audio_encoding") != "base64_json":
            raise ProtocolError("INVALID_FORMAT", "audio JSON is only valid for base64_json sessions")
        try:
            data = base64.b64decode(message.data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ProtocolError("INVALID_FORMAT", "audio data must be valid base64") from error
        self._remember_sequence(message.seq)
        channels = int(self.session_config["channels"])
        inner = _resolve_base64_inner_encoding(self.session_config)
        if inner == "pcm_f32le":
            alignment = 4 * channels
            if len(data) % alignment:
                raise ProtocolError("INVALID_FORMAT", "base64 PCM payload is not channel aligned")
            pcm = np.frombuffer(data, dtype="<f4")
        else:
            alignment = 2 * channels
            if len(data) % alignment:
                raise ProtocolError("INVALID_FORMAT", "base64 PCM payload is not channel aligned")
            pcm = np.frombuffer(data, dtype="<i2")
        if self._inference_enabled():
            await self._begin_inferred_utterance()
        await self.bus.push_audio(
            pcm,
            int(self.session_config["sample_rate"]),
            channels,
            sequence_number=message.seq,
        )
        self._utterance_active = True
        await self._rearm_silence_timer()

    def _require_negotiated(self) -> None:
        if not self._negotiated:
            raise ProtocolError("NOT_NEGOTIATED", "Send start before media or lifecycle messages")

    def _remember_sequence(self, sequence: int) -> None:
        if sequence < self.last_protocol_sequence:
            raise ProtocolError("INVALID_FORMAT", "seq must not move backwards")
        self.last_protocol_sequence = sequence

    def _boundary_sequence(self) -> int | None:
        """Protocol sequence for a boundary call, or None.

        ``last_protocol_sequence`` stays ``-1`` when no client sequenced
        message was seen; the audio bus rejects negative sequences, so
        provider-driven and sequence-less boundaries pass ``None`` and let
        the bus assign the frame sequence. Validated client sequences are
        preserved unchanged.
        """
        if self.last_protocol_sequence >= 0:
            return self.last_protocol_sequence
        return None

    def _apply_utterance_silence_ms(self, value) -> None:
        """Apply a per-session silence override (already parsed as a value).

        Mirrors the wire validation so direct ``connect()`` callers cannot
        sneak an out-of-range gap past the ``start`` message checks.
        """
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or (value != 0 and not MIN_UTTERANCE_SILENCE_MS <= value <= MAX_UTTERANCE_SILENCE_MS)
        ):
            raise ProtocolError(
                "INVALID_FORMAT",
                "utterance_silence_ms must be 0 or an integer between "
                f"{MIN_UTTERANCE_SILENCE_MS} and {MAX_UTTERANCE_SILENCE_MS}",
            )
        self._utterance_silence_ms = value

    def _inference_enabled(self) -> bool:
        return self._utterance_mode == "inferred" and self._utterance_silence_ms > 0

    async def _begin_inferred_utterance(self) -> None:
        """Implicitly open an utterance on the first frame of a gap."""
        if not self._utterance_active:
            await self.start_utterance()

    async def _latch_explicit_mode(self) -> None:
        """Permanently latch this session to client-driven utterance control."""
        self._utterance_mode = "explicit"
        await self._cancel_silence_timer()

    async def _rearm_silence_timer(self) -> None:
        """Cancel any pending silence timer and arm a fresh one."""
        await self._cancel_silence_timer()
        if (
            not self._inference_enabled()
            or not self._negotiated
            or not self._connected
            or not self._utterance_active
        ):
            return
        task = asyncio.create_task(
            self._silence_timer_run(),
            name=f"utterance-silence-{self.session_config.get('session_id', '')}",
        )
        task.add_done_callback(
            make_task_guard(
                logger,
                self.session_config.get("session_id", "-"),
                "utterance_silence_timer",
            )
        )
        self._silence_timer_task = task

    async def _cancel_silence_timer(self) -> None:
        """Cancel the pending silence timer and await its teardown.

        Awaiting the cancelled task guarantees no task is left pending when
        the adapter disconnects or the session ends.
        """
        task = self._silence_timer_task
        self._silence_timer_task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _wait_silence_delay(self, delay_s: float) -> None:
        """Wait out the silence gap; isolated so tests can drive it manually."""
        await asyncio.sleep(delay_s)

    async def _silence_timer_run(self) -> None:
        try:
            await self._wait_silence_delay(self._utterance_silence_ms / 1000.0)
            await self._finish_inferred_utterance()
        finally:
            if self._silence_timer_task is asyncio.current_task():
                self._silence_timer_task = None

    async def _finish_inferred_utterance(self) -> None:
        """Emit an inferred boundary through the normal end_utterance path.

        The inferred boundary carries no client sequence: ``None`` lets the
        bus assign the frame sequence, which the downstream drain path then
        echoes in ``UtteranceEndedMessage``.
        """
        if (
            self._utterance_mode != "inferred"
            or not self._utterance_active
            or not self._connected
            or not self._negotiated
        ):
            return
        self._utterance_active = False
        await self.bus.end_utterance(None)


def _strip_wav_header(data: bytes) -> bytes:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    marker = data.find(b"data", 12)
    if marker < 0 or marker + 8 > len(data):
        return data
    payload_start = marker + 8
    payload_size = int.from_bytes(data[marker + 4 : payload_start], "little")
    return data[payload_start : payload_start + payload_size]
