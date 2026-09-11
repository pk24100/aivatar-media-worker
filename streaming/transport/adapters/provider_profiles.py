"""Provider-specific audio and event profiles for streaming adapters.

The profiles describe the audio produced by a provider, not the canonical bus
format.  Provider adapters can use the sample rate, channel count, encoding,
and event names to build a normal WebsocketInputAdapter session without
embedding provider-specific literals in their receive loops.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from streaming.protocol.messages import (
    MAX_INPUT_SAMPLE_RATE,
    MIN_INPUT_SAMPLE_RATE,
    PREFERRED_SAMPLE_RATES,
)


logger = logging.getLogger(__name__)

PathPart: TypeAlias = str | int
JsonPath: TypeAlias = str | Sequence[PathPart] | None

_MISSING = object()
_WILDCARD = "*"
_SUPPORTED_CHANNELS = frozenset({1, 2})


def _normalize_event_names(value: Any, field_name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise TypeError(f"{field_name} must contain strings") from exc

    normalized: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{field_name} must contain non-empty strings")
        normalized.add(item)
    return frozenset(normalized)


def _normalize_profile_path(value: JsonPath, field_name: str) -> JsonPath:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        parts = tuple(value)
        for part in parts:
            if isinstance(part, bool) or not isinstance(part, (str, int)):
                raise TypeError(f"{field_name} must contain only strings or integers")
        return parts
    raise TypeError(f"{field_name} must be a dotted string or a path sequence")


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Immutable description of a provider's audio WebSocket contract.

    ``audio_field`` is a dotted JSON path when a provider carries audio in a
    JSON message.  The path may contain ``*`` to select every item in a list.
    ``None`` means that audio is carried as a binary WebSocket message.

    ``sample_rate`` and ``channels`` are safe defaults for a provider output
    stream.  A provider adapter should prefer values negotiated in a session
    when the provider exposes them.
    """

    name: str
    sample_rate: int
    channels: int = 1
    audio_encoding: str = "pcm_s16le"
    audio_field: JsonPath = None
    event_field: JsonPath = "type"
    audio_events: frozenset[str] = frozenset()
    end_events: frozenset[str] = frozenset()
    interrupt_events: frozenset[str] = frozenset()
    channel_extraction: str | None = None
    start_utterance_event: str | None = None
    end_utterance_event: str | None = None
    cancel_utterance_event: str | None = None
    strip_wav_header: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile name must be a non-empty string")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise ValueError("sample_rate must be an integer")
        if not MIN_INPUT_SAMPLE_RATE <= self.sample_rate <= MAX_INPUT_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate must be between {MIN_INPUT_SAMPLE_RATE} and {MAX_INPUT_SAMPLE_RATE}"
            )
        if self.sample_rate not in PREFERRED_SAMPLE_RATES:
            logger.info("Using uncommon provider sample rate: %s", self.sample_rate)
        if isinstance(self.channels, bool) or not isinstance(self.channels, int):
            raise ValueError("channels must be an integer")
        if self.channels not in _SUPPORTED_CHANNELS:
            raise ValueError("channels must be 1 or 2")
        if not isinstance(self.audio_encoding, str) or not self.audio_encoding.strip():
            raise ValueError("audio_encoding must be a non-empty string")

        object.__setattr__(self, "name", self.name.strip().lower())
        object.__setattr__(self, "audio_encoding", self.audio_encoding.strip())
        object.__setattr__(self, "audio_field", _normalize_profile_path(self.audio_field, "audio_field"))
        object.__setattr__(self, "event_field", _normalize_profile_path(self.event_field, "event_field"))
        object.__setattr__(self, "audio_events", _normalize_event_names(self.audio_events, "audio_events"))
        object.__setattr__(self, "end_events", _normalize_event_names(self.end_events, "end_events"))
        object.__setattr__(
            self,
            "interrupt_events",
            _normalize_event_names(self.interrupt_events, "interrupt_events"),
        )

    @property
    def provider(self) -> str:
        """Return the provider identifier used by lookup helpers."""

        return self.name

    @property
    def encoding(self) -> str:
        """Short alias used by adapters that call the field ``encoding``."""

        return self.audio_encoding

    @property
    def audio_format(self) -> str:
        """Alias matching the public Phase 2 profile contract."""

        return self.audio_encoding

    @property
    def default_sample_rate(self) -> int:
        """Return the configured fallback sample rate."""

        return self.sample_rate

    @property
    def default_channels(self) -> int:
        """Return the configured fallback channel count."""

        return self.channels

    @property
    def audio_path(self) -> JsonPath:
        """Return the JSON path that contains audio payloads."""

        return self.audio_field

    @property
    def event_path(self) -> JsonPath:
        """Return the JSON path used for a provider event name."""

        return self.event_field

    @property
    def audio_event_names(self) -> frozenset[str]:
        return self.audio_events

    @property
    def end_event_names(self) -> frozenset[str]:
        return self.end_events

    @property
    def turn_end_events(self) -> frozenset[str]:
        return self.end_events

    @property
    def interruption_events(self) -> frozenset[str]:
        return self.interrupt_events

    @property
    def interruption_event_names(self) -> frozenset[str]:
        return self.interrupt_events

    @property
    def all_event_names(self) -> frozenset[str]:
        return self.audio_events | self.end_events | self.interrupt_events

    @property
    def is_json_audio(self) -> bool:
        return self.audio_field is not None


# The first profile is the default for callers that do not have provider
# metadata yet.  Adapters should still use an explicit provider when possible.
DEFAULT_PROFILE_NAME = "deepgram"


PROFILES: dict[str, ProviderProfile] = {
    "deepgram": ProviderProfile(
        name="deepgram",
        sample_rate=16_000,
        channels=1,
        audio_encoding="pcm_s16le",
        audio_field=None,
        event_field="type",
        audio_events=frozenset(),
        end_events=frozenset({"AgentAudioDone"}),
        interrupt_events=frozenset({"UserStartedSpeaking"}),
        start_utterance_event="UserStartedSpeaking",
        end_utterance_event="AgentAudioDone",
        cancel_utterance_event="UserStartedSpeaking",
    ),
    "gemini": ProviderProfile(
        name="gemini",
        sample_rate=16_000,
        channels=1,
        audio_encoding="base64_json",
        audio_field="serverContent.modelTurn.parts.*.inlineData.data",
        event_field=None,
        audio_events=frozenset(),
        end_events=frozenset({"turnComplete"}),
        interrupt_events=frozenset({"interrupted"}),
        end_utterance_event="turnComplete",
        cancel_utterance_event="interrupted",
    ),
}

DEFAULT_PROFILE = PROFILES[DEFAULT_PROFILE_NAME]

_PROFILE_ALIASES = {
    "deepgram_voice_agent": "deepgram",
    "deepgram_voiceagent": "deepgram",
    "gemini_live": "gemini",
    "gemini_live_api": "gemini",
}


def _normalize_provider_name(provider: str) -> str:
    if not isinstance(provider, str):
        raise TypeError("provider profile name must be a string")
    normalized = provider.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("provider profile name must be non-empty")
    return _PROFILE_ALIASES.get(normalized, normalized)


def get_provider_profile(
    provider: str | ProviderProfile | None = None,
    *,
    default: str | ProviderProfile | None = None,
) -> ProviderProfile:
    """Look up a provider profile.

    Lookup is case-insensitive and accepts the common provider aliases.  With
    no provider name, the explicit ``default`` is used when supplied; otherwise
    :data:`DEFAULT_PROFILE` is returned.  Unknown names raise ``ValueError``
    instead of silently falling back to a different provider contract.
    """

    if provider is None:
        provider = DEFAULT_PROFILE if default is None else default
    if isinstance(provider, ProviderProfile):
        return provider
    if not isinstance(provider, str):
        raise TypeError("provider profile name must be a string or ProviderProfile")

    key = _normalize_provider_name(provider)
    try:
        return PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown provider profile {provider!r}; expected one of: {supported}") from exc


def lookup_provider_profile(provider: str | ProviderProfile | None = None) -> ProviderProfile:
    """Compatibility alias for :func:`get_provider_profile`."""

    return get_provider_profile(provider)


def get_profile(provider: str | ProviderProfile | None = None) -> ProviderProfile:
    """Short compatibility alias for :func:`get_provider_profile`."""

    return get_provider_profile(provider)


def validate_provider_profile(
    profile: str | ProviderProfile | None,
) -> ProviderProfile:
    """Resolve and validate a profile for adapter use.

    The dataclass validates custom profile instances at construction time.  A
    string is additionally checked against the supported profile registry.
    Invalid names raise ``ValueError`` and invalid input types raise
    ``TypeError``.
    """

    return get_provider_profile(profile)


def validate_profile(profile: str | ProviderProfile | None) -> ProviderProfile:
    """Compatibility alias for :func:`validate_provider_profile`."""

    return validate_provider_profile(profile)


def is_valid_provider_profile(profile: str | ProviderProfile | None) -> bool:
    """Return whether a profile value can be used by an adapter."""

    if profile is None:
        return False
    try:
        validate_provider_profile(profile)
    except (TypeError, ValueError):
        return False
    return True


def normalize_json_path(path: JsonPath) -> tuple[PathPart, ...]:
    """Convert a dotted, JSON Pointer, or sequence path to path parts.

    Dotted paths use ``*`` as a wildcard for every member of a mapping or
    every item in a list.  JSON Pointer paths use the usual ``~0`` and ``~1``
    escapes.  An empty path addresses the input object itself.
    """

    if path is None:
        return ()
    if isinstance(path, str):
        value = path.strip()
        if not value or value == "$":
            return ()
        if value.startswith("/"):
            parts: tuple[PathPart, ...] = tuple(
                part.replace("~1", "/").replace("~0", "~") for part in value[1:].split("/")
            )
        else:
            if value.startswith("$."):
                value = value[2:]
            elif value.startswith("$"):
                value = value[1:].lstrip(".")
            parts = tuple(value.split(".")) if value else ()
    elif isinstance(path, Sequence) and not isinstance(path, (bytes, bytearray, str)):
        parts = tuple(path)
    else:
        raise TypeError("JSON path must be a dotted string or a path sequence")

    normalized: list[PathPart] = []
    for part in parts:
        if isinstance(part, bool) or not isinstance(part, (str, int)):
            raise TypeError("JSON path parts must be strings or integers")
        if isinstance(part, str) and part.isdecimal():
            normalized.append(int(part))
        else:
            normalized.append(part)
    return tuple(normalized)


def _walk_json(value: Any, path: tuple[PathPart, ...]):
    if not path:
        yield value
        return

    part = path[0]
    remainder = path[1:]
    if part == _WILDCARD:
        if isinstance(value, Mapping):
            for child in value.values():
                yield from _walk_json(child, remainder)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                yield from _walk_json(child, remainder)
        return

    if isinstance(value, Mapping):
        if part in value:
            yield from _walk_json(value[part], remainder)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if isinstance(part, int) and 0 <= part < len(value):
            yield from _walk_json(value[part], remainder)


def extract_json_values(payload: Any, path: JsonPath) -> list[Any]:
    """Return every value selected by a nested JSON path.

    Missing keys, wrong container types, and out-of-range list indexes return
    an empty list.  The helper never raises for malformed provider payloads,
    which lets a WebSocket adapter report a provider-format error itself.
    """

    parts = normalize_json_path(path)
    return list(_walk_json(payload, parts))


def extract_json_path(payload: Any, path: JsonPath, default: Any = None) -> Any:
    """Return the first value selected by a nested JSON path."""

    values = extract_json_values(payload, path)
    return values[0] if values else default


def get_nested_value(payload: Any, path: JsonPath, default: Any = None) -> Any:
    """Compatibility alias for :func:`extract_json_path`."""

    return extract_json_path(payload, path, default)


def extract_nested(payload: Any, path: JsonPath, default: Any = None) -> Any:
    """Compatibility alias for :func:`extract_json_path`."""

    return extract_json_path(payload, path, default)


def extract_audio_values(
    payload: Any,
    profile: str | ProviderProfile | None = None,
) -> list[Any]:
    """Extract all JSON audio fields described by a provider profile."""

    selected = validate_provider_profile(profile)
    if selected.audio_field is None:
        return []
    return extract_json_values(payload, selected.audio_field)


def extract_audio_field(
    payload: Any,
    profile: str | ProviderProfile | None = None,
    default: Any = None,
) -> Any:
    """Extract the first JSON audio field described by a provider profile."""

    values = extract_audio_values(payload, profile)
    return values[0] if values else default


def extract_audio_data(
    payload: Any,
    profile: str | ProviderProfile | None = None,
    default: Any = None,
) -> Any:
    """Compatibility alias for :func:`extract_audio_field`."""

    return extract_audio_field(payload, profile, default)


def _iter_key_value_pairs(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield key, child
            yield from _iter_key_value_pairs(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _iter_key_value_pairs(child)


def _contains_event_name(payload: Any, event_name: str) -> bool:
    if payload == event_name:
        return True
    for key, value in _iter_key_value_pairs(payload):
        # Nested provider flags, such as Gemini's
        # {"serverContent": {"turnComplete": true}}, use the key as the
        # event name.  A false flag is not an emitted event.
        if key == event_name and value not in (False, None):
            return True
        if value == event_name:
            return True
    return False


def event_name(payload: Any, profile: str | ProviderProfile | None = None) -> str | None:
    """Return the provider event name from a JSON payload when available."""

    selected = validate_provider_profile(profile)
    if selected.event_field is not None:
        value = extract_json_path(payload, selected.event_field, _MISSING)
        if isinstance(value, str):
            return value

    for candidate in selected.all_event_names:
        if _contains_event_name(payload, candidate):
            return candidate
    if isinstance(payload, str):
        return payload
    return None


def event_matches(
    payload: Any,
    profile: str | ProviderProfile | None,
    event_names: Sequence[str] | str,
) -> bool:
    """Return whether a payload contains one of the requested event names."""

    if isinstance(event_names, str):
        names = (event_names,)
    else:
        names = tuple(event_names)
    return any(_contains_event_name(payload, name) for name in names)


def is_audio_event(payload: Any, profile: str | ProviderProfile | None = None) -> bool:
    """Return whether a payload contains profile-defined JSON audio."""

    selected = validate_provider_profile(profile)
    if selected.audio_field is not None and extract_audio_values(payload, selected):
        return True
    return event_matches(payload, selected, selected.audio_events)


def is_end_event(payload: Any, profile: str | ProviderProfile | None = None) -> bool:
    """Return whether a payload marks the end of a provider turn."""

    selected = validate_provider_profile(profile)
    return event_matches(payload, selected, selected.end_events)


def is_interruption_event(
    payload: Any,
    profile: str | ProviderProfile | None = None,
) -> bool:
    """Return whether a payload marks provider-side interruption."""

    selected = validate_provider_profile(profile)
    return event_matches(payload, selected, selected.interrupt_events)


__all__ = [
    "DEFAULT_PROFILE",
    "DEFAULT_PROFILE_NAME",
    "JsonPath",
    "PROFILES",
    "PathPart",
    "ProviderProfile",
    "event_matches",
    "event_name",
    "extract_audio_data",
    "extract_audio_field",
    "extract_audio_values",
    "extract_json_path",
    "extract_json_values",
    "extract_nested",
    "get_nested_value",
    "get_profile",
    "get_provider_profile",
    "is_audio_event",
    "is_end_event",
    "is_interruption_event",
    "is_valid_provider_profile",
    "lookup_provider_profile",
    "normalize_json_path",
    "validate_profile",
    "validate_provider_profile",
]
