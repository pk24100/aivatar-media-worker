from dataclasses import FrozenInstanceError

import pytest

from streaming.transport.adapters.provider_profiles import (
    DEFAULT_PROFILE,
    DEFAULT_PROFILE_NAME,
    PROFILES,
    ProviderProfile,
    event_name,
    extract_audio_field,
    extract_audio_values,
    extract_json_path,
    extract_json_values,
    get_provider_profile,
    is_end_event,
    is_interruption_event,
    is_valid_provider_profile,
)


def test_profile_lookup_is_case_insensitive_and_supports_provider_aliases():
    assert set(PROFILES) == {"deepgram", "gemini"}
    assert get_provider_profile("DEEPGRAM") is PROFILES["deepgram"]
    assert get_provider_profile("gemini-live") is PROFILES["gemini"]
    assert get_provider_profile("deepgram voice agent") is PROFILES["deepgram"]


def test_lookup_without_provider_uses_the_declared_default_profile():
    assert DEFAULT_PROFILE_NAME == "deepgram"
    assert get_provider_profile() is DEFAULT_PROFILE
    assert get_provider_profile(default="gemini") is PROFILES["gemini"]


def test_profiles_have_expected_audio_defaults():
    deepgram = PROFILES["deepgram"]
    gemini = PROFILES["gemini"]

    assert (deepgram.sample_rate, deepgram.channels, deepgram.audio_encoding) == (16_000, 1, "pcm_s16le")
    assert (gemini.sample_rate, gemini.channels, gemini.audio_encoding) == (16_000, 1, "base64_json")
    assert deepgram.audio_field is None
    assert gemini.audio_field == "serverContent.modelTurn.parts.*.inlineData.data"


def test_profiles_expose_provider_event_names_and_match_nested_flags():
    assert PROFILES["deepgram"].end_events == frozenset({"AgentAudioDone"})
    assert PROFILES["deepgram"].interrupt_events == frozenset({"UserStartedSpeaking"})
    assert PROFILES["gemini"].end_events == frozenset({"turnComplete"})
    assert PROFILES["gemini"].interrupt_events == frozenset({"interrupted"})

    assert is_end_event({"type": "AgentAudioDone"}, "deepgram")
    assert is_interruption_event({"serverContent": {"interrupted": True}}, "gemini")
    assert event_name({"serverContent": {"turnComplete": True}}, "gemini") == "turnComplete"


def test_audio_fields_support_simple_and_wildcard_nested_paths():
    assert extract_audio_values({"binary": b"audio"}, "deepgram") == []

    gemini_message = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"inlineData": {"data": "chunk-1"}},
                    {"inlineData": {"data": "chunk-2"}},
                ]
            }
        }
    }
    assert extract_audio_values(gemini_message, "gemini") == ["chunk-1", "chunk-2"]
    assert extract_audio_field({}, "gemini", default="missing") == "missing"


def test_nested_json_path_helper_handles_indexes_pointers_and_missing_values():
    payload = {"outer": {"items": [{"value": 10}, {"value": 20}]}}

    assert extract_json_path(payload, "outer.items.1.value") == 20
    assert extract_json_path(payload, "/outer/items/0/value") == 10
    assert extract_json_values(payload, "outer.items.*.value") == [10, 20]
    assert extract_json_path(payload, "outer.missing", default="fallback") == "fallback"


def test_invalid_profile_values_raise_and_are_not_silently_defaulted():
    with pytest.raises(ValueError, match="unknown provider profile"):
        get_provider_profile("not-a-provider")
    with pytest.raises(TypeError, match="provider profile name must be a string"):
        get_provider_profile(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sample_rate must be between 8000 and 48000"):
        ProviderProfile("custom", sample_rate=0)
    with pytest.raises(ValueError, match="profile name"):
        ProviderProfile("", sample_rate=16_000)

    assert is_valid_provider_profile("deepgram") is True
    assert is_valid_provider_profile("not-a-provider") is False
    assert is_valid_provider_profile(None) is False


def test_provider_profile_accepts_and_logs_uncommon_in_range_sample_rate(caplog):
    with caplog.at_level("INFO", logger="streaming.transport.adapters.provider_profiles"):
        profile = ProviderProfile("custom", sample_rate=12_345)

    assert profile.sample_rate == 12_345
    assert "Using uncommon provider sample rate: 12345" in caplog.messages


def test_provider_profiles_are_frozen():
    with pytest.raises(FrozenInstanceError):
        PROFILES["deepgram"].sample_rate = 8_000  # type: ignore[misc]
