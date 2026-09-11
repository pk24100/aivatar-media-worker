import asyncio
import base64
import json
from unittest.mock import AsyncMock, Mock

import numpy as np

from streaming.transport.adapters.websocket_input import WebsocketInputAdapter
from streaming.core.audio_bus import CanonicalAudioBus
from streaming.protocol.messages import StartedMessage


def run(coroutine):
    return asyncio.run(coroutine)


def make_bus():
    bus = Mock(spec=CanonicalAudioBus)
    bus.canonical_sample_rate = 48_000
    bus.canonical_channels = 1
    bus.connect = AsyncMock()
    bus.start_utterance = AsyncMock()
    bus.push_audio = AsyncMock()
    bus.end_utterance = AsyncMock()
    bus.cancel_utterance = AsyncMock()
    bus.end_session = AsyncMock()
    return bus


def test_deepgram_profile_accepts_binary_pcm():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus, expected_session_id="deepgram-bin-1", provider_profile="deepgram")
    pcm = np.array([1, -2, 3], dtype="<i2")

    started = run(adapter.handle_message(pcm.tobytes()))

    assert isinstance(started[0], StartedMessage)
    push_args = bus.push_audio.await_args.args
    np.testing.assert_array_equal(push_args[0], pcm)
    assert push_args[1:] == (16_000, 1)


def test_deepgram_profile_maps_settings_turn_start_and_end():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus, expected_session_id="deepgram-1", provider_profile="deepgram")
    run(adapter.handle_message(json.dumps({"type": "Settings", "audio": {"sampleRate": 24_000}})))
    run(adapter.handle_message(json.dumps({"type": "UserStartedSpeaking"})))
    run(adapter.handle_message(np.zeros(4, dtype="<i2").tobytes()))
    run(adapter.handle_message(json.dumps({"type": "AgentAudioDone"})))

    assert adapter.session_config["sample_rate"] == 24_000
    bus.cancel_utterance.assert_awaited_once()
    bus.start_utterance.assert_awaited_once()
    bus.end_utterance.assert_awaited_once()


def test_gemini_profile_extracts_nested_base64_audio_and_turn_events():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus, expected_session_id="gemini-1", provider_profile="gemini")
    pcm = np.array([100, -100], dtype="<i2")
    payload = {
        "serverContent": {
            "modelTurn": {"parts": [{"inlineData": {"data": base64.b64encode(pcm).decode()}}]},
            "turnComplete": True,
            "interrupted": True,
        }
    }

    response = run(adapter.handle_message(json.dumps(payload)))

    assert isinstance(response[0], StartedMessage)
    push_args = bus.push_audio.await_args.args
    np.testing.assert_array_equal(push_args[0], pcm)
    assert push_args[1:] == (16_000, 1)
    # Provider-driven boundaries carry no client sequence; the bus assigns it.
    bus.cancel_utterance.assert_awaited_once_with(None)
    bus.end_utterance.assert_awaited_once_with(None)


def test_gemini_profile_default_remains_s16le_without_opt_in():
    # Default preserves base64->s16le for gemini (audio_encoding==base64_json).
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus, expected_session_id="gemini-s16-1", provider_profile="gemini")
    pcm = np.array([7, -8, 9], dtype="<i2")
    payload = {
        "serverContent": {
            "modelTurn": {"parts": [{"inlineData": {"data": base64.b64encode(pcm.tobytes()).decode()}}]},
        }
    }

    run(adapter.handle_message(json.dumps(payload)))

    push_args = bus.push_audio.await_args.args
    assert push_args[0].dtype == np.dtype("<i2")
    np.testing.assert_array_equal(push_args[0], pcm)


def test_gemini_profile_f32le_opt_in_via_metadata_inner_encoding():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus, expected_session_id="gemini-f32-1", provider_profile="gemini")
    # Prime the provider session so session_config exists, then opt-in.
    run(adapter.handle_message(json.dumps({"type": "Settings", "audio": {"sampleRate": 16_000}})))
    adapter.session_config.setdefault("metadata", {})["inner_encoding"] = "pcm_f32le"
    pcm = np.array([0.125, -0.25, 0.5], dtype="<f4")
    payload = {
        "serverContent": {
            "modelTurn": {"parts": [{"inlineData": {"data": base64.b64encode(pcm.tobytes()).decode()}}]},
        }
    }

    run(adapter.handle_message(json.dumps(payload)))

    push_args = bus.push_audio.await_args.args
    assert push_args[0].dtype == np.dtype("<f4")
    np.testing.assert_array_equal(push_args[0], pcm)
    assert push_args[1:] == (16_000, 1)


def test_gemini_profile_f32le_misaligned_is_rejected():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus, expected_session_id="gemini-f32-err-1", provider_profile="gemini")
    run(adapter.handle_message(json.dumps({"type": "Settings"})))
    adapter.session_config.setdefault("metadata", {})["inner_encoding"] = "pcm_f32le"
    payload = {
        "serverContent": {
            "modelTurn": {"parts": [{"inlineData": {"data": base64.b64encode(b"\x00\x01\x02").decode()}}]},
        }
    }

    response = run(adapter.handle_message(json.dumps(payload)))

    assert response[0].code == "INVALID_FORMAT"
    assert "not channel aligned" in response[0].message
