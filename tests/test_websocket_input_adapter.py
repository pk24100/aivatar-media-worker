import asyncio
import base64
import json
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from streaming.transport.adapters.websocket_input import WebsocketInputAdapter
from streaming.core.audio_bus import CanonicalAudioBus
from streaming.protocol.messages import ErrorMessage, EndedMessage, PongMessage, ProtocolError, StartedMessage


def run(coroutine):
    return asyncio.run(coroutine)


def make_bus():
    bus = Mock(spec=CanonicalAudioBus)
    bus.canonical_sample_rate = 48_000
    bus.canonical_channels = 1
    bus.connect = AsyncMock()
    bus.disconnect = AsyncMock()
    bus.start_utterance = AsyncMock()
    bus.push_audio = AsyncMock()
    bus.end_utterance = AsyncMock()
    bus.cancel_utterance = AsyncMock()
    bus.end_session = AsyncMock()
    return bus


def start_payload(
    *,
    session_id="session-1",
    audio_encoding="pcm_s16le",
    sample_rate=16_000,
    channels=1,
    avatar_id="avatar-1",
    metadata=None,
):
    return json.dumps(
        {
            "type": "start",
            "session_id": session_id,
            "audio_encoding": audio_encoding,
            "sample_rate": sample_rate,
            "channels": channels,
            "avatar_id": avatar_id,
            "metadata": {} if metadata is None else metadata,
        }
    )


def negotiate(adapter, **kwargs):
    response = run(adapter.handle_message(start_payload(**kwargs)))
    assert len(response) == 1
    assert isinstance(response[0], StartedMessage)
    return response[0]


def error_from(response):
    assert len(response) == 1
    assert isinstance(response[0], ErrorMessage)
    return response[0]


def test_adapter_starts_disconnected_and_unnegotiated():
    adapter = WebsocketInputAdapter(make_bus(), expected_session_id="session-1")

    assert adapter.is_connected is False
    assert adapter.is_negotiated is False
    assert adapter.last_protocol_sequence == -1


def test_adapter_requires_start_before_protocol_messages():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus, expected_session_id="session-1")

    response = run(adapter.handle_message(json.dumps({"type": "ping", "ts": 1})))
    error = error_from(response)

    assert error.code == "NOT_NEGOTIATED"
    assert error.fatal is False
    bus.connect.assert_not_awaited()
    with pytest.raises(ProtocolError, match="Send start before media"):
        run(adapter.start_utterance())


def test_start_message_negotiates_and_returns_server_audio_format():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus, expected_session_id="session-1")

    response = negotiate(
        adapter,
        audio_encoding="pcm_f32le",
        sample_rate=24_000,
        channels=2,
        metadata={"locale": "en-US"},
    )

    assert response.session_id == "session-1"
    assert response.server_sample_rate == 48_000
    assert response.server_channels == 1
    assert adapter.is_connected is True
    assert adapter.is_negotiated is True
    assert adapter.session_config == {
        "audio_encoding": "pcm_f32le",
        "sample_rate": 24_000,
        "channels": 2,
        "avatar_id": "avatar-1",
        "session_id": "session-1",
        "metadata": {"locale": "en-US"},
    }
    bus.connect.assert_awaited_once_with(adapter.session_config)


def test_expected_session_id_mismatch_is_fatal_and_does_not_connect():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus, expected_session_id="expected")

    response = run(adapter.handle_message(start_payload(session_id="received")))
    error = error_from(response)

    assert error.code == "SESSION_NOT_FOUND"
    assert error.fatal is True
    assert adapter.is_connected is False
    assert adapter.is_negotiated is False
    bus.connect.assert_not_awaited()


def test_second_start_message_is_rejected_after_negotiation():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter)

    response = run(adapter.handle_message(start_payload()))
    error = error_from(response)

    assert error.code == "INVALID_FORMAT"
    assert error.message == "start may only be the first message"
    bus.connect.assert_awaited_once()


def test_lifecycle_messages_forward_monotonic_sequences():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter)

    assert run(adapter.handle_message(json.dumps({"type": "start_utterance", "seq": 4}))) == []
    assert adapter.last_protocol_sequence == 4
    assert run(adapter.handle_message(json.dumps({"type": "end_utterance", "seq": 4}))) == []
    assert run(adapter.handle_message(json.dumps({"type": "cancel_utterance", "seq": 5}))) == []

    bus.start_utterance.assert_awaited_once_with()
    bus.end_utterance.assert_awaited_once_with(4)
    bus.cancel_utterance.assert_awaited_once_with(5)
    assert adapter.last_protocol_sequence == 5
    assert adapter._utterance_active is False


def test_backwards_sequence_is_rejected_without_forwarding_lifecycle_call():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter)
    run(adapter.handle_message(json.dumps({"type": "start_utterance", "seq": 3})))

    response = run(adapter.handle_message(json.dumps({"type": "end_utterance", "seq": 2})))
    error = error_from(response)

    assert error.code == "INVALID_FORMAT"
    assert error.message == "seq must not move backwards"
    assert adapter.last_protocol_sequence == 3
    bus.end_utterance.assert_not_awaited()


def test_binary_pcm_s16le_is_decoded_and_forwarded():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="pcm_s16le", sample_rate=16_000, channels=1)
    pcm = np.array([-32768, 0, 32767], dtype="<i2")

    response = run(adapter.handle_message(pcm.tobytes()))

    assert response == []
    args, kwargs = bus.push_audio.call_args
    np.testing.assert_array_equal(args[0], pcm)
    assert args[1:] == (16_000, 1)
    assert kwargs == {}
    assert adapter._utterance_active is True


def test_binary_pcm_f32le_is_decoded_with_stereo_alignment():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="pcm_f32le", sample_rate=24_000, channels=2)
    pcm = np.array([0.25, -0.5, 0.75, 1.0], dtype="<f4")

    response = run(adapter.handle_message(pcm.tobytes()))

    assert response == []
    args, kwargs = bus.push_audio.call_args
    np.testing.assert_array_equal(args[0], pcm)
    assert args[1:] == (24_000, 2)
    assert kwargs == {}


def test_binary_pcm_payload_must_be_channel_aligned():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="pcm_s16le", channels=2)

    response = run(adapter.handle_message(b"\x00\x01"))
    error = error_from(response)

    assert error.code == "INVALID_FORMAT"
    assert error.message == "pcm_s16le payload is not channel aligned"
    bus.push_audio.assert_not_awaited()


def test_binary_audio_is_rejected_for_base64_json_sessions():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="base64_json")

    response = run(adapter.handle_message(b"\x00\x00"))
    error = error_from(response)

    assert error.code == "INVALID_FORMAT"
    assert error.message == "base64_json sessions must send audio JSON messages"
    bus.push_audio.assert_not_awaited()


def test_base64_json_audio_is_decoded_with_protocol_sequence():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="base64_json", sample_rate=8_000, channels=1)
    pcm = np.array([1, -2, 3], dtype="<i2")
    encoded = base64.b64encode(pcm.tobytes()).decode()

    response = run(
        adapter.handle_message(json.dumps({"type": "audio", "data": encoded, "seq": 7}))
    )

    assert response == []
    args, kwargs = bus.push_audio.call_args
    np.testing.assert_array_equal(args[0], pcm)
    assert args[1:] == (8_000, 1)
    assert kwargs == {"sequence_number": 7}
    assert adapter.last_protocol_sequence == 7
    assert adapter._utterance_active is True


def test_json_audio_is_rejected_for_non_base64_sessions():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="pcm_s16le")
    encoded = base64.b64encode(b"\x00\x00").decode()

    response = run(
        adapter.handle_message(json.dumps({"type": "audio", "data": encoded, "seq": 1}))
    )
    error = error_from(response)

    assert error.code == "INVALID_FORMAT"
    assert error.message == "audio JSON is only valid for base64_json sessions"
    bus.push_audio.assert_not_awaited()


def test_invalid_base64_is_reported_as_protocol_error():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="base64_json")

    response = run(
        adapter.handle_message(json.dumps({"type": "audio", "data": "bad!", "seq": 1}))
    )
    error = error_from(response)

    assert error.code == "INVALID_FORMAT"
    assert error.message == "audio data must be valid base64"
    bus.push_audio.assert_not_awaited()


def test_bus_value_errors_are_converted_to_nonfatal_format_errors():
    bus = make_bus()
    bus.push_audio.side_effect = ValueError("audio rejected")
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="pcm_s16le")

    response = run(adapter.handle_message(np.zeros(1, dtype="<i2").tobytes()))
    error = error_from(response)

    assert error.code == "INVALID_FORMAT"
    assert error.message == "audio rejected"
    assert error.fatal is False


def test_ping_returns_pong_with_current_server_timestamp(monkeypatch):
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter)
    monkeypatch.setattr("streaming.transport.adapters.websocket_input.time.time", lambda: 1234.5)

    response = run(adapter.handle_message(json.dumps({"type": "ping", "ts": 99.8})))

    assert response == [PongMessage(99, 1_234_500)]


def test_end_session_forwards_sequence_invokes_callback_and_resets_state():
    bus = make_bus()
    callback = AsyncMock()
    adapter = WebsocketInputAdapter(bus, on_end_session=callback)
    negotiate(adapter)

    response = run(adapter.handle_message(json.dumps({"type": "end_session", "seq": 10})))

    assert response == [EndedMessage("session-1")]
    bus.end_session.assert_awaited_once_with(10)
    callback.assert_awaited_once_with()
    assert adapter.last_protocol_sequence == 10
    assert adapter.is_connected is False
    assert adapter.is_negotiated is False
    assert adapter._utterance_active is False


def test_on_interrupted_uses_latest_protocol_sequence():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter)
    run(adapter.handle_message(json.dumps({"type": "start_utterance", "seq": 6})))

    run(adapter.on_interrupted())

    bus.cancel_utterance.assert_awaited_once_with(6)
    assert adapter._utterance_active is False


def test_disconnect_only_changes_adapter_connection_state():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter)

    run(adapter.disconnect())

    assert adapter.is_connected is False
    assert adapter.is_negotiated is False
    bus.disconnect.assert_not_awaited()


def test_base64_json_default_remains_s16le_without_opt_in():
    # Default preserves base64->s16le when no inner_encoding is set.
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="base64_json", sample_rate=16_000, channels=1)
    pcm = np.array([10, -20, 30, -40], dtype="<i2")
    encoded = base64.b64encode(pcm.tobytes()).decode()

    response = run(
        adapter.handle_message(json.dumps({"type": "audio", "data": encoded, "seq": 1}))
    )

    assert response == []
    args, _ = bus.push_audio.call_args
    assert args[0].dtype == np.dtype("<i2")
    np.testing.assert_array_equal(args[0], pcm)


def test_base64_json_f32le_opt_in_via_metadata():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(
        adapter,
        audio_encoding="base64_json",
        sample_rate=16_000,
        channels=1,
        metadata={"inner_encoding": "pcm_f32le"},
    )
    pcm = np.array([0.25, -0.5, 0.75, -1.0], dtype="<f4")
    encoded = base64.b64encode(pcm.tobytes()).decode()

    response = run(
        adapter.handle_message(json.dumps({"type": "audio", "data": encoded, "seq": 2}))
    )

    assert response == []
    args, kwargs = bus.push_audio.call_args
    assert args[0].dtype == np.dtype("<f4")
    np.testing.assert_array_equal(args[0], pcm)
    assert args[1:] == (16_000, 1)
    assert kwargs == {"sequence_number": 2}


def test_base64_json_f32le_misaligned_is_rejected():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(
        adapter,
        audio_encoding="base64_json",
        sample_rate=16_000,
        channels=1,
        metadata={"inner_encoding": "pcm_f32le"},
    )
    # 3 bytes is not aligned to 4*1 for f32le.
    encoded = base64.b64encode(b"\x00\x01\x02").decode()

    response = run(
        adapter.handle_message(json.dumps({"type": "audio", "data": encoded, "seq": 3}))
    )
    error = error_from(response)

    assert error.code == "INVALID_FORMAT"
    assert error.message == "base64 PCM payload is not channel aligned"
    bus.push_audio.assert_not_awaited()


def test_base64_json_s16le_misaligned_is_rejected():
    bus = make_bus()
    adapter = WebsocketInputAdapter(bus)
    negotiate(adapter, audio_encoding="base64_json", channels=1)
    # 3 bytes is not aligned to 2*1 for s16le.
    encoded = base64.b64encode(b"\x00\x01\x02").decode()

    response = run(
        adapter.handle_message(json.dumps({"type": "audio", "data": encoded, "seq": 4}))
    )
    error = error_from(response)

    assert error.code == "INVALID_FORMAT"
    assert error.message == "base64 PCM payload is not channel aligned"
    bus.push_audio.assert_not_awaited()
