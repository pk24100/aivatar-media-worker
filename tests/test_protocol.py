import base64
import json

import pytest

from streaming.protocol.messages import (
    AudioMessage,
    AudioReadyMessage,
    BinaryAudioMessage,
    CancelUtteranceMessage,
    EndSessionMessage,
    EndUtteranceMessage,
    EndedMessage,
    ErrorMessage,
    PingMessage,
    PongMessage,
    PREFERRED_SAMPLE_RATES,
    ProtocolError,
    ProtocolMessage,
    StartMessage,
    StartUtteranceMessage,
    StartedMessage,
    UtteranceEndedMessage,
    error_message,
    parse_message,
    serialize_message,
)


AUDIO_DATA = base64.b64encode(b"pcm payload").decode()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            StartMessage(
                "session-1",
                "pcm_s16le",
                16_000,
                1,
                avatar_id="avatar-1",
                metadata={"locale": "en-US"},
            ),
            {
                "type": "start",
                "session_id": "session-1",
                "audio_encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
                "avatar_id": "avatar-1",
                "metadata": {"locale": "en-US"},
            },
        ),
        (AudioMessage(AUDIO_DATA, 3), {"type": "audio", "data": AUDIO_DATA, "seq": 3}),
        (StartUtteranceMessage(4), {"type": "start_utterance", "seq": 4}),
        (EndUtteranceMessage(5), {"type": "end_utterance", "seq": 5}),
        (CancelUtteranceMessage(6), {"type": "cancel_utterance", "seq": 6}),
        (EndSessionMessage(7), {"type": "end_session", "seq": 7}),
        (PingMessage(8), {"type": "ping", "ts": 8}),
        (
            StartedMessage("session-1", server_sample_rate=24_000, server_channels=2),
            {
                "type": "started",
                "session_id": "session-1",
                "server_sample_rate": 24_000,
                "server_channels": 2,
            },
        ),
        (AudioReadyMessage(9), {"type": "audio_ready", "seq": 9}),
        (
            UtteranceEndedMessage(10, 2),
            {"type": "utterance_ended", "seq": 10, "cycles": 2},
        ),
        (EndedMessage("session-1"), {"type": "ended", "session_id": "session-1"}),
        (PongMessage(11, 12), {"type": "pong", "ts": 11, "server_ts": 12}),
        (
            ErrorMessage("RATE_LIMITED", "try again", fatal=True),
            {"type": "error", "code": "RATE_LIMITED", "message": "try again", "fatal": True},
        ),
    ],
)
def test_serialize_message_emits_protocol_payload(message, expected):
    serialized = serialize_message(message)

    assert isinstance(serialized, str)
    assert json.loads(serialized) == expected


def test_serialize_message_preserves_unicode_text():
    serialized = serialize_message(ErrorMessage("INTERNAL_ERROR", "café"))

    assert "café" in serialized
    assert json.loads(serialized)["message"] == "café"


def test_serialize_binary_message_returns_original_bytes():
    payload = b"\x00\x01\xff"

    assert serialize_message(BinaryAudioMessage(payload)) == payload


def test_serialize_message_rejects_non_protocol_values():
    with pytest.raises(TypeError, match="msg must be a ProtocolMessage"):
        serialize_message({"type": "ping"})


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        (
            {
                "type": "start",
                "session_id": "session-1",
                "audio_encoding": "base64_json",
                "sample_rate": 48_000,
                "channels": 2,
                "avatar_id": "avatar-1",
                "metadata": {"tenant": "test", "enabled": True},
            },
            StartMessage,
        ),
        ({"type": "audio", "data": AUDIO_DATA, "seq": 0}, AudioMessage),
        ({"type": "start_utterance", "seq": 1}, StartUtteranceMessage),
        ({"type": "end_utterance", "seq": 2}, EndUtteranceMessage),
        ({"type": "cancel_utterance", "seq": 3}, CancelUtteranceMessage),
        ({"type": "end_session", "seq": 4}, EndSessionMessage),
        ({"type": "ping", "ts": 123.9}, PingMessage),
    ],
)
def test_parse_message_builds_supported_inbound_messages(payload, expected_type):
    message = parse_message(json.dumps(payload))

    assert isinstance(message, expected_type)
    assert message.type == payload["type"]

    if isinstance(message, StartMessage):
        assert message.session_id == payload["session_id"]
        assert message.metadata == payload["metadata"]
    elif isinstance(message, AudioMessage):
        assert message.data == payload["data"]
        assert message.seq == payload["seq"]
    elif isinstance(message, PingMessage):
        assert message.ts == 123
    else:
        assert message.seq == payload["seq"]


def test_parse_start_message_applies_optional_defaults():
    message = parse_message(
        json.dumps(
            {
                "type": "start",
                "session_id": "session-1",
                "audio_encoding": "pcm_f32le",
                "sample_rate": 8_000,
                "channels": 1,
            }
        )
    )

    assert message == StartMessage("session-1", "pcm_f32le", 8_000, 1)
    assert message.avatar_id == ""
    assert message.metadata == {}


@pytest.mark.parametrize("sample_rate", sorted(PREFERRED_SAMPLE_RATES))
def test_parse_start_accepts_preferred_provider_sample_rates(sample_rate):
    message = parse_message(
        json.dumps(
            {
                "type": "start",
                "session_id": "session-1",
                "audio_encoding": "pcm_s16le",
                "sample_rate": sample_rate,
                "channels": 1,
            }
        )
    )

    assert message.sample_rate == sample_rate


def test_parse_start_accepts_and_logs_uncommon_in_range_sample_rate(caplog):
    with caplog.at_level("INFO", logger="streaming.protocol.messages"):
        message = parse_message(
            json.dumps(
                {
                    "type": "start",
                    "session_id": "session-1",
                    "audio_encoding": "pcm_s16le",
                    "sample_rate": 12_345,
                    "channels": 1,
                }
            )
        )

    assert message.sample_rate == 12_345
    assert "Using uncommon input sample rate: 12345" in caplog.messages


def test_parse_binary_message_wraps_bytes_without_decoding():
    payload = b"binary pcm"

    message = parse_message(payload)

    assert message == BinaryAudioMessage(payload)
    assert message.data == payload


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        json.dumps([]),
        json.dumps({}),
        json.dumps({"type": 1}),
        json.dumps({"type": "start", "session_id": "s", "audio_encoding": "wav", "sample_rate": 16_000, "channels": 1}),
        json.dumps({"type": "start", "session_id": "s", "audio_encoding": "pcm_s16le", "sample_rate": True, "channels": 1}),
        json.dumps({"type": "start", "session_id": "s", "audio_encoding": "pcm_s16le", "sample_rate": 16_000.0, "channels": 1}),
        json.dumps({"type": "start", "session_id": "s", "audio_encoding": "pcm_s16le", "sample_rate": 7_999, "channels": 1}),
        json.dumps({"type": "start", "session_id": "s", "audio_encoding": "pcm_s16le", "sample_rate": 48_001, "channels": 1}),
        json.dumps({"type": "start", "session_id": "s", "audio_encoding": "pcm_s16le", "sample_rate": 16_000, "channels": 3}),
        json.dumps({"type": "start", "session_id": "", "audio_encoding": "pcm_s16le", "sample_rate": 16_000, "channels": 1}),
        json.dumps({"type": "start", "session_id": "s", "audio_encoding": "pcm_s16le", "sample_rate": 16_000, "channels": 1, "avatar_id": 4}),
        json.dumps({"type": "start", "session_id": "s", "audio_encoding": "pcm_s16le", "sample_rate": 16_000, "channels": 1, "metadata": []}),
        json.dumps({"type": "audio", "data": "", "seq": 0}),
        json.dumps({"type": "audio", "data": "not-base64", "seq": 0}),
        json.dumps({"type": "audio", "data": AUDIO_DATA, "seq": True}),
        json.dumps({"type": "audio", "data": AUDIO_DATA, "seq": -1}),
        json.dumps({"type": "start_utterance", "seq": -1}),
        json.dumps({"type": "ping", "ts": True}),
        json.dumps({"type": "unsupported"}),
    ],
)
def test_parse_message_rejects_invalid_text(raw):
    with pytest.raises(ProtocolError) as raised:
        parse_message(raw)

    assert raised.value.code == "INVALID_FORMAT"
    assert raised.value.fatal is False
    assert str(raised.value) == raised.value.message


def test_parse_message_rejects_non_text_non_binary_values():
    with pytest.raises(ProtocolError, match="WebSocket message must be text or binary"):
        parse_message(bytearray(b"pcm"))


def test_error_message_converts_protocol_error_attributes():
    error = ProtocolError("SESSION_NOT_FOUND", "unknown session", fatal=True)

    message = error_message(error)

    assert message == ErrorMessage("SESSION_NOT_FOUND", "unknown session", fatal=True)
    assert message.type == "error"


def test_protocol_message_is_the_common_message_type():
    message = PingMessage(42)

    assert isinstance(message, ProtocolMessage)
