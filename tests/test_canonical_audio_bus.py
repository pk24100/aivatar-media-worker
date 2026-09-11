import asyncio
from unittest.mock import Mock

import numpy as np
import pytest

from streaming.core import audio_bus
from streaming.core.audio_bus import (
    CanonicalAudioBus,
    normalize_channels,
    resample_audio,
)
from streaming.core.session import SessionConfig


def run(coroutine):
    return asyncio.run(coroutine)


def test_normalize_channels_converts_int16_mono_to_float32():
    pcm = np.array([0, 32767, -32768], dtype=np.int16)

    normalized = normalize_channels(pcm, channels=1)

    assert normalized.shape == (3, 1)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized[:, 0], [0.0, 32767 / 32768, -1.0])


def test_normalize_channels_downmixes_stereo_and_upmixes_mono():
    stereo = np.array([[-1.0, 1.0], [0.25, 0.75]], dtype=np.float64)
    mono = normalize_channels(stereo, channels=2, target_channels=1)
    stereo_again = normalize_channels(mono, channels=1, target_channels=2)

    np.testing.assert_allclose(mono, [[0.0], [0.5]])
    np.testing.assert_allclose(stereo_again, [[0.0, 0.0], [0.5, 0.5]])
    assert mono.dtype == np.float32
    assert stereo_again.dtype == np.float32


@pytest.mark.parametrize(
    ("pcm", "channels", "target_channels"),
    [
        (np.array([1, 2, 3]), 0, 1),
        (np.array([1, 2, 3]), 2, 1),
        (np.ones((2, 3, 1)), 1, 1),
        (np.ones((2, 2)), 2, 3),
    ],
)
def test_normalize_channels_rejects_invalid_shapes_and_channel_counts(
    pcm, channels, target_channels
):
    with pytest.raises(ValueError):
        normalize_channels(pcm, channels, target_channels)


def test_resample_audio_returns_float32_without_changing_length_at_same_rate():
    pcm = np.array([1, 2, 3], dtype=np.int16)

    resampled = resample_audio(pcm, source_rate=16_000, target_rate=16_000)

    assert resampled.dtype == np.float32
    np.testing.assert_array_equal(resampled, [1.0, 2.0, 3.0])


def test_resample_audio_calls_resampler_with_target_length(monkeypatch):
    resampler = Mock(return_value=np.array([[0.0], [0.5], [1.0]], dtype=np.float64))
    monkeypatch.setattr(audio_bus, "resample", resampler)
    pcm = np.array([[0.0], [1.0]], dtype=np.float64)

    result = resample_audio(pcm, source_rate=16_000, target_rate=24_000)

    resampler.assert_called_once()
    args, kwargs = resampler.call_args
    np.testing.assert_array_equal(args[0], pcm.astype(np.float32))
    assert args[1] == 3
    assert kwargs == {"axis": 0}
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.0], [0.5], [1.0]])


def test_resample_audio_fallback_interpolates_each_channel(monkeypatch):
    monkeypatch.setattr(audio_bus, "resample", None)
    pcm = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    result = resample_audio(pcm, source_rate=2, target_rate=4)

    expected = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [2.0, 3.0]], dtype=np.float32)
    assert result.shape == (4, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("source_rate,target_rate", [(0, 16_000), (16_000, 0), (-1, 16_000)])
def test_resample_audio_rejects_non_positive_rates(source_rate, target_rate):
    with pytest.raises(ValueError, match="sample rates must be positive"):
        resample_audio(np.ones(2, dtype=np.float32), source_rate, target_rate)


def test_bus_validates_canonical_configuration():
    with pytest.raises(ValueError, match="canonical_sample_rate must be positive"):
        CanonicalAudioBus(canonical_sample_rate=0)
    with pytest.raises(ValueError, match="canonical_channels must be 1 or 2"):
        CanonicalAudioBus(canonical_channels=3)


def test_bus_connect_accepts_dict_session_configuration():
    bus = CanonicalAudioBus()
    config = {
        "audio_encoding": "pcm_s16le",
        "sample_rate": 16_000,
        "channels": 2,
        "session_id": "session-1",
    }

    run(bus.connect(config))

    assert bus.is_connected is True
    assert isinstance(bus._config, SessionConfig)
    assert bus._config.session_id == "session-1"
    assert bus._config.channels == 2


def test_bus_rejects_media_operations_before_connection():
    bus = CanonicalAudioBus()
    operations = [
        lambda: bus.start_utterance(),
        lambda: bus.push_audio(np.zeros(1, dtype=np.float32), 48_000, 1),
        lambda: bus.end_utterance(),
        lambda: bus.cancel_utterance(),
        lambda: bus.end_session(),
    ]

    for operation in operations:
        with pytest.raises(RuntimeError, match="audio bus is not connected"):
            run(operation())


def test_push_audio_normalizes_channels_resamples_and_queues_frame(monkeypatch):
    monkeypatch.setattr(audio_bus, "resample", None)
    bus = CanonicalAudioBus(canonical_sample_rate=48_000, canonical_channels=1)
    run(bus.connect({"sample_rate": 16_000, "channels": 2}))
    pcm = np.array([[32767, -32768], [0, 32767]], dtype=np.int16)

    frame = run(
        bus.push_audio(
            pcm,
            sample_rate=16_000,
            channels=2,
            sequence_number=4,
            timestamp=12.5,
        )
    )

    downmixed = np.array([(32767 / 32768 + -1.0) / 2, 32767 / 65536], dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, 2, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, 6, endpoint=False)
    expected = np.interp(target_positions, source_positions, downmixed)
    assert frame.sample_rate == 48_000
    assert frame.channels == 1
    assert frame.sequence_number == 4
    assert frame.protocol_sequence_number == 4
    assert frame.timestamp == 12.5
    assert frame.pcm.shape == (6,)
    assert frame.pcm.dtype == np.float32
    assert frame.pcm.flags.c_contiguous
    np.testing.assert_allclose(frame.pcm, expected)
    assert run(bus.queue.get()) is frame
    assert bus.last_sequence_number == 4
    assert bus._state.utterance_active is True


def test_bus_sequence_numbers_never_move_backwards():
    bus = CanonicalAudioBus()
    run(bus.connect())
    pcm = np.zeros(1, dtype=np.float32)

    first = run(bus.push_audio(pcm, 48_000, 1, sequence_number=3))
    ended = run(bus.end_utterance())
    second = run(bus.push_audio(pcm, 48_000, 1, sequence_number=1))
    cancelled = run(bus.cancel_utterance(sequence_number=9))
    final = run(bus.end_utterance())

    assert [first.sequence_number, ended.sequence_number, second.sequence_number] == [3, 4, 5]
    assert [first.protocol_sequence_number, second.protocol_sequence_number, cancelled.protocol_sequence_number] == [3, 1, 9]
    assert ended.protocol_sequence_number is None
    assert cancelled.sequence_number == 9
    assert final.sequence_number == 10
    assert final.protocol_sequence_number is None
    assert bus.last_sequence_number == 10


def test_bus_rejects_negative_supplied_sequence():
    bus = CanonicalAudioBus()
    run(bus.connect())

    with pytest.raises(ValueError, match="sequence_number must be non-negative"):
        run(bus.push_audio(np.zeros(1, dtype=np.float32), 48_000, 1, sequence_number=-1))

    assert bus.last_sequence_number == -1
    assert bus.queue.empty()


def test_end_utterance_queues_empty_end_marker_and_clears_active_state():
    bus = CanonicalAudioBus()
    run(bus.connect())
    run(bus.start_utterance())

    marker = run(bus.end_utterance(sequence_number=8))

    assert marker.pcm.shape == (0,)
    assert marker.pcm.dtype == np.float32
    assert marker.sample_rate == bus.canonical_sample_rate
    assert marker.channels == bus.canonical_channels
    assert marker.sequence_number == 8
    assert marker.protocol_sequence_number == 8
    assert marker.is_end_of_utterance is True
    assert marker.is_cancelled is False
    assert bus._state.utterance_active is False
    assert run(bus.queue.get()) is marker


def test_interruption_clears_audio_and_queues_cancel_marker():
    bus = CanonicalAudioBus()
    run(bus.connect())
    run(bus.push_audio(np.ones(4, dtype=np.float32), 48_000, 1))

    run(bus.on_interrupted())

    marker = run(bus.queue.get())
    assert marker.pcm.shape == (0,)
    assert marker.is_cancelled is True
    assert marker.is_end_of_utterance is False
    assert bus._state.utterance_active is False
    assert bus.queue.empty()


def test_end_session_ends_active_utterance_then_queues_sentinel():
    bus = CanonicalAudioBus()
    run(bus.connect())
    run(bus.start_utterance())

    run(bus.end_session(sequence_number=12))

    marker = run(bus.queue.get())
    sentinel = run(bus.queue.get())
    assert marker.is_end_of_utterance is True
    assert marker.sequence_number == 12
    assert marker.protocol_sequence_number == 12
    assert sentinel is None
    assert bus.is_connected is False
    assert bus._state.utterance_active is False
