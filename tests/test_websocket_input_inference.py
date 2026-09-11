"""Deterministic tests for silence-gap utterance boundary inference.

The adapter infers utterance boundaries when a canonical client streams audio
without explicit start_utterance/end_utterance controls. All timer behavior is
driven through an injected wait callable (``adapter._wait_silence_delay``) so no
test sleeps on wall-clock time.
"""

import asyncio
import base64
import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

import streaming
from streaming.core.audio_bus import CanonicalAudioBus
from streaming.protocol.messages import (
    ErrorMessage,
    ProtocolError,
    StartMessage,
    StartedMessage,
    parse_message,
)
from streaming.transport.adapters.websocket_input import WebsocketInputAdapter


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
    utterance_silence_ms="__absent__",
):
    payload = {
        "type": "start",
        "session_id": session_id,
        "audio_encoding": audio_encoding,
        "sample_rate": sample_rate,
        "channels": channels,
        "avatar_id": avatar_id,
        "metadata": {} if metadata is None else metadata,
    }
    if utterance_silence_ms != "__absent__":
        payload["utterance_silence_ms"] = utterance_silence_ms
    return json.dumps(payload)


def pcm_bytes(*samples):
    return np.array(samples, dtype="<i2").tobytes()


async def negotiate(adapter, **kwargs):
    responses = await adapter.handle_message(start_payload(**kwargs))
    assert len(responses) == 1
    assert isinstance(responses[0], StartedMessage)
    return responses[0]


async def pump(turns=3):
    for _ in range(turns):
        await asyncio.sleep(0)


class SilenceGate:
    """Manually released stand-in for the adapter's silence wait."""

    def __init__(self):
        self.event = asyncio.Event()
        self.delays = []

    async def wait(self, delay_s):
        self.delays.append(delay_s)
        await self.event.wait()

    def fire(self):
        self.event.set()


def _load_idle_timeout_s():
    """Import IDLE_TIMEOUT_S from _session_states.py without the package init
    (the batched package __init__ requires the flash_head module)."""
    module_path = (
        Path(streaming.__file__).resolve().parent
        / "inference"
        / "batched"
        / "_session_states.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_session_states_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IDLE_TIMEOUT_S


# ---------------------------------------------------------------------------
# start message parsing and configuration bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 100, 800, 4_999, 5_000])
def test_parse_start_accepts_valid_utterance_silence_ms(value):
    raw = json.loads(start_payload())
    raw["utterance_silence_ms"] = value

    message = parse_message(json.dumps(raw))

    assert isinstance(message, StartMessage)
    assert message.utterance_silence_ms == value


def test_parse_start_defaults_utterance_silence_ms_to_none():
    message = parse_message(start_payload())

    assert isinstance(message, StartMessage)
    assert message.utterance_silence_ms is None


@pytest.mark.parametrize(
    "value",
    [True, False, "800", "0", 800.0, 0.0, -1, 1, 50, 99, 5_001, 10_000, None, [800], {"ms": 800}],
)
def test_parse_start_rejects_invalid_utterance_silence_ms(value):
    raw = json.loads(start_payload())
    raw["utterance_silence_ms"] = value

    with pytest.raises(ProtocolError) as raised:
        parse_message(json.dumps(raw))

    assert raised.value.code == "INVALID_FORMAT"
    assert raised.value.fatal is False


def test_adapter_rejects_invalid_utterance_silence_ms_without_connecting():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)

        responses = await adapter.handle_message(start_payload(utterance_silence_ms=99))

        assert len(responses) == 1
        assert isinstance(responses[0], ErrorMessage)
        assert responses[0].code == "INVALID_FORMAT"
        bus.connect.assert_not_awaited()
        assert adapter.is_negotiated is False

    run(scenario())


def test_default_silence_ms_comes_from_env_default_800(monkeypatch):
    monkeypatch.delenv("FACEMODE_UTTERANCE_SILENCE_MS", raising=False)

    adapter = WebsocketInputAdapter(make_bus())

    assert adapter._utterance_silence_ms == 800


def test_env_silence_ms_override(monkeypatch):
    monkeypatch.setenv("FACEMODE_UTTERANCE_SILENCE_MS", "1500")

    adapter = WebsocketInputAdapter(make_bus())

    assert adapter._utterance_silence_ms == 1500


@pytest.mark.parametrize("raw", ["abc", "", "99999", "-5"])
def test_invalid_env_silence_ms_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv("FACEMODE_UTTERANCE_SILENCE_MS", raw)

    adapter = WebsocketInputAdapter(make_bus())

    assert adapter._utterance_silence_ms == 800


def test_start_message_overrides_silence_ms():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)

        await negotiate(adapter, utterance_silence_ms=1200)

        assert adapter._utterance_silence_ms == 1200
        # The tuning key is consumed by the adapter, not forwarded into the
        # bus SessionConfig.
        assert "utterance_silence_ms" not in adapter.session_config
        bus.connect.assert_awaited_once()

    run(scenario())


def test_direct_connect_consumes_silence_override_from_config():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)

        await adapter.connect(
            {
                "audio_encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
                "session_id": "session-1",
                "utterance_silence_ms": 250,
            }
        )

        assert adapter._utterance_silence_ms == 250
        assert "utterance_silence_ms" not in adapter.session_config

    run(scenario())


def test_inferred_boundary_stays_strictly_below_idle_timeout(monkeypatch):
    # B2: the inferred end must always land before the engine removes an idle
    # session (IDLE_TIMEOUT_S). Both the env default and the largest accepted
    # override are asserted against the real constant.
    monkeypatch.delenv("FACEMODE_UTTERANCE_SILENCE_MS", raising=False)
    idle_timeout_s = _load_idle_timeout_s()

    adapter = WebsocketInputAdapter(make_bus())

    assert adapter._utterance_silence_ms < idle_timeout_s * 1000
    assert 5_000 < idle_timeout_s * 1000


# ---------------------------------------------------------------------------
# implicit start and silence-timer driving
# ---------------------------------------------------------------------------


def test_binary_audio_implicitly_starts_utterance_and_arms_timer():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1, -2, 3))

        bus.start_utterance.assert_awaited_once_with()
        bus.push_audio.assert_awaited_once()
        assert adapter._utterance_active is True
        assert adapter._silence_timer_task is not None
        assert not adapter._silence_timer_task.done()
        await pump()
        assert gate.delays == [0.8]
        bus.end_utterance.assert_not_awaited()

        await adapter.disconnect()

    run(scenario())


def test_base64_audio_implicitly_starts_utterance_and_arms_timer():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter, audio_encoding="base64_json", sample_rate=8_000)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait
        encoded = base64.b64encode(pcm_bytes(1, -2)).decode()

        await adapter.handle_message(
            json.dumps({"type": "audio", "data": encoded, "seq": 3})
        )

        bus.start_utterance.assert_awaited_once_with()
        _, kwargs = bus.push_audio.call_args
        assert kwargs == {"sequence_number": 3}
        assert adapter._silence_timer_task is not None
        assert not adapter._silence_timer_task.done()

        await adapter.disconnect()

    run(scenario())


def test_silence_timeout_emits_inferred_end_with_bus_assigned_sequence():
    async def scenario():
        bus = CanonicalAudioBus()
        adapter = WebsocketInputAdapter(bus, expected_session_id="session-1")
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1, -2, 3, -4))
        timer = adapter._silence_timer_task
        assert timer is not None

        gate.fire()
        await timer

        frames = []
        while not bus.queue.empty():
            frames.append(bus.queue.get_nowait())
        end_frames = [frame for frame in frames if frame.is_end_of_utterance]
        assert len(end_frames) == 1
        end = end_frames[0]
        # Inferred boundaries carry no client sequence: the bus assigns one.
        assert end.protocol_sequence_number is None
        assert end.sequence_number >= 0
        # Audio frames precede the boundary marker on the bus queue.
        assert frames[-1] is end
        assert adapter._utterance_active is False

    run(scenario())


def test_inferred_end_does_not_echo_client_audio_sequence():
    async def scenario():
        bus = CanonicalAudioBus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter, audio_encoding="base64_json", sample_rate=8_000)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait
        encoded = base64.b64encode(pcm_bytes(1, -2)).decode()
        await adapter.handle_message(
            json.dumps({"type": "audio", "data": encoded, "seq": 7})
        )

        gate.fire()
        await adapter._silence_timer_task

        end = None
        while not bus.queue.empty():
            frame = bus.queue.get_nowait()
            if frame.is_end_of_utterance:
                end = frame
        assert end is not None
        assert adapter.last_protocol_sequence == 7
        # Even though the client sent audio seqs, the inferred boundary itself
        # carries no client sequence.
        assert end.protocol_sequence_number is None

    run(scenario())


def test_each_audio_frame_replaces_pending_silence_timer():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        first = adapter._silence_timer_task
        await adapter.handle_message(pcm_bytes(2))
        second = adapter._silence_timer_task

        assert first is not second
        # The previous timer was cancelled and awaited before re-arming.
        assert first.cancelled()
        assert not second.done()
        # Still one implicit start; an already-open utterance is not reopened.
        bus.start_utterance.assert_awaited_once()

        gate.fire()
        await second
        bus.end_utterance.assert_awaited_once_with(None)

    run(scenario())


def test_new_audio_after_inferred_end_starts_fresh_utterance():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        gate.fire()
        await adapter._silence_timer_task
        bus.end_utterance.assert_awaited_once_with(None)

        gate.event.clear()
        await adapter.handle_message(pcm_bytes(2))
        assert bus.start_utterance.await_count == 2
        second = adapter._silence_timer_task
        assert second is not None and not second.done()

        gate.fire()
        await second
        assert bus.end_utterance.await_count == 2

    run(scenario())


def test_inferred_end_fires_exactly_once_per_utterance():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        gate.fire()
        await adapter._silence_timer_task
        bus.end_utterance.assert_awaited_once()

        # A stray second fire must not emit a duplicate boundary.
        await adapter._finish_inferred_utterance()
        bus.end_utterance.assert_awaited_once()

    run(scenario())


# ---------------------------------------------------------------------------
# explicit latch and preserved client sequences
# ---------------------------------------------------------------------------


def test_client_start_utterance_latches_explicit_and_cancels_timer():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        timer = adapter._silence_timer_task
        assert timer is not None

        await adapter.handle_message(json.dumps({"type": "start_utterance", "seq": 3}))

        assert adapter._utterance_mode == "explicit"
        assert timer.cancelled()
        assert adapter._silence_timer_task is None

        # Further audio in an explicit session never re-arms the timer.
        await adapter.handle_message(pcm_bytes(2))
        assert adapter._silence_timer_task is None

        gate.fire()
        await pump()
        bus.end_utterance.assert_not_awaited()

    run(scenario())


def test_client_end_utterance_latches_and_preserves_client_sequence():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        timer = adapter._silence_timer_task

        await adapter.handle_message(json.dumps({"type": "end_utterance", "seq": 9}))

        assert adapter._utterance_mode == "explicit"
        assert timer.cancelled()
        bus.end_utterance.assert_awaited_once_with(9)

    run(scenario())


def test_client_cancel_utterance_latches_and_preserves_client_sequence():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        timer = adapter._silence_timer_task

        await adapter.handle_message(json.dumps({"type": "cancel_utterance", "seq": 5}))

        assert adapter._utterance_mode == "explicit"
        assert timer.cancelled()
        bus.cancel_utterance.assert_awaited_once_with(5)

    run(scenario())


def test_explicit_latch_survives_renegotiation_on_the_same_adapter():
    # B5: a plugin that drops its transport and re-negotiates on the same
    # session must never fall back into inferred mode.
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        await adapter.handle_message(json.dumps({"type": "start_utterance", "seq": 1}))
        assert adapter._utterance_mode == "explicit"

        await adapter.disconnect()
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait
        await adapter.handle_message(pcm_bytes(1))

        assert adapter._utterance_mode == "explicit"
        assert adapter._silence_timer_task is None
        bus.end_utterance.assert_not_awaited()

    run(scenario())


def test_zero_silence_disables_inference_but_keeps_explicit_controls():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter, utterance_silence_ms=0)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))

        bus.start_utterance.assert_not_awaited()
        assert adapter._silence_timer_task is None

        # Explicit controls still work when inference is disabled.
        await adapter.handle_message(json.dumps({"type": "end_utterance", "seq": 4}))
        bus.end_utterance.assert_awaited_once_with(4)

    run(scenario())


# ---------------------------------------------------------------------------
# provider profiles never infer; provider boundaries carry no client sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["deepgram", "gemini"])
def test_provider_profile_sessions_force_explicit_mode(profile):
    adapter = WebsocketInputAdapter(
        make_bus(), expected_session_id="provider-1", provider_profile=profile
    )

    assert adapter._utterance_mode == "explicit"


def test_deepgram_binary_audio_does_not_arm_inference_timer():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(
            bus, expected_session_id="dg-1", provider_profile="deepgram"
        )
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(0, 0, 0, 0))

        assert adapter._silence_timer_task is None
        bus.start_utterance.assert_not_awaited()

    run(scenario())


def test_provider_driven_end_event_passes_no_client_sequence():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(
            bus, expected_session_id="dg-2", provider_profile="deepgram"
        )

        await adapter.handle_message(json.dumps({"type": "AgentAudioDone"}))

        bus.end_utterance.assert_awaited_once_with(None)

    run(scenario())


def test_provider_driven_interruption_passes_no_client_sequence():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(
            bus, expected_session_id="g-1", provider_profile="gemini"
        )

        await adapter.handle_message(json.dumps({"interrupted": True}))

        bus.cancel_utterance.assert_awaited_once_with(None)

    run(scenario())


# ---------------------------------------------------------------------------
# cleanup paths
# ---------------------------------------------------------------------------


def test_end_session_cancels_pending_silence_timer():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        timer = adapter._silence_timer_task

        await adapter.handle_message(json.dumps({"type": "end_session", "seq": 4}))

        assert timer.cancelled()
        assert adapter._silence_timer_task is None
        bus.end_session.assert_awaited_once_with(4)
        bus.end_utterance.assert_not_awaited()

    run(scenario())


def test_disconnect_cancels_and_awaits_silence_timer():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        timer = adapter._silence_timer_task

        await adapter.disconnect()

        assert timer.cancelled()
        assert adapter._silence_timer_task is None
        bus.end_utterance.assert_not_awaited()

    run(scenario())


def test_fatal_protocol_error_cancels_silence_timer():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        timer = adapter._silence_timer_task

        bus.push_audio.side_effect = ProtocolError("INTERNAL_ERROR", "boom", fatal=True)
        responses = await adapter.handle_message(pcm_bytes(2))

        assert len(responses) == 1
        assert isinstance(responses[0], ErrorMessage)
        assert responses[0].fatal is True
        assert timer.cancelled()
        assert adapter._silence_timer_task is None

    run(scenario())


def test_finish_after_disconnect_is_a_no_op():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        await adapter.disconnect()

        # A timer that somehow still runs after disconnect must not emit a
        # boundary on a torn-down adapter.
        await adapter._finish_inferred_utterance()
        bus.end_utterance.assert_not_awaited()

    run(scenario())


def test_completed_timer_leaves_no_pending_task_reference():
    async def scenario():
        bus = make_bus()
        adapter = WebsocketInputAdapter(bus)
        await negotiate(adapter)
        gate = SilenceGate()
        adapter._wait_silence_delay = gate.wait

        await adapter.handle_message(pcm_bytes(1))
        timer = adapter._silence_timer_task
        gate.fire()
        await timer

        assert adapter._silence_timer_task is None
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
        assert pending == []

    run(scenario())
