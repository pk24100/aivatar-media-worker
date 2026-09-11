import asyncio
import logging
import socket
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np
import pytest
from livekit import rtc

from streaming.transport.adapters import livekit_egress
from streaming.transport.adapters.livekit_egress import LiveKitEgressAdapter


def run(coroutine):
    return asyncio.run(coroutine)


class FakeVideoPublisher:
    instances = []

    def __init__(self, room, fps, session_id):
        self.room = room
        self.fps = fps
        self.session_id = session_id
        self.first_frame_published = asyncio.Event()
        self._ensure_track = AsyncMock()
        self._send_frame = AsyncMock()
        self.publish_from_state_manager = AsyncMock()
        self.aclose = AsyncMock()
        self.__class__.instances.append(self)


class FakeAudioPublisher:
    instances = []

    def __init__(self, room, sample_rate, num_channels):
        self.room = room
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.push_audio = AsyncMock()
        self.aclose = AsyncMock()
        self.__class__.instances.append(self)


class FakeRoom:
    def __init__(self, *, connect_error=None, disconnect_error=None):
        self.connection_state = rtc.ConnectionState.CONN_CONNECTED
        self.connect_error = connect_error
        self.disconnect_error = disconnect_error
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.connect_started = asyncio.Event()
        self.connect_release = None
        self.disconnect_started = asyncio.Event()
        self.disconnect_release = None
        self.local_participant = Mock()

    async def connect(self, *_args, **_kwargs):
        self.connect_calls += 1
        self.connect_started.set()
        if self.connect_release is not None:
            await self.connect_release.wait()
        if self.connect_error is not None:
            raise self.connect_error

    async def disconnect(self):
        self.disconnect_calls += 1
        self.disconnect_started.set()
        if self.disconnect_release is not None:
            await self.disconnect_release.wait()
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self.connection_state = rtc.ConnectionState.CONN_DISCONNECTED


def install_fake_publishers(monkeypatch):
    FakeVideoPublisher.instances.clear()
    FakeAudioPublisher.instances.clear()
    monkeypatch.setattr(livekit_egress, "VideoPublisher", FakeVideoPublisher)
    monkeypatch.setattr(livekit_egress, "AudioPublisher", FakeAudioPublisher)


def test_livekit_egress_adapter_preserves_publisher_contract(monkeypatch):
    install_fake_publishers(monkeypatch)

    room = FakeRoom()
    adapter = LiveKitEgressAdapter(sample_rate=16_000, channels=1, session_id="session-1")

    async def scenario():
        await adapter.connect({"room": room, "fps": 30})
        await adapter.publish_video_frame(np.zeros((2, 3, 3), dtype=np.uint8), 1.25)
        await adapter.publish_audio_chunk(np.zeros(160, dtype=np.float32), 16_000)
        return adapter

    result = run(scenario())

    assert result.is_connected is True
    assert result.video_publisher.fps == 30
    result.video_publisher._ensure_track.assert_awaited_once()
    result.video_publisher._send_frame.assert_awaited_once()
    audio_publisher = result.audio_publisher
    video_publisher = result.video_publisher
    audio_publisher.push_audio.assert_awaited_once()

    run(result.disconnect())
    assert result.is_connected is False
    assert result.audio_publisher is None
    assert room.disconnect_calls == 0
    video_publisher.aclose.assert_awaited_once()
    audio_publisher.aclose.assert_awaited_once()


def test_livekit_egress_adapter_waits_for_first_frame(monkeypatch):
    install_fake_publishers(monkeypatch)
    adapter = LiveKitEgressAdapter()

    async def scenario():
        await adapter.connect({"room": FakeRoom()})
        task = asyncio.create_task(adapter.wait_for_ready())
        await asyncio.sleep(0)
        assert not task.done()
        adapter.first_frame_published.set()
        await task

    run(scenario())


def test_failed_connect_candidate_is_disconnected_before_retry(monkeypatch):
    install_fake_publishers(monkeypatch)
    first = FakeRoom(connect_error=ConnectionError("synthetic failure"))
    second = FakeRoom()
    candidates = iter((first, second))
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: next(candidates))
    adapter = LiveKitEgressAdapter(room_url="wss://room.example.test", room_token="synthetic")
    adapter._CONNECT_RETRY_BASE_SECONDS = 0

    async def scenario():
        await adapter.connect({})
        assert adapter.room is second
        await adapter.disconnect()

    run(scenario())

    assert first.disconnect_calls == 1
    assert second.disconnect_calls == 1
    assert adapter._teardown_task is None


def test_cancelled_connect_cleans_candidate_and_preserves_cancellation(monkeypatch):
    install_fake_publishers(monkeypatch)
    candidate = FakeRoom()

    async def scenario():
        candidate.connect_release = asyncio.Event()
        monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: candidate)
        adapter = LiveKitEgressAdapter(
            room_url="wss://room.example.test",
            room_token="synthetic",
        )
        task = asyncio.create_task(adapter.connect({}))
        await candidate.connect_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert adapter.room is None
        assert await adapter.drain_teardown() is True

    run(scenario())
    assert candidate.disconnect_calls == 1


def test_hung_candidate_cleanup_blocks_reconnect_without_hanging_cancellation(monkeypatch):
    install_fake_publishers(monkeypatch)
    candidate = FakeRoom()

    async def scenario():
        candidate.connect_release = asyncio.Event()
        candidate.disconnect_release = asyncio.Event()
        monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: candidate)
        adapter = LiveKitEgressAdapter(
            room_url="wss://room.example.test",
            room_token="synthetic",
        )
        adapter._TEARDOWN_WAIT_SECONDS = 0.01

        task = asyncio.create_task(adapter.connect({}))
        await candidate.connect_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.1)

        assert adapter._teardown_task is not None
        with pytest.raises(RuntimeError, match="Previous LiveKit teardown"):
            await adapter.connect({})

        candidate.disconnect_release.set()
        assert await adapter.drain_teardown() is True

    run(scenario())


def test_concurrent_disconnect_callers_share_one_teardown(monkeypatch):
    install_fake_publishers(monkeypatch)
    room = FakeRoom()

    async def scenario():
        adapter = LiveKitEgressAdapter()
        await adapter.connect({"room": room})
        video_publisher = adapter.video_publisher
        close_started = asyncio.Event()
        close_release = asyncio.Event()

        async def slow_close():
            close_started.set()
            await close_release.wait()

        video_publisher.aclose.side_effect = slow_close
        first = asyncio.create_task(adapter.disconnect())
        await close_started.wait()
        second = asyncio.create_task(adapter.disconnect())
        await asyncio.sleep(0)
        close_release.set()
        await asyncio.gather(first, second)
        return adapter, video_publisher

    adapter, video_publisher = run(scenario())
    video_publisher.aclose.assert_awaited_once()
    assert room.disconnect_calls == 0
    assert adapter._teardown_task is None


def test_teardown_error_resets_state_and_allows_reconnect(monkeypatch):
    install_fake_publishers(monkeypatch)
    first = FakeRoom(disconnect_error=RuntimeError("synthetic disconnect failure"))
    second = FakeRoom()
    candidates = iter((first, second))
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: next(candidates))

    async def scenario():
        adapter = LiveKitEgressAdapter(
            room_url="wss://room.example.test",
            room_token="synthetic",
        )
        await adapter.connect({})
        await adapter.disconnect()
        assert adapter._teardown_task is None
        await adapter.connect({})
        assert adapter.room is second
        await adapter.disconnect()

    run(scenario())
    assert first.disconnect_calls == 1
    assert second.disconnect_calls == 1


class DiagRoom(FakeRoom):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handlers = {}
        self.subscribed_events = []

    def on(self, event_name, callback=None):
        self.subscribed_events.append(event_name)
        if callback is not None:
            self.handlers[event_name] = callback
            return callback
        return lambda fn: (self.handlers.setdefault(event_name, fn))


def test_connect_logs_attempt_outcomes(monkeypatch, caplog):
    install_fake_publishers(monkeypatch)
    first = FakeRoom(connect_error=ConnectionError("synthetic failure"))
    second = DiagRoom()
    candidates = iter((first, second))
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: next(candidates))
    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test",
        room_token="synthetic",
        session_id="diag-session",
    )
    adapter._CONNECT_RETRY_BASE_SECONDS = 0

    with caplog.at_level(logging.INFO, logger="LiveKitEgressAdapter"):
        run(adapter.connect({}))

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "LIVEKIT_CONNECT_ATTEMPT_FAILED" in message
        and "session=diag-session" in message
        and "attempt=1/3" in message
        and "error=synthetic failure" in message
        for message in messages
    )
    assert any(
        "LIVEKIT_CONNECT_SUCCEEDED" in message
        and "session=diag-session" in message
        and "attempt=2/3" in message
        for message in messages
    )


def test_room_diag_subscribes_and_logs_connection_events(monkeypatch, caplog):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test",
        room_token="synthetic",
        session_id="diag-session",
    )

    run(adapter.connect({}))

    assert set(room.handlers) == {
        "connection_state_changed",
        "disconnected",
        "reconnecting",
        "reconnected",
    }

    with caplog.at_level(logging.INFO, logger="LiveKitEgressAdapter"):
        room.handlers["connection_state_changed"]("CONN_CONNECTED")
        room.handlers["disconnected"]("PARTICIPANT_REMOVED")
        room.handlers["reconnecting"]()
        room.handlers["reconnected"]()

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "LIVEKIT_ROOM_STATE" in message
        and "session=diag-session" in message
        and "state=CONN_CONNECTED" in message
        for message in messages
    )
    assert any(
        "LIVEKIT_ROOM_DISCONNECTED" in message
        and "session=diag-session" in message
        and "reason=PARTICIPANT_REMOVED" in message
        for message in messages
    )
    assert any("LIVEKIT_ROOM_RECONNECTING" in message for message in messages)
    assert any("LIVEKIT_ROOM_RECONNECTED" in message for message in messages)


def install_fake_net_probe(monkeypatch, outcomes, *, calls=None, targets=None):
    if calls is None:
        calls = []
    if targets is None:
        targets = []

    def fake_udp_egress_sendable(host, port):
        calls.append(len(calls))
        targets.append((host, port))
        return outcomes[min(len(calls) - 1, len(outcomes) - 1)]

    monkeypatch.setattr(livekit_egress, "_udp_egress_sendable", fake_udp_egress_sendable)
    monkeypatch.setenv("LIVEKIT_NET_PROBE_ENABLED", "true")
    monkeypatch.setenv("LIVEKIT_NET_PROBE_INTERVAL_SECONDS", "0.01")
    return calls, targets


def install_fake_connect_options(monkeypatch):
    room_options_calls = []
    rtc_config_calls = []

    def fake_room_options(**kwargs):
        room_options_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    def fake_rtc_configuration(**kwargs):
        rtc_config_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(livekit_egress.rtc, "RoomOptions", fake_room_options)
    monkeypatch.setattr(livekit_egress.rtc, "RtcConfiguration", fake_rtc_configuration)
    return room_options_calls, rtc_config_calls


def test_probe_selects_relay_first_when_udp_is_silent_and_tcp_443_works(
    monkeypatch, caplog
):
    # Given: resolvable UDP targets return no STUN response while TCP 443 connects.
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    install_fake_net_probe(monkeypatch, [(False, "response timed out", True)])
    tcp_calls = []
    tcp_socket = MagicMock()
    tcp_socket.__enter__ = Mock(return_value=tcp_socket)
    tcp_socket.__exit__ = Mock(return_value=False)
    tcp_socket.family = socket.AF_INET

    def fake_create_connection(address, timeout):
        tcp_calls.append((address, timeout))
        return tcp_socket

    monkeypatch.setattr(livekit_egress.socket, "create_connection", fake_create_connection)
    room_options_calls, rtc_config_calls = install_fake_connect_options(monkeypatch)
    adapter = LiveKitEgressAdapter(
        room_url="wss://ai-14ksodzi.livekit.cloud/rtc/v1",
        room_token="synthetic",
        session_id="relay-first-session",
    )

    # When: the adapter selects the path before its first room connection.
    with caplog.at_level(logging.INFO, logger="LiveKitEgressAdapter"):
        run(adapter.connect({}))

    # Then: attempt 1 is relay-only using server-provided ICE credentials.
    assert tcp_calls == [(('ai-14ksodzi.turn.livekit.cloud', 443), 1.0)]
    assert rtc_config_calls == [
        {"ice_transport_type": livekit_egress._ICE_TRANSPORT_RELAY}
    ]
    assert "ice_servers" not in rtc_config_calls[0]
    assert room_options_calls[0]["auto_subscribe"] is False
    assert room_options_calls[0]["single_peer_connection"] is True
    assert room_options_calls[0]["connect_timeout"] == 7.0
    assert any(
        "LIVEKIT_RELAY_FIRST" in record.getMessage()
        and "session=relay-first-session" in record.getMessage()
        and "reason=udp_silent_tcp_443_ready" in record.getMessage()
        for record in caplog.records
    )


def test_probe_keeps_direct_first_when_udp_responds(monkeypatch):
    # Given: the first UDP target returns a STUN response.
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    install_fake_net_probe(monkeypatch, [(True, None, True)])
    tcp_connect = Mock()
    monkeypatch.setattr(livekit_egress.socket, "create_connection", tcp_connect)
    room_options_calls, rtc_config_calls = install_fake_connect_options(monkeypatch)
    adapter = LiveKitEgressAdapter(
        room_url="wss://ai-14ksodzi.livekit.cloud/rtc/v1",
        room_token="synthetic",
    )

    # When: the adapter selects the path before its first room connection.
    run(adapter.connect({}))

    # Then: attempt 1 remains direct and TCP probing is unnecessary.
    tcp_connect.assert_not_called()
    assert rtc_config_calls == []
    assert "rtc_config" not in room_options_calls[0]


def test_probe_keeps_direct_first_when_udp_and_tcp_both_fail(monkeypatch, caplog):
    # Given: UDP is silent and TCP 443 cannot connect on either target.
    # Deep-blanket correction: still goes RELAY (TURN/TLS chance) instead of
    # burning 15s on ALL. Kill switch LIVEKIT_DEEP_BLANKET_RELAY=0 restores direct.
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    install_fake_net_probe(monkeypatch, [(False, "response timed out", True)])
    tcp_calls = []

    def fake_create_connection(address, timeout):
        tcp_calls.append((address, timeout))
        raise OSError(101, "Network is unreachable")

    monkeypatch.setattr(livekit_egress.socket, "create_connection", fake_create_connection)
    room_options_calls, rtc_config_calls = install_fake_connect_options(monkeypatch)
    adapter = LiveKitEgressAdapter(
        room_url="wss://ai-14ksodzi.livekit.cloud/rtc/v1",
        room_token="synthetic",
    )

    # When: the adapter selects the path before its first room connection.
    with caplog.at_level(logging.INFO, logger="LiveKitEgressAdapter"):
        run(adapter.connect({}))

    # Then: it proceeds via RELAY as the deep-blanket policy.
    assert tcp_calls == [
        (("ai-14ksodzi.turn.livekit.cloud", 443), 1.0),
        (("ai-14ksodzi.livekit.cloud", 443), 1.0),
    ]
    assert rtc_config_calls == [
        {"ice_transport_type": livekit_egress._ICE_TRANSPORT_RELAY}
    ]
    assert "rtc_config" in room_options_calls[0]
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "LIVEKIT_NET_PROBE_TIMEOUT" in message and "proceeding" in message
        for message in messages
    )
    assert any("LIVEKIT_RELAY_FIRST" in message and "deep_blanket" in message for message in messages)


def test_deep_blanket_relay_kill_switch_restores_direct(monkeypatch):
    # Given: deep-blanket branch with the kill switch off.
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    install_fake_net_probe(monkeypatch, [(False, "response timed out", True)])
    monkeypatch.setattr(
        livekit_egress.socket,
        "create_connection",
        lambda address, timeout: (_ for _ in ()).throw(OSError(101, "Network is unreachable")),
    )
    monkeypatch.setenv("LIVEKIT_DEEP_BLANKET_RELAY", "0")
    room_options_calls, rtc_config_calls = install_fake_connect_options(monkeypatch)
    adapter = LiveKitEgressAdapter(
        room_url="wss://ai-14ksodzi.livekit.cloud/rtc/v1",
        room_token="synthetic",
    )

    run(adapter.connect({}))

    assert rtc_config_calls == []
    assert "rtc_config" not in room_options_calls[0]


def test_net_probe_blocked_tcp_ready_selects_relay(monkeypatch, caplog):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    calls, targets = install_fake_net_probe(
        monkeypatch,
        [
            (False, "OSError: [Errno 101] Network is unreachable", True),
            (False, "OSError: [Errno 101] Network is unreachable", True),
        ],
    )
    tcp_socket = MagicMock()
    tcp_socket.__enter__ = Mock(return_value=tcp_socket)
    tcp_socket.__exit__ = Mock(return_value=False)
    tcp_socket.family = socket.AF_INET
    monkeypatch.setattr(
        livekit_egress.socket,
        "create_connection",
        lambda address, timeout: tcp_socket,
    )
    adapter = LiveKitEgressAdapter(
        room_url="wss://ai-14ksodzi.livekit.cloud/rtc/v1",
        room_token="synthetic",
        session_id="probe-session",
    )

    with caplog.at_level(logging.INFO, logger="LiveKitEgressAdapter"):
        run(adapter.connect({}))

    assert len(calls) == 2
    assert targets[0] == ("ai-14ksodzi.turn.livekit.cloud", 3478)
    assert targets[1] == ("ai-14ksodzi.livekit.cloud", 3478)
    assert room.connect_calls == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "LIVEKIT_NET_PROBE_BLOCKED" in message
        and "session=probe-session" in message
        and "error=OSError: [Errno 101] Network is unreachable" in message
        for message in messages
    )
    assert any("LIVEKIT_RELAY_FIRST" in message for message in messages)


def test_net_probe_immediately_ready_connects_without_warning(monkeypatch, caplog):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    # First target (TURN host) succeeds immediately, so only 1 call.
    calls, _targets = install_fake_net_probe(monkeypatch, [(True, None, True)])
    adapter = LiveKitEgressAdapter(
        room_url="wss://ai-14ksodzi.livekit.cloud/rtc/v1",
        room_token="synthetic",
        session_id="probe-session",
    )

    with caplog.at_level(logging.DEBUG, logger="LiveKitEgressAdapter"):
        run(adapter.connect({}))

    assert len(calls) == 1
    assert room.connect_calls == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("LIVEKIT_NET_PROBE_OK" in message for message in messages)
    assert not any("LIVEKIT_NET_PROBE_BLOCKED" in message for message in messages)


def test_net_probe_timeout_proceeds_with_connect(monkeypatch, caplog):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    calls, _targets = install_fake_net_probe(
        monkeypatch,
        [(False, "OSError: [Errno 101] Network is unreachable", True)],
    )
    monkeypatch.setenv("LIVEKIT_NET_PROBE_TCP_TIMEOUT_SECONDS", "0.05")

    def blocked_tcp(_address, timeout):
        raise OSError(101, "Network is unreachable")

    monkeypatch.setattr(livekit_egress.socket, "create_connection", blocked_tcp)
    adapter = LiveKitEgressAdapter(
        room_url="wss://ai-14ksodzi.livekit.cloud/rtc/v1",
        room_token="synthetic",
        session_id="probe-session",
    )

    with caplog.at_level(logging.INFO, logger="LiveKitEgressAdapter"):
        run(adapter.connect({}))

    # 2 targets probed per iteration, so at least 2 calls.
    assert len(calls) >= 2
    assert room.connect_calls == 1
    assert adapter.is_connected is True
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "LIVEKIT_NET_PROBE_TIMEOUT" in message
        and "session=probe-session" in message
        and "proceeding" in message
        for message in messages
    )
    # Deep-blanket policy: TIMEOUT still goes RELAY.
    assert any(
        "LIVEKIT_RELAY_FIRST" in message and "deep_blanket" in message
        for message in messages
    )


def test_net_probe_disabled_skips_probe(monkeypatch, caplog):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    probe = Mock()
    monkeypatch.setattr(livekit_egress, "_udp_egress_sendable", probe)
    monkeypatch.setenv("LIVEKIT_NET_PROBE_ENABLED", "false")
    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test",
        room_token="synthetic",
        session_id="probe-session",
    )

    with caplog.at_level(logging.INFO, logger="LiveKitEgressAdapter"):
        run(adapter.connect({}))

    probe.assert_not_called()
    assert room.connect_calls == 1


def test_net_probe_env_host_override(monkeypatch):
    monkeypatch.setenv("LIVEKIT_NET_PROBE_HOST", "probe.example.test")
    assert livekit_egress._derive_net_probe_host("wss://ai-14ksodzi.livekit.cloud/rtc/v1") == "probe.example.test"
    monkeypatch.delenv("LIVEKIT_NET_PROBE_HOST")
    assert livekit_egress._derive_net_probe_host("wss://ai-14ksodzi.livekit.cloud/rtc/v1") == "ai-14ksodzi.turn.livekit.cloud"
    assert livekit_egress._derive_net_probe_host("wss://otoronto1a.turn.livekit.cloud/rtc/v1") == "otoronto1a.turn.livekit.cloud"
    assert livekit_egress._derive_net_probe_host("wss://livekit.internal.corp/rtc/v1") == "livekit.internal.corp"


def test_net_probe_unprobeable_proceeds_immediately(monkeypatch, caplog):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    calls, _targets = install_fake_net_probe(
        monkeypatch,
        [
            (
                False,
                "resolve failed (gaierror: [Errno -2] Name or service not known)",
                False,
            ),
        ],
    )
    adapter = LiveKitEgressAdapter(
        room_url="wss://client-proj.livekit.cloud",
        room_token="synthetic",
        session_id="probe-session",
    )

    with caplog.at_level(logging.WARNING, logger="LiveKitEgressAdapter"):
        run(adapter.connect({}))

    # 2 targets probed (TURN host + signaling host), both unprobeable.
    assert len(calls) == 2
    assert room.connect_calls == 1
    assert adapter.is_connected is True
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "LIVEKIT_NET_PROBE_UNPROBEABLE" in message
        and "session=probe-session" in message
        and "proceeding" in message
        for message in messages
    )
    assert not any("LIVEKIT_NET_PROBE_BLOCKED" in message for message in messages)


def _make_mock_socket(sendto_side_effect, *, recvfrom_side_effect=None):
    m = MagicMock()
    m.__enter__ = Mock(return_value=m)
    m.__exit__ = Mock(return_value=False)
    m.sendto = Mock(side_effect=sendto_side_effect)
    if recvfrom_side_effect is None:
        m.recvfrom = Mock(
            return_value=(
                struct.pack("!HHI", 0x0101, 0, 0x2112A442) + b"\x00" * 12,
                ("127.0.0.1", 3478),
            )
        )
    else:
        m.recvfrom = Mock(side_effect=recvfrom_side_effect)
    return m


def test_udp_egress_sendable_ipv4_ready(monkeypatch):
    sent = []

    def fake_socket(family, socktype, proto=0):
        m = _make_mock_socket(
            lambda data, addr: sent.append((family, data, addr)) or None
        )
        return m

    monkeypatch.setattr(
        livekit_egress.socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0: [
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("127.0.0.1", port))
        ],
    )
    monkeypatch.setattr(livekit_egress.socket, "socket", fake_socket)

    sendable, error, can_probe = livekit_egress._udp_egress_sendable("host", 3478)
    assert sendable is True
    assert error is None
    assert can_probe is True
    assert len(sent) == 1
    assert sent[0][0] == socket.AF_INET
    assert sent[0][1] == livekit_egress._NET_PROBE_STUN_REQUEST
    assert sent[0][2] == ("127.0.0.1", 3478)


def test_udp_egress_sendable_ipv6_only_uses_ipv6(monkeypatch):
    sent = []

    def fake_socket(family, socktype, proto=0):
        return _make_mock_socket(
            lambda data, addr: sent.append((family, data, addr)) or None
        )

    monkeypatch.setattr(
        livekit_egress.socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0: [
            (socket.AF_INET6, socket.SOCK_DGRAM, 0, "", ("::1", port, 0, 0))
        ],
    )
    monkeypatch.setattr(livekit_egress.socket, "socket", fake_socket)

    sendable, error, can_probe = livekit_egress._udp_egress_sendable("host", 3478)
    assert sendable is True
    assert error is None
    assert can_probe is True
    assert len(sent) == 1
    assert sent[0][0] == socket.AF_INET6
    assert sent[0][1] == livekit_egress._NET_PROBE_STUN_REQUEST


def test_udp_egress_sendable_ipv4_fails_back_to_ipv6(monkeypatch):
    sent = []

    def fake_socket(family, socktype, proto=0):
        if family == socket.AF_INET:
            return _make_mock_socket(OSError(101, "Network is unreachable"))
        return _make_mock_socket(
            lambda data, addr: sent.append((family, data, addr)) or None
        )

    monkeypatch.setattr(
        livekit_egress.socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0: [
            (socket.AF_INET6, socket.SOCK_DGRAM, 0, "", ("::1", port, 0, 0)),
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("127.0.0.1", port)),
        ],
    )
    monkeypatch.setattr(livekit_egress.socket, "socket", fake_socket)

    sendable, error, can_probe = livekit_egress._udp_egress_sendable("host", 3478)
    assert sendable is True
    assert error is None
    assert can_probe is True
    assert len(sent) == 1
    assert sent[0][0] == socket.AF_INET6


def test_udp_egress_sendable_dns_failure_unprobeable(monkeypatch):
    def fake_getaddrinfo(*_args, **_kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(livekit_egress.socket, "getaddrinfo", fake_getaddrinfo)
    sendable, error, can_probe = livekit_egress._udp_egress_sendable(
        "no-such-host-xyz.example", 3478
    )
    assert sendable is False
    assert "resolve failed" in error
    assert "gaierror" in error
    assert can_probe is False


def test_udp_egress_sendable_blocked_when_all_families_fail(monkeypatch):
    def fake_socket(family, socktype, proto=0):
        return _make_mock_socket(OSError(101, "Network is unreachable"))

    monkeypatch.setattr(
        livekit_egress.socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0: [
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("127.0.0.1", port)),
            (socket.AF_INET6, socket.SOCK_DGRAM, 0, "", ("::1", port, 0, 0)),
        ],
    )
    monkeypatch.setattr(livekit_egress.socket, "socket", fake_socket)

    sendable, error, can_probe = livekit_egress._udp_egress_sendable("host", 3478)
    assert sendable is False
    assert can_probe is True
    assert "Network is unreachable" in error


def test_udp_egress_sendable_requires_a_stun_response(monkeypatch):
    # Given: a UDP send succeeds but no response arrives before the socket timeout.
    probe_socket = _make_mock_socket(None, recvfrom_side_effect=socket.timeout("timed out"))
    monkeypatch.setattr(
        livekit_egress.socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0: [
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("127.0.0.1", port))
        ],
    )
    monkeypatch.setattr(
        livekit_egress.socket, "socket", lambda family, socktype, proto=0: probe_socket
    )
    monkeypatch.setenv("LIVEKIT_NET_PROBE_RESPONSE_TIMEOUT_SECONDS", "0.25")

    # When: UDP readiness is checked.
    sendable, error, can_probe = livekit_egress._udp_egress_sendable("host", 3478)

    # Then: send success alone does not mark the path ready.
    assert sendable is False
    assert error is not None and "timed out" in error
    assert can_probe is True
    probe_socket.settimeout.assert_called_once_with(0.25)
    probe_socket.recvfrom.assert_called_once_with(2048)


# ---------------------------------------------------------------------------
# Multi-target probe tests (Option 3)
# ---------------------------------------------------------------------------

def test_derive_net_probe_signaling_host():
    assert (
        livekit_egress._derive_net_probe_signaling_host(
            "wss://ai-14ksodzi.livekit.cloud/rtc/v1"
        )
        == "ai-14ksodzi.livekit.cloud"
    )
    assert livekit_egress._derive_net_probe_signaling_host("not-a-url") is None


def test_probe_targets_first_success():
    calls = []

    def fake_sendable(host, port):
        calls.append((host, port))
        if host == "turn.example.test":
            return (False, "blocked", True)
        return (True, None, True)

    orig = livekit_egress._udp_egress_sendable
    livekit_egress._udp_egress_sendable = fake_sendable
    try:
        sendable, error, can_probe, target = livekit_egress._probe_targets(
            [("turn.example.test", 3478), ("signal.example.test", 3478)]
        )
    finally:
        livekit_egress._udp_egress_sendable = orig

    assert sendable is True
    assert error is None
    assert can_probe is True
    assert target == ("signal.example.test", 3478)
    assert len(calls) == 2


def test_probe_targets_all_fail():
    def fake_sendable(host, port):
        return (False, "blocked", True)

    orig = livekit_egress._udp_egress_sendable
    livekit_egress._udp_egress_sendable = fake_sendable
    try:
        sendable, error, can_probe, target = livekit_egress._probe_targets(
            [("turn.example.test", 3478), ("signal.example.test", 3478)]
        )
    finally:
        livekit_egress._udp_egress_sendable = orig

    assert sendable is False
    assert can_probe is True
    assert target is None


def test_probe_targets_all_unprobeable():
    def fake_sendable(host, port):
        return (False, "resolve failed", False)

    orig = livekit_egress._udp_egress_sendable
    livekit_egress._udp_egress_sendable = fake_sendable
    try:
        sendable, error, can_probe, target = livekit_egress._probe_targets(
            [("turn.example.test", 3478), ("signal.example.test", 3478)]
        )
    finally:
        livekit_egress._udp_egress_sendable = orig

    assert sendable is False
    assert can_probe is False
    assert target is None


def test_probe_targets_single_target_skips_secondary():
    """When TURN host and signaling host are the same, only 1 target is probed."""
    calls = []

    def fake_sendable(host, port):
        calls.append((host, port))
        return (True, None, True)

    orig = livekit_egress._udp_egress_sendable
    livekit_egress._udp_egress_sendable = fake_sendable
    try:
        sendable, _error, _can_probe, target = livekit_egress._probe_targets(
            [("same.host.test", 3478)]
        )
    finally:
        livekit_egress._udp_egress_sendable = orig

    assert sendable is True
    assert target == ("same.host.test", 3478)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# TURN relay fallback tests (Option 6)
# ---------------------------------------------------------------------------

def test_turn_relay_fallback_on_retry(monkeypatch, caplog):
    """Attempt 1 uses TRANSPORT_ALL, attempt 2+ uses TRANSPORT_RELAY."""
    install_fake_publishers(monkeypatch)
    first = FakeRoom(connect_error=ConnectionError("wait_pc_connection timed out"))
    second = DiagRoom()
    candidates = iter((first, second))
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: next(candidates))

    # Bypass the probe gate.
    monkeypatch.setenv("LIVEKIT_NET_PROBE_ENABLED", "false")

    # Track connect options.
    connect_options = []
    orig_connect = second.connect

    async def tracking_connect(*args, **kwargs):
        connect_options.append(kwargs.get("options"))
        return await orig_connect(*args, **kwargs)

    second.connect = tracking_connect

    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test",
        room_token="synthetic",
        session_id="relay-session",
    )
    adapter._CONNECT_RETRY_BASE_SECONDS = 0

    with caplog.at_level(logging.INFO, logger="LiveKitEgressAdapter"):
        run(adapter.connect({}))

    # First room failed, second succeeded.
    assert first.connect_calls == 1
    assert second.connect_calls == 1
    # The second attempt should have used TRANSPORT_RELAY.
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "LIVEKIT_CONNECT_RELAY_FALLBACK" in message
        and "session=relay-session" in message
        for message in messages
    )
    # The second connect should have rtc_config with relay policy.
    assert len(connect_options) == 1
    opts = connect_options[0]
    if opts is not None and hasattr(opts, "rtc_config") and opts.rtc_config is not None:
        assert opts.rtc_config.ice_transport_type == livekit_egress._ICE_TRANSPORT_RELAY


def test_first_attempt_uses_transport_all(monkeypatch):
    """Attempt 1 should NOT use relay fallback."""
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    monkeypatch.setenv("LIVEKIT_NET_PROBE_ENABLED", "false")

    connect_options = []
    orig_connect = room.connect

    async def tracking_connect(*args, **kwargs):
        connect_options.append(kwargs.get("options"))
        return await orig_connect(*args, **kwargs)

    room.connect = tracking_connect

    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test",
        room_token="synthetic",
    )
    run(adapter.connect({}))

    assert room.connect_calls == 1
    assert len(connect_options) == 1
    opts = connect_options[0]
    # First attempt should NOT have rtc_config set (TRANSPORT_ALL is the default).
    if opts is not None and hasattr(opts, "rtc_config"):
        assert opts.rtc_config is None


# ---------------------------------------------------------------------------
# Netstack priming tests (Option 1)
# ---------------------------------------------------------------------------

def test_prime_netstack_success(monkeypatch, capsys):
    """General netstack priming sends a UDP packet and logs success."""
    from modal_worker.worker_base import WorkerBase

    sent = []

    def fake_socket(family, socktype, proto=0):
        m = MagicMock()
        m.__enter__ = Mock(return_value=m)
        m.__exit__ = Mock(return_value=False)
        m.sendto = Mock(side_effect=lambda data, addr: sent.append((family, data, addr)))
        return m

    monkeypatch.setattr(
        livekit_egress.socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0: [
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("1.1.1.1", port))
        ],
    )

    import modal_worker.worker_base as wbm
    monkeypatch.setattr(wbm.socket, "socket", fake_socket)

    worker = WorkerBase.__new__(WorkerBase)
    worker._prime_netstack()

    captured = capsys.readouterr()
    assert "Netstack prime OK" in captured.out
    assert len(sent) == 1
    assert sent[0][0] == socket.AF_INET


def test_prime_netstack_failure_does_not_raise(monkeypatch, capsys):
    """Netstack priming failure is logged but never blocks startup."""
    import modal_worker.worker_base as wbm

    def fake_socket(family, socktype, proto=0):
        m = MagicMock()
        m.__enter__ = Mock(return_value=m)
        m.__exit__ = Mock(return_value=False)
        m.sendto = Mock(side_effect=OSError(101, "Network is unreachable"))
        return m

    monkeypatch.setattr(
        wbm.socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0: [
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("1.1.1.1", port))
        ],
    )
    monkeypatch.setattr(wbm.socket, "socket", fake_socket)

    from modal_worker.worker_base import WorkerBase
    worker = WorkerBase.__new__(WorkerBase)
    # Should not raise.
    worker._prime_netstack()

    captured = capsys.readouterr()
    assert "Netstack prime FAILED" in captured.out
    assert "Network is unreachable" in captured.out


def test_prime_netstack_env_override(monkeypatch, capsys):
    """LIVEKIT_NET_PRIME_HOST overrides the default priming target."""
    import modal_worker.worker_base as wbm

    monkeypatch.setenv("LIVEKIT_NET_PRIME_HOST", "8.8.8.8")
    monkeypatch.setenv("LIVEKIT_NET_PRIME_PORT", "53")

    sent = []

    def fake_socket(family, socktype, proto=0):
        m = MagicMock()
        m.__enter__ = Mock(return_value=m)
        m.__exit__ = Mock(return_value=False)
        m.sendto = Mock(side_effect=lambda data, addr: sent.append((family, data, addr)))
        return m

    monkeypatch.setattr(
        wbm.socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0: [
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", (host, port))
        ],
    )
    monkeypatch.setattr(wbm.socket, "socket", fake_socket)

    from modal_worker.worker_base import WorkerBase
    worker = WorkerBase.__new__(WorkerBase)
    worker._prime_netstack()

    captured = capsys.readouterr()
    assert "8.8.8.8" in captured.out
    assert len(sent) == 1


def test_prime_netstack_dns_failure_proceeds(monkeypatch, capsys):
    """DNS failure during priming is logged but never blocks startup."""
    import modal_worker.worker_base as wbm

    def fake_getaddrinfo(*_args, **_kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(wbm.socket, "getaddrinfo", fake_getaddrinfo)

    from modal_worker.worker_base import WorkerBase
    worker = WorkerBase.__new__(WorkerBase)
    # Should not raise.
    worker._prime_netstack()

    captured = capsys.readouterr()
    assert "Netstack prime" in captured.out


# ---------------------------------------------------------------------------
# Deep-blanket relay + retry-budget surgery + warming + heartbeat (Sep 6)
# ---------------------------------------------------------------------------

def test_is_ice_timeout_error_classifies_burns():
    assert livekit_egress._is_ice_timeout_error(ConnectionError("wait_pc_connection timed out")) is True
    assert livekit_egress._is_ice_timeout_error(TimeoutError("ICE connect timed out")) is True
    assert livekit_egress._is_ice_timeout_error(ConnectionError("synthetic failure")) is False


def test_connect_timeout_env_override(monkeypatch):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    monkeypatch.setenv("LIVEKIT_NET_PROBE_ENABLED", "false")
    monkeypatch.setenv("LIVEKIT_CONNECT_TIMEOUT_SECONDS", "5.5")
    room_options_calls, _rtc_config_calls = install_fake_connect_options(monkeypatch)
    adapter = LiveKitEgressAdapter(room_url="wss://room.example.test", room_token="synthetic")
    run(adapter.connect({}))
    assert room_options_calls[0]["connect_timeout"] == 5.5


def test_ice_timeout_uses_fast_backoff(monkeypatch):
    install_fake_publishers(monkeypatch)
    first = FakeRoom(connect_error=ConnectionError("wait_pc_connection timed out"))
    second = DiagRoom()
    candidates = iter((first, second))
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: next(candidates))
    monkeypatch.setenv("LIVEKIT_NET_PROBE_ENABLED", "false")
    monkeypatch.setenv("LIVEKIT_ICEFAIL_BACKOFF_S", "0.01")
    monkeypatch.setenv("LIVEKIT_ICEFAIL_TEARDOWN_S", "0.2")
    sleeps = []
    orig_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleeps.append(delay)
        await orig_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test", room_token="synthetic", session_id="fast"
    )
    adapter._CONNECT_RETRY_BASE_SECONDS = 5.0
    run(adapter.connect({}))
    assert second.connect_calls == 1
    assert sleeps == [0.01]


def test_non_ice_error_keeps_slow_backoff(monkeypatch):
    install_fake_publishers(monkeypatch)
    first = FakeRoom(connect_error=ConnectionError("synthetic failure"))
    second = DiagRoom()
    candidates = iter((first, second))
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: next(candidates))
    monkeypatch.setenv("LIVEKIT_NET_PROBE_ENABLED", "false")
    monkeypatch.setenv("LIVEKIT_ICEFAIL_BACKOFF_S", "0.01")
    sleeps = []
    orig_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleeps.append(delay)
        await orig_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test", room_token="synthetic", session_id="slow"
    )
    adapter._CONNECT_RETRY_BASE_SECONDS = 5.0
    run(adapter.connect({}))
    assert sleeps == [5.0]


def test_warm_egress_targets_never_raises(monkeypatch):
    monkeypatch.setattr(
        livekit_egress.socket,
        "getaddrinfo",
        lambda *a, **k: (_ for _ in ()).throw(socket.gaierror(-2, "nope")),
    )
    livekit_egress._warm_egress_targets(["nope.example.test"], 3478)
    monkeypatch.setattr(
        livekit_egress.socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0, proto=0, flags=0: [
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("127.0.0.1", port))
        ],
    )
    sent = []

    def fake_socket(family, socktype, proto=0):
        m = MagicMock()
        m.__enter__ = Mock(return_value=m)
        m.__exit__ = Mock(return_value=False)
        m.sendto = Mock(side_effect=lambda data, addr: sent.append(addr))
        return m

    monkeypatch.setattr(livekit_egress.socket, "socket", fake_socket)
    livekit_egress._warm_egress_targets(["h1.test", "h1.test", "h2.test"], 3478)
    assert len(sent) >= 2


def test_egress_warm_disabled_skips_warm(monkeypatch):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    install_fake_net_probe(monkeypatch, [(True, None, True)])
    monkeypatch.setenv("LIVEKIT_EGRESS_WARM_ENABLED", "false")
    warmed = []
    monkeypatch.setattr(livekit_egress, "_warm_egress_targets", lambda hosts, port, **k: warmed.append(hosts))
    adapter = LiveKitEgressAdapter(room_url="wss://room.example.test", room_token="synthetic")
    run(adapter.connect({}))
    assert warmed == []


def test_udp_heartbeat_start_stop(capsys):
    from modal_worker.worker_base import WorkerBase

    worker = WorkerBase.__new__(WorkerBase)
    worker._udp_heartbeat_thread = None
    worker._udp_heartbeat_stop = None
    import os

    os.environ["LIVEKIT_HEARTBEAT_INTERVAL_S"] = "0.05"
    os.environ["LIVEKIT_HEARTBEAT_ENABLED"] = "true"
    try:
        worker.start_udp_heartbeat()
        assert worker._udp_heartbeat_thread is not None
        worker._udp_heartbeat_thread.join(timeout=2)
        health = worker.get_nethealth()
        assert "lastUdpOkEpoch" in health
        assert "consecutiveFailures" in health
    finally:
        worker.stop_udp_heartbeat()
        os.environ.pop("LIVEKIT_HEARTBEAT_INTERVAL_S", None)
        os.environ.pop("LIVEKIT_HEARTBEAT_ENABLED", None)


# ---------------------------------------------------------------------------
# Early warm + probe overlap + recent-hosts (Sep 6 #1+#3+slim #2)
# ---------------------------------------------------------------------------

def test_warm_egress_for_url_derives_hosts(monkeypatch):
    captured = {}

    def fake_warm(hosts, port, **kwargs):
        captured["hosts"] = list(hosts)
        captured["port"] = port

    monkeypatch.setattr(livekit_egress, "_warm_egress_targets", fake_warm)
    livekit_egress.warm_egress_for_url("wss://ai-14ksodzi.livekit.cloud/rtc/v1")
    assert captured["hosts"][0] == "ai-14ksodzi.turn.livekit.cloud"
    assert "ai-14ksodzi.livekit.cloud" in captured["hosts"]
    assert captured["port"] == 3478


def test_warm_egress_for_url_respects_disable(monkeypatch):
    called = []
    monkeypatch.setattr(
        livekit_egress, "_warm_egress_targets", lambda hosts, port, **k: called.append(1)
    )
    monkeypatch.setenv("LIVEKIT_EGRESS_WARM_ENABLED", "false")
    livekit_egress.warm_egress_for_url("wss://ai-14ksodzi.livekit.cloud/rtc/v1")
    assert called == []


def test_warm_egress_for_url_never_raises():
    livekit_egress.warm_egress_for_url(None)
    livekit_egress.warm_egress_for_url("")
    livekit_egress.warm_egress_for_url("not-a-url")


def test_warm_egress_for_url_no_hardcoded_turn(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        livekit_egress,
        "_warm_egress_targets",
        lambda hosts, port, **k: captured.update(hosts=list(hosts)),
    )
    livekit_egress.warm_egress_for_url("wss://cust.example.com/rtc")
    assert captured["hosts"] == ["cust.example.com"]
    import inspect

    src = inspect.getsource(livekit_egress.warm_egress_for_url)
    assert "turn.livekit.cloud" not in src
    assert "ai-14ksodzi" not in src


def test_recent_hosts_cap_dedupe_lru():
    import importlib

    mod = livekit_egress
    with mod._recent_egress_lock:
        mod._recent_egress_hosts.clear()
    for i in range(6):
        mod.record_egress_hosts([(f"h{i}.test", 3478)])
    hosts = mod.get_recent_egress_hosts()
    assert len(hosts) == 5
    assert hosts[0] == ("h5.test", 3478)
    mod.record_egress_hosts([("h3.test", 3478)])
    assert mod.get_recent_egress_hosts()[0] == ("h3.test", 3478)
    assert len(mod.get_recent_egress_hosts()) == 5
    with mod._recent_egress_lock:
        mod._recent_egress_hosts.clear()


def test_probe_records_recent_hosts_even_on_failure(monkeypatch):
    with livekit_egress._recent_egress_lock:
        livekit_egress._recent_egress_hosts.clear()
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    install_fake_net_probe(monkeypatch, [(False, "response timed out", True)])
    monkeypatch.setattr(
        livekit_egress.socket,
        "create_connection",
        lambda address, timeout: (_ for _ in ()).throw(OSError(101, "nope")),
    )
    monkeypatch.setenv("LIVEKIT_DEEP_BLANKET_RELAY", "0")
    adapter = LiveKitEgressAdapter(
        room_url="wss://cust123.livekit.cloud/rtc/v1", room_token="synthetic"
    )
    run(adapter.connect({}))
    recent = livekit_egress.get_recent_egress_hosts()
    assert ("cust123.turn.livekit.cloud", 3478) in recent
    with livekit_egress._recent_egress_lock:
        livekit_egress._recent_egress_hosts.clear()


def test_start_probe_early_is_idempotent(monkeypatch):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    install_fake_net_probe(monkeypatch, [(True, None, True)])
    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test", room_token="synthetic"
    )

    async def scenario():
        t1 = adapter.start_probe_early()
        t2 = adapter.start_probe_early()
        assert t1 is t2
        await adapter.connect({})
        return t1

    run(scenario())
    assert room.connect_calls == 1
    run(adapter.disconnect())


def test_connect_joins_early_probe_without_reprobing(monkeypatch):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    calls, _targets = install_fake_net_probe(monkeypatch, [(True, None, True)])
    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test", room_token="synthetic"
    )

    async def scenario():
        adapter.start_probe_early()
        await adapter._probe_task
        before = len(calls)
        await adapter.connect({})
        return before

    before = run(scenario())
    assert before == 1
    assert len(calls) == 1
    run(adapter.disconnect())


def test_disconnect_cancels_inflight_probe(monkeypatch):
    install_fake_publishers(monkeypatch)
    room = DiagRoom()
    monkeypatch.setattr(livekit_egress.rtc, "Room", lambda: room)
    release = asyncio.Event()

    async def blocking_probe(self, room_url):
        await release.wait()
        return False, None

    monkeypatch.setattr(livekit_egress.LiveKitEgressAdapter, "_wait_for_udp_egress_ready", blocking_probe)
    adapter = LiveKitEgressAdapter(
        room_url="wss://room.example.test", room_token="synthetic"
    )

    async def scenario():
        adapter.start_probe_early()
        assert adapter._probe_task is not None
        await adapter.disconnect()
        assert adapter._probe_task is None
        release.set()

    run(scenario())


def test_nethealth_has_gated_fields():
    from modal_worker.worker_base import WorkerBase

    worker = WorkerBase.__new__(WorkerBase)
    worker._udp_heartbeat_thread = None
    worker._udp_heartbeat_stop = None
    import os

    os.environ["LIVEKIT_HEARTBEAT_INTERVAL_S"] = "0.05"
    os.environ["LIVEKIT_HEARTBEAT_ENABLED"] = "true"
    os.environ["LIVEKIT_HEARTBEAT_RESPONSE_CHECK_S"] = "0"
    try:
        worker.start_udp_heartbeat()
        worker._udp_heartbeat_thread.join(timeout=2)
        health = worker.get_nethealth()
        assert "recentHosts" in health
        assert "lastResponseGatedOkEpoch" in health
        assert "responseGatedFailures" in health
    finally:
        worker.stop_udp_heartbeat()
        for key in (
            "LIVEKIT_HEARTBEAT_INTERVAL_S",
            "LIVEKIT_HEARTBEAT_ENABLED",
            "LIVEKIT_HEARTBEAT_RESPONSE_CHECK_S",
        ):
            os.environ.pop(key, None)


def test_no_hardcoded_customer_host_in_prod():
    import pathlib as _pl

    root = _pl.Path(__file__).resolve().parent.parent
    for rel in ("modal_worker", "streaming", "modal_app.py"):
        base = root / rel
        files = [base] if base.is_file() else list(base.rglob("*.py"))
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            assert "ai-14ksodzi" not in text, f"hardcoded host in {f}"


# Shielded bounded-connect reverted Sep 9: outer shield caused overlapping
# native Rooms with same identity (DuplicateIdentity kick) and wasted
# background successes. Direct await restored. No shield tests.
