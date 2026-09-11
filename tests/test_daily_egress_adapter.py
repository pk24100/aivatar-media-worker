import asyncio
from types import SimpleNamespace

import numpy as np

from streaming.transport.adapters import daily_egress
from streaming.transport.adapters.daily_egress import DailyEgressAdapter


def run(coroutine):
    return asyncio.run(coroutine)


class FakeDevice:
    def __init__(self, name):
        self.name = name
        self.frames = []

    def write_frames(self, payload):
        self.frames.append(payload)

    def write_frame(self, payload):
        self.frames.append(payload)


class FakeCallClient:
    def __init__(self, event_handler=None):
        self.event_handler = event_handler
        self.join_args = None
        self.leave_called = False
        self.released = False

    def join(self, url, token, client_settings=None, completion=None):
        self.join_args = (url, token, client_settings)
        if completion:
            completion({"joined": True}, None)

    def leave(self, completion=None):
        self.leave_called = True
        if completion:
            completion(None)

    def release(self):
        self.released = True


class FakeDaily:
    initialized = False
    microphone = None
    camera = None

    @staticmethod
    def init():
        FakeDaily.initialized = True

    @staticmethod
    def create_microphone_device(name, **kwargs):
        FakeDaily.microphone = FakeDevice(name)
        return FakeDaily.microphone

    @staticmethod
    def create_camera_device(name, width, height, color_format="RGBA"):
        FakeDaily.camera = FakeDevice(name)
        return FakeDaily.camera


def test_daily_adapter_joins_publishes_and_leaves(monkeypatch):
    client = FakeCallClient()

    def call_client_factory(event_handler=None):
        client.event_handler = event_handler
        return client

    monkeypatch.setattr(
        daily_egress,
        "daily",
        SimpleNamespace(Daily=FakeDaily, CallClient=call_client_factory),
    )

    adapter = DailyEgressAdapter(
        room_url="https://example.daily.co/room",
        room_token="daily-token",
        width=2,
        height=2,
        sample_rate=16_000,
        channels=1,
    )

    async def scenario():
        await adapter.connect({})
        await adapter.publish_video_frame(np.zeros((2, 2, 3), dtype=np.uint8), 1.0)
        await adapter.publish_audio_chunk(np.zeros(160, dtype=np.float32), 16_000)
        await adapter.wait_for_ready()
        await adapter.disconnect()

    run(scenario())

    assert adapter.is_connected is False
    assert client.join_args[0] == "https://example.daily.co/room"
    assert client.join_args[1] == "daily-token"
    assert client.leave_called is True
    assert client.released is True
    assert FakeDaily.camera.frames
    assert FakeDaily.microphone.frames
    assert client.join_args[2]["inputs"]["camera"]["settings"]["deviceId"] == FakeDaily.camera.name
    # Verify the event handler was passed to CallClient
    assert client.event_handler is not None
    assert hasattr(client.event_handler, "on_call_state_updated")


def test_daily_adapter_stops_publishing_when_call_ends(monkeypatch):
    """Verify that is_connected returns False when the call state is 'left'."""
    client = FakeCallClient()

    def call_client_factory(event_handler=None):
        client.event_handler = event_handler
        return client

    monkeypatch.setattr(
        daily_egress,
        "daily",
        SimpleNamespace(Daily=FakeDaily, CallClient=call_client_factory),
    )

    adapter = DailyEgressAdapter(
        room_url="https://example.daily.co/room",
        room_token="daily-token",
        width=2,
        height=2,
        sample_rate=16_000,
        channels=1,
    )

    async def scenario():
        await adapter.connect({})
        assert adapter.is_connected is True
        # Simulate server-side disconnection
        client.event_handler.on_call_state_updated("left")
        assert adapter.is_connected is False
        # publish_video_frame should silently drop instead of raising
        await adapter.publish_video_frame(np.zeros((2, 2, 3), dtype=np.uint8), 1.0)
        await adapter.publish_audio_chunk(np.zeros(160, dtype=np.float32), 16_000)
        await adapter.disconnect()

    run(scenario())
