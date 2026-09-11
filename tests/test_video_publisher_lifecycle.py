import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from livekit import rtc

from streaming.transport.livekit.video_publisher import VideoPublisher


def run(coroutine):
    return asyncio.run(coroutine)


class FakeRoom:
    def __init__(self):
        self.connection_state = rtc.ConnectionState.CONN_CONNECTED
        self.local_participant = SimpleNamespace(unpublish_track=AsyncMock())


class FakeVideoSource:
    def __init__(self):
        self.aclose = AsyncMock()


def install_published_track(publisher):
    source = FakeVideoSource()
    publisher.track = object()
    publisher.video_source = source
    publisher.publication = SimpleNamespace(sid="synthetic-track-sid")
    return source


def test_video_publisher_cleanup_unpublishes_without_disconnecting_room():
    room = FakeRoom()
    publisher = VideoPublisher(room)
    source = install_published_track(publisher)

    async def scenario():
        await publisher.aclose()
        await publisher.aclose()

    run(scenario())

    room.local_participant.unpublish_track.assert_awaited_once_with("synthetic-track-sid")
    source.aclose.assert_awaited_once()
    assert publisher.track is None
    assert publisher.video_source is None
    assert publisher.publication is None
    assert room.connection_state == rtc.ConnectionState.CONN_CONNECTED


def test_state_manager_cancellation_runs_publisher_cleanup():
    room = FakeRoom()
    publisher = VideoPublisher(room)
    source = install_published_track(publisher)
    frame_release = threading.Event()
    frame_started = threading.Event()

    class BlockingStateManager:
        def get_next_frame(self):
            frame_started.set()
            frame_release.wait(timeout=1)
            return None

    async def scenario():
        task = asyncio.create_task(
            publisher.publish_from_state_manager(BlockingStateManager())
        )
        await asyncio.to_thread(frame_started.wait, 1)
        task.cancel()
        frame_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())

    room.local_participant.unpublish_track.assert_awaited_once_with("synthetic-track-sid")
    source.aclose.assert_awaited_once()
