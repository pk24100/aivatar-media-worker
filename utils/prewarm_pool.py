"""Reference-only pre-warm room pool used by the stress-test wrapper."""

import asyncio
import time


PREWARM_ROOM_TIMEOUT = 60.0


class PrewarmRoomPool:
    def __init__(self, size=1):
        self.size = size
        self._queue = asyncio.Queue(maxsize=size)
        self._entries = {}
        self._claimed = {}

    async def add(self, room_name, room, participant_identity, worker_token):
        entry = {
            "roomName": room_name,
            "room": room,
            "participantIdentity": participant_identity,
            "workerToken": worker_token,
            "createdAt": time.monotonic(),
        }
        self._entries[room_name] = entry
        await self._queue.put(entry)

    def claim(self):
        try:
            entry = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        self._entries.pop(entry["roomName"], None)
        self._claimed[entry["roomName"]] = entry
        return entry

    def take_claimed(self, room_name):
        return self._claimed.pop(room_name, None)

    async def release(self, room_name):
        entry = self._claimed.pop(room_name, None)
        if entry is None:
            entry = self._entries.pop(room_name, None)
        if entry is None:
            return

        async def _bg_disconnect():
            try:
                await entry["room"].disconnect()
            except Exception:
                pass

        asyncio.create_task(_bg_disconnect())

    async def cleanup_expired(self):
        now = time.monotonic()
        expired = [
            name
            for name, entry in self._entries.items()
            if now - entry["createdAt"] > PREWARM_ROOM_TIMEOUT
        ]
        for name in expired:
            await self.release(name)

        stale = [
            name
            for name, entry in self._claimed.items()
            if now - entry["createdAt"] > PREWARM_ROOM_TIMEOUT
        ]
        for name in stale:
            await self.release(name)

    def available_count(self):
        return self._queue.qsize()
