"""Daily egress adapter backed by daily-python virtual media devices."""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from streaming.core.audio_bus import normalize_channels, resample_audio
from streaming.core.session import MediaEgressAdapter
from streaming.core.upscale import get_output_size

try:
    daily = importlib.import_module("daily")
except ImportError:  # pragma: no cover - exercised in Modal image
    daily = None


_logger = logging.getLogger("DailyEgressAdapter")

_DAILY_DEVICE_LOCK = asyncio.Lock()

# Resolve the EventHandler base class from the daily module at import time.
# Falls back to object when daily-python is not installed (e.g. test envs),
# so the handler class can still be instantiated as a plain duck-typed object.
_EventHandlerBase = object
if daily is not None:
    _EventHandlerBase = getattr(daily, "EventHandler", object)


class _CallStateHandler(_EventHandlerBase):
    """Daily EventHandler that tracks call state for disconnect detection.

    The Daily SDK calls on_call_state_updated from its worker thread whenever
    the call transitions between states (initialized/joining/joined/leaving/left).
    We store the latest state so the asyncio publish loop can detect when the
    call has ended server-side and stop publishing instead of looping forever
    on a dead virtual camera device.
    """

    def __init__(self):
        super().__init__()
        self.call_state = "initialized"

    def on_call_state_updated(self, state: str) -> None:
        self.call_state = state
        if state == "left":
            _logger.warning(
                "Daily call state changed to 'left' - call has ended"
            )

    def on_error(self, message: str) -> None:
        _logger.warning("Daily call error: %s", message)


class DailyEgressAdapter(MediaEgressAdapter):
    """Join a Daily room and publish RGB video plus linear PCM audio."""

    def __init__(
        self,
        *,
        room_url: str | None = None,
        room_token: str | None = None,
        fps: int = 25,
        sample_rate: int = 16_000,
        channels: int = 1,
        session_id: str = "",
        width: int | None = None,
        height: int | None = None,
        join_timeout: float = 30.0,
    ):
        default_width, default_height = get_output_size()
        self.room_url = room_url
        self.room_token = room_token
        self.fps = int(fps)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.session_id = session_id
        self.width = int(width or default_width)
        self.height = int(height or default_height)
        self.join_timeout = float(join_timeout)
        self.client = None
        self.microphone = None
        self.camera = None
        self.video_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._connected = False
        self._device_lock_acquired = False
        self._call_state_handler: _CallStateHandler | None = None

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        # Detect server-side disconnection: if the Daily call state has
        # transitioned to "left" (e.g. kicked, room expired, network loss),
        # treat the adapter as disconnected so the publish loop exits.
        if self._call_state_handler is not None:
            if self._call_state_handler.call_state == "left":
                return False
        return True

    @property
    def first_frame_published(self):
        return self._ready

    async def connect(self, room_config: dict[str, Any] | Mapping[str, Any]) -> None:
        config = dict(room_config)
        self.room_url = config.get("room_url", config.get("url", self.room_url))
        self.room_token = config.get("room_token", config.get("token", self.room_token))
        self.fps = int(config.get("fps", self.fps))
        self.sample_rate = int(config.get("sample_rate", self.sample_rate))
        self.channels = int(config.get("channels", self.channels))
        self.session_id = str(config.get("session_id", self.session_id))
        self.width = int(config.get("width", self.width))
        self.height = int(config.get("height", self.height))
        if not self.room_url or not self.room_token:
            raise ValueError("Daily egress requires room_url and room_token")

        sdk = _daily_sdk()
        _initialize_daily(sdk)
        try:
            await asyncio.wait_for(_DAILY_DEVICE_LOCK.acquire(), timeout=self.join_timeout)
        except asyncio.TimeoutError as error:
            raise RuntimeError(
                "Daily virtual microphone is already active in this worker process"
            ) from error
        self._device_lock_acquired = True

        create_microphone = getattr(sdk, "create_microphone_device", None)
        create_camera = getattr(sdk, "create_camera_device", None)
        if create_microphone is None or create_camera is None:
            self._release_device_lock()
            raise RuntimeError("daily-python does not expose virtual media device factories")

        try:
            self.microphone = create_microphone(
                f"facemode-mic-{self.session_id}",
                sample_rate=self.sample_rate,
                channels=self.channels,
                non_blocking=True,
            )
            self.camera = create_camera(
                f"facemode-camera-{self.session_id}",
                self.width,
                self.height,
                color_format="RGB",
            )
            call_client_type = getattr(daily, "CallClient", None) or getattr(sdk, "CallClient", None)
            if call_client_type is None:
                raise RuntimeError("daily-python does not expose CallClient")
            self._call_state_handler = _CallStateHandler()
            try:
                self.client = call_client_type(event_handler=self._call_state_handler)
            except TypeError:
                # Older daily-python versions may not accept event_handler kwarg
                self.client = call_client_type()
            mic_name = getattr(self.microphone, "name", f"facemode-mic-{self.session_id}")
            camera_name = getattr(self.camera, "name", f"facemode-camera-{self.session_id}")
            client_settings = {
                "inputs": {
                    "microphone": {
                        "isEnabled": True,
                        "settings": {"deviceId": mic_name},
                    },
                    "camera": {
                        "isEnabled": True,
                        "settings": {"deviceId": camera_name},
                    },
                }
            }
            await _join_call_client(
                self.client,
                self.room_url,
                self.room_token,
                client_settings,
                timeout=self.join_timeout,
            )
        except Exception:
            if self.client is not None:
                release = getattr(self.client, "release", None)
                if release is not None:
                    release()
            self.client = None
            self.microphone = None
            self.camera = None
            self._release_device_lock()
            raise
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        if self.video_task is not None:
            self.video_task.cancel()
            try:
                await self.video_task
            except asyncio.CancelledError:
                pass
        self.video_task = None
        client = self.client
        self.client = None
        self._call_state_handler = None
        try:
            if client is not None:
                try:
                    await _leave_call_client(client, timeout=self.join_timeout)
                finally:
                    release = getattr(client, "release", None)
                    if release is not None:
                        release()
        finally:
            self.microphone = None
            self.camera = None
            self._release_device_lock()

    async def publish_video_frame(self, frame: np.ndarray, pts: float) -> None:
        self._require_connected()
        if not self.is_connected:
            _logger.warning(
                "Daily call has ended - dropping video frame for session=%s",
                self.session_id,
            )
            return
        if self.camera is None:
            raise RuntimeError("Daily camera device is not initialized")
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Daily video frames must be an HxWx3 RGB array")
        image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
        if image.shape[:2] != (self.height, self.width):
            image = _resize_rgb(image, self.width, self.height)
        writer = getattr(self.camera, "write_frame", None) or getattr(self.camera, "write_frames", None)
        if writer is None:
            raise RuntimeError("Daily camera device does not expose write_frame(s)")
        result = writer(image.tobytes())
        if asyncio.iscoroutine(result):
            await result
        self._ready.set()

    async def publish_audio_chunk(self, pcm: np.ndarray, sample_rate: int) -> None:
        self._require_connected()
        if not self.is_connected:
            _logger.warning(
                "Daily call has ended - dropping audio chunk for session=%s",
                self.session_id,
            )
            return
        if self.microphone is None:
            raise RuntimeError("Daily microphone device is not initialized")
        audio = np.asarray(pcm)
        source_channels = audio.shape[1] if audio.ndim == 2 else 1
        audio = normalize_channels(audio, source_channels, self.channels)
        audio = resample_audio(audio, int(sample_rate), self.sample_rate)
        pcm16 = _to_pcm16(audio)
        writer = getattr(self.microphone, "write_frames", None) or getattr(self.microphone, "write_frame", None)
        if writer is None:
            raise RuntimeError("Daily microphone device does not expose write_frames")
        result = writer(pcm16.tobytes())
        if asyncio.iscoroutine(result):
            await result

    async def wait_for_ready(self) -> None:
        self._require_connected()
        await self._ready.wait()

    def _release_device_lock(self) -> None:
        if self._device_lock_acquired:
            self._device_lock_acquired = False
            _DAILY_DEVICE_LOCK.release()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Daily egress is not connected")


def _daily_sdk():
    global daily
    if daily is None:
        try:
            daily = importlib.import_module("daily")
        except ImportError as error:  # pragma: no cover - depends on image
            raise RuntimeError("daily-python is required for Daily egress") from error
    return getattr(daily, "Daily", daily)


def _initialize_daily(sdk) -> None:
    initializer = getattr(sdk, "init", None) or getattr(sdk, "initialize", None)
    if initializer is not None:
        try:
            initializer()
        except RuntimeError as error:
            if "already" not in str(error).lower():
                raise


async def _join_call_client(client, url: str, token: str, settings: dict, *, timeout: float) -> None:
    loop = asyncio.get_running_loop()
    completion = loop.create_future()

    def done(_result=None, error=None):
        if completion.done():
            return
        if error:
            loop.call_soon_threadsafe(
                completion.set_exception, RuntimeError(str(error))
            )
        else:
            loop.call_soon_threadsafe(completion.set_result, _result)

    join = getattr(client, "join")
    try:
        result = join(url, token, client_settings=settings, completion=done)
    except TypeError:
        result = join(url, token, settings)
        if asyncio.iscoroutine(result):
            await result
        return
    if asyncio.iscoroutine(result):
        await result
    if not completion.done():
        await asyncio.wait_for(completion, timeout=timeout)


async def _leave_call_client(client, *, timeout: float) -> None:
    leave = getattr(client, "leave", None)
    if leave is None:
        return
    loop = asyncio.get_running_loop()
    completion = loop.create_future()

    def done(error=None):
        if not completion.done():
            if error:
                loop.call_soon_threadsafe(
                    completion.set_exception, RuntimeError(str(error))
                )
            else:
                loop.call_soon_threadsafe(completion.set_result, None)

    try:
        result = leave(completion=done)
    except TypeError:
        result = leave()
    if asyncio.iscoroutine(result):
        await result
    if not completion.done():
        try:
            await asyncio.wait_for(completion, timeout=timeout)
        except asyncio.TimeoutError:
            pass


def _to_pcm16(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.reshape(-1)
    return np.asarray(np.clip(values, -1.0, 1.0) * 32767, dtype="<i2")


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        y_indices = np.linspace(0, image.shape[0] - 1, height).astype(int)
        x_indices = np.linspace(0, image.shape[1] - 1, width).astype(int)
        return np.ascontiguousarray(image[y_indices][:, x_indices])
