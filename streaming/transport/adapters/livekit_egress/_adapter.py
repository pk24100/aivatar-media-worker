"""LiveKitEgressAdapter - Publish avatar media to a LiveKit room."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from livekit import rtc

from streaming.core.audio_bus import resample_audio
from streaming.core.session import MediaEgressAdapter
from streaming.transport.livekit.audio_publisher import AudioPublisher
from streaming.transport.livekit.video_publisher import VideoPublisher

from ._net_probe import (
    _attach_room_diag,
    _derive_net_probe_host,
    _derive_net_probe_signaling_host,
    _env_bool,
    _env_float,
    _is_ice_timeout_error,
    _ICE_TRANSPORT_RELAY,
    _probe_targets,
    _tcp_egress_connectable,
    record_egress_hosts,
)

logger = logging.getLogger("LiveKitEgressAdapter")


class LiveKitEgressAdapter(MediaEgressAdapter):
    """Publish avatar media to a LiveKit room.

    A pre-connected room may be supplied for callers that own the connection. When
    only ``room_url`` and ``room_token`` are supplied, the adapter owns the room
    lifecycle and connects it during :meth:`connect`.
    """

    _TEARDOWN_WAIT_SECONDS = 10.0
    _CONNECT_RETRY_BASE_SECONDS = 1.0

    def __init__(
        self,
        *,
        room=None,
        room_url: str | None = None,
        room_token: str | None = None,
        fps: int = 25,
        sample_rate: int = 16_000,
        channels: int = 1,
        session_id: str = "",
    ):
        self.room = room
        self.room_url = room_url
        self.room_token = room_token
        self.fps = int(fps)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.session_id = session_id
        self.video_publisher: VideoPublisher | None = None
        self.audio_publisher: AudioPublisher | None = None
        self.video_task: asyncio.Task | None = None
        self._connected = False
        self._owns_room = False
        self._lifecycle_lock = asyncio.Lock()
        self._teardown_task: asyncio.Task | None = None
        self._probe_task: asyncio.Task | None = None
        self._probe_room_url: str | None = None
        self._probe_result: tuple[bool, str | None] | None = None

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        if self.room is not None and self.room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
            return False
        return True

    @property
    def first_frame_published(self):
        if self.video_publisher is None:
            return None
        return self.video_publisher.first_frame_published

    async def _disconnect_room(self, room) -> None:
        disconnect = getattr(room, "disconnect", None)
        if disconnect is None:
            return
        try:
            await disconnect()
        except Exception as exc:
            logger.debug("Room disconnect failed: %s", exc)

    async def _cleanup_candidate_room(self, room, *, raise_on_timeout: bool, teardown_timeout: float | None = None) -> bool:
        """Retain failed candidate cleanup as the next lifecycle barrier."""
        task = asyncio.create_task(
            self._run_teardown(None, None, None, room, True),
            name=f"livekit-candidate-teardown-{self.session_id}",
        )
        self._teardown_task = task
        return await self._await_teardown(task, raise_on_timeout=raise_on_timeout, timeout=teardown_timeout)

    async def _run_teardown(
        self,
        video_task,
        video_publisher,
        audio_publisher,
        room,
        owns_room,
    ) -> None:
        try:
            if video_task is not None:
                video_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await video_task

            if video_publisher is not None:
                try:
                    await video_publisher.aclose()
                except Exception as exc:
                    logger.debug("VideoPublisher teardown failed: %s", exc)

            if audio_publisher is not None:
                try:
                    await audio_publisher.aclose()
                except Exception as exc:
                    logger.debug("AudioPublisher teardown failed: %s", exc)

            if owns_room and room is not None:
                await self._disconnect_room(room)
        finally:
            # State was detached before this task started. Clearing this reference
            # only after all native teardown steps prevents a new connect from
            # overlapping a previous room's shutdown.
            if self._teardown_task is asyncio.current_task():
                self._teardown_task = None

    async def _await_teardown(self, task: asyncio.Task, *, raise_on_timeout: bool, timeout: float | None = None) -> bool:
        wait_seconds = self._TEARDOWN_WAIT_SECONDS if timeout is None else timeout
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=wait_seconds)
            return True
        except asyncio.TimeoutError:
            logger.warning("LiveKit teardown still pending after %.1fs", wait_seconds)
            if raise_on_timeout:
                raise RuntimeError("Previous LiveKit teardown is still pending")
            return False
        except asyncio.CancelledError:
            # Preserve caller cancellation, but bound how long it can be delayed by
            # native teardown. The retained task remains the reconnect barrier.
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=wait_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "LiveKit teardown still pending after cancelled caller waited %.1fs",
                    wait_seconds,
                )
            except asyncio.CancelledError:
                pass
            raise

    async def drain_teardown(self) -> bool:
        """Await retained native cleanup without allowing concurrent reconnects."""
        task = self._teardown_task
        if task is None:
            return True
        return await self._await_teardown(task, raise_on_timeout=False)

    async def _wait_for_udp_egress_ready(
        self, room_url: str
    ) -> tuple[bool, str | None]:
        """Probe UDP responses and TCP 443, returning the relay-first decision."""
        # Deferred import: tests patch _warm_egress_targets on the package
        # namespace via monkeypatch.setattr(livekit_egress, ...).
        from streaming.transport.adapters.livekit_egress import _warm_egress_targets

        if os.environ.get("LIVEKIT_NET_PROBE_ENABLED", "true").strip().lower() in {
            "0",
            "false",
            "no",
        }:
            return False, None
        turn_host = _derive_net_probe_host(room_url)
        signaling_host = _derive_net_probe_signaling_host(room_url)
        try:
            port = int(os.environ.get("LIVEKIT_NET_PROBE_PORT", "3478"))
        except ValueError:
            port = 3478
        try:
            tcp_timeout = float(
                os.environ.get("LIVEKIT_NET_PROBE_TCP_TIMEOUT_SECONDS", "1.0")
            )
        except ValueError:
            tcp_timeout = 1.0
        tcp_timeout = max(tcp_timeout, 0.05)

        targets: list[tuple[str, int]] = [(turn_host, port)]
        if signaling_host and signaling_host != turn_host:
            targets.append((signaling_host, port))
        try:
            record_egress_hosts(targets)
        except Exception:
            pass

        started = time.monotonic()
        warm_enabled = _env_bool("LIVEKIT_EGRESS_WARM_ENABLED", True)
        warm_task = None
        if warm_enabled:
            warm_hosts = [host for host, _port in targets]
            try:
                room_host = urlsplit(room_url).hostname
            except ValueError:
                room_host = None
            if room_host and room_host not in warm_hosts:
                warm_hosts.append(room_host)
            warm_task = asyncio.create_task(
                asyncio.to_thread(_warm_egress_targets, warm_hosts, port)
            )
        try:
            sendable, error, can_probe, target = await asyncio.to_thread(
                _probe_targets, targets
            )
        finally:
            if warm_task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(asyncio.shield(warm_task), timeout=2.0)
        if sendable:
            logger.debug(
                "LIVEKIT_NET_PROBE_OK session=%s target=%s:%d",
                self.session_id,
                target[0],
                target[1],
            )
            return False, None
        if not can_probe:
            logger.warning(
                "LIVEKIT_NET_PROBE_UNPROBEABLE session=%s targets=%s error=%s proceeding",
                self.session_id,
                ",".join(f"{h}:{p}" for h, p in targets),
                error,
            )
            return False, None

        logger.warning(
            "LIVEKIT_NET_PROBE_BLOCKED session=%s targets=%s error=%s",
            self.session_id,
            ",".join(f"{h}:{p}" for h, p in targets),
            error,
        )
        tcp_ready, tcp_error, family, tcp_host = await asyncio.to_thread(
            _tcp_egress_connectable,
            [host for host, _port in targets],
            tcp_timeout,
        )
        if tcp_ready:
            family_name = {
                socket.AF_INET: "IPv4",
                socket.AF_INET6: "IPv6",
            }.get(family, str(family))
            logger.warning(
                "LIVEKIT_NET_PROBE_READY session=%s target=%s:443 waitedMs=%.0f transport=tcp",
                self.session_id,
                tcp_host,
                (time.monotonic() - started) * 1000.0,
            )
            return True, f"udp_silent_tcp_443_ready family={family_name} host={tcp_host}"

        logger.warning(
            "LIVEKIT_NET_PROBE_TIMEOUT session=%s targets=%s waitedMs=%.0f error=%s tcpError=%s proceeding",
            self.session_id,
            ",".join(f"{h}:{p}" for h, p in targets),
            (time.monotonic() - started) * 1000.0,
            error,
            tcp_error,
        )
        # Deep-blanket correction: UDP silent plus TCP probe ambiguous still
        # goes RELAY. Fleet evidence says signaling TCP works throughout, so a
        # failed 1s raw-TCP probe is more likely probe flake than proof that
        # TURN/TLS on 443 is dead. ALL in a UDP-dead netstack is a guaranteed
        # 15s ICE burn while RELAY at least offers TURN/TLS plus ICE/TCP.
        # Server ICE is preserved (no custom ice_servers). Kill switch:
        # LIVEKIT_DEEP_BLANKET_RELAY=0 restores the old direct best-effort.
        if _env_bool("LIVEKIT_DEEP_BLANKET_RELAY", True):
            return True, "deep_blanket_udp_silent_tcp_uncertain"
        return False, None

    def start_probe_early(self, *, session_id: str | None = None) -> asyncio.Task | None:
        """Start _wait_for_udp_egress_ready in background. Idempotent, never raises."""
        try:
            if session_id:
                self.session_id = str(session_id)
            if self.room is not None:
                return None
            room_url = self.room_url
            if not room_url:
                return None
            existing = self._probe_task
            if existing is not None and not existing.done():
                return existing
            if self._probe_result is not None and self._probe_room_url == room_url:
                return None
            loop = asyncio.get_running_loop()
            self._probe_room_url = room_url
            task = loop.create_task(
                self._wait_for_udp_egress_ready(room_url),
                name=f"livekit-probe-{self.session_id}",
            )

            def _cache(t: asyncio.Task) -> None:
                try:
                    if not t.cancelled():
                        self._probe_result = t.result()
                except Exception:
                    self._probe_result = None

            task.add_done_callback(_cache)
            self._probe_task = task
            return task
        except RuntimeError:
            return None

    def cancel_probe_early(self) -> None:
        """Cancel in-flight early probe. No-op when done or absent. Never raises."""
        try:
            task = self._probe_task
            if task is not None and not task.done():
                task.cancel()
        except Exception:
            pass

    async def _get_probe_decision(self, room_url: str) -> tuple[bool, str | None]:
        """Join early probe when usable, else run fresh. Preserves all markers."""
        task = self._probe_task
        if task is not None and self._probe_room_url == room_url:
            if self._probe_result is not None and task.done() and not task.cancelled():
                return self._probe_result
            if not task.done():
                try:
                    return await task
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
        return await self._wait_for_udp_egress_ready(room_url)

    async def connect(self, room_config: dict[str, Any] | Mapping[str, Any]) -> None:
        # Deferred import: tests patch VideoPublisher and AudioPublisher on
        # the package namespace via monkeypatch.setattr(livekit_egress, ...).
        from streaming.transport.adapters.livekit_egress import VideoPublisher, AudioPublisher

        config = dict(room_config)
        async with self._lifecycle_lock:
            if self._teardown_task is not None:
                await self._await_teardown(self._teardown_task, raise_on_timeout=True)

            self.room = config.get("room", self.room)
            self.room_url = config.get("room_url", config.get("url", self.room_url))
            self.room_token = config.get("room_token", config.get("token", self.room_token))
            self.fps = int(config.get("fps", self.fps))
            self.sample_rate = int(config.get("sample_rate", self.sample_rate))
            self.channels = int(config.get("channels", self.channels))
            self.session_id = str(config.get("session_id", self.session_id))

            if self.room is None:
                if not self.room_url or not self.room_token:
                    raise ValueError("LiveKit egress requires room or room_url and room_token")

                max_attempts = 3
                base_delay = self._CONNECT_RETRY_BASE_SECONDS
                # Retry-budget surgery: ICE-timeout burns get fast teardown
                # and backoff so two RELAY attempts fit inside a useful media
                # budget. Other errors keep the slow barrier. All env-gated.
                icefail_teardown = _env_float("LIVEKIT_ICEFAIL_TEARDOWN_S", 1.5, 0.2)
                icefail_backoff = _env_float("LIVEKIT_ICEFAIL_BACKOFF_S", 0.5, 0.0)
                connect_timeout = _env_float("LIVEKIT_CONNECT_TIMEOUT_SECONDS", 7.0, 1.0)
                last_error: Exception | None = None
                relay_first, relay_first_reason = await self._get_probe_decision(
                    self.room_url
                )
                if relay_first:
                    logger.info(
                        "LIVEKIT_RELAY_FIRST session=%s reason=%s",
                        self.session_id,
                        relay_first_reason,
                    )

                for attempt in range(1, max_attempts + 1):
                    candidate_room = rtc.Room()
                    _attach_room_diag(candidate_room, self.session_id)
                    attempt_started = time.monotonic()

                    use_relay = relay_first or attempt > 1
                    if attempt > 1:
                        logger.info(
                            "LIVEKIT_CONNECT_RELAY_FALLBACK session=%s attempt=%d/%d",
                            self.session_id,
                            attempt,
                            max_attempts,
                        )

                    try:
                        options_type = getattr(rtc, "RoomOptions", None)
                        if options_type is None:
                            await candidate_room.connect(self.room_url, self.room_token)
                        else:
                            rtc_config = None
                            if use_relay:
                                rtc_config_type = getattr(rtc, "RtcConfiguration", None)
                                if rtc_config_type is not None:
                                    rtc_config = rtc_config_type(
                                        ice_transport_type=_ICE_TRANSPORT_RELAY,
                                    )
                            options_kwargs: dict[str, Any] = {
                                "auto_subscribe": False,
                                "single_peer_connection": True,
                                "connect_timeout": connect_timeout,
                            }
                            if rtc_config is not None:
                                options_kwargs["rtc_config"] = rtc_config
                            await candidate_room.connect(
                                self.room_url,
                                self.room_token,
                                options=options_type(**options_kwargs),
                            )
                    except asyncio.CancelledError:
                        await self._cleanup_candidate_room(
                            candidate_room,
                            raise_on_timeout=False,
                        )
                        raise
                    except Exception as exc:
                        ice_timeout = _is_ice_timeout_error(exc)
                        await self._cleanup_candidate_room(
                            candidate_room,
                            raise_on_timeout=True,
                            teardown_timeout=icefail_teardown if ice_timeout else None,
                        )
                        last_error = exc
                        logger.warning(
                            "LIVEKIT_CONNECT_ATTEMPT_FAILED session=%s attempt=%d/%d elapsedMs=%.1f error=%s iceTimeout=%s",
                            self.session_id,
                            attempt,
                            max_attempts,
                            (time.monotonic() - attempt_started) * 1000.0,
                            exc,
                            ice_timeout,
                        )
                        if attempt < max_attempts:
                            if ice_timeout:
                                await asyncio.sleep(icefail_backoff)
                            else:
                                await asyncio.sleep(base_delay * attempt)
                        continue

                    self.room = candidate_room
                    self._owns_room = True
                    last_error = None
                    logger.info(
                        "LIVEKIT_CONNECT_SUCCEEDED session=%s attempt=%d/%d elapsedMs=%.1f",
                        self.session_id,
                        attempt,
                        max_attempts,
                        (time.monotonic() - attempt_started) * 1000.0,
                    )
                    break

                if last_error is not None:
                    raise last_error

            self.video_publisher = VideoPublisher(self.room, fps=self.fps, session_id=self.session_id)
            self.audio_publisher = AudioPublisher(
                room=self.room,
                sample_rate=self.sample_rate,
                num_channels=self.channels,
            )
            self._connected = True

    async def disconnect(self) -> None:
        async with self._lifecycle_lock:
            self.cancel_probe_early()
            self._probe_task = None
            self._probe_room_url = None
            self._probe_result = None
            task = self._teardown_task
            if task is None:
                room = self.room
                owns_room = self._owns_room
                video_publisher = self.video_publisher
                audio_publisher = self.audio_publisher
                video_task = self.video_task

                # Detach state before teardown, but retain a shared task as the
                # lifecycle barrier. A future connect must join this task first.
                self.video_task = None
                self.audio_publisher = None
                self.video_publisher = None
                self.room = None if owns_room else room
                self._owns_room = False
                self._connected = False

                task = asyncio.create_task(
                    self._run_teardown(
                        video_task,
                        video_publisher,
                        audio_publisher,
                        room,
                        owns_room,
                    ),
                    name=f"livekit-egress-teardown-{self.session_id}",
                )
                self._teardown_task = task

        await self._await_teardown(task, raise_on_timeout=False)

    async def publish_video_frame(self, frame: np.ndarray, pts: float) -> None:
        self._require_connected()
        if self.video_publisher is None:
            raise RuntimeError("LiveKit video publisher is not initialized")
        await self.video_publisher._ensure_track(frame)
        await self.video_publisher._send_frame(frame)

    async def publish_audio_chunk(self, pcm: np.ndarray, sample_rate: int) -> None:
        self._require_connected()
        if self.audio_publisher is None:
            raise RuntimeError("LiveKit audio publisher is not initialized")
        audio = np.asarray(pcm, dtype=np.float32)
        audio = resample_audio(audio, int(sample_rate), self.sample_rate)
        await self.audio_publisher.push_audio(audio)

    async def wait_for_ready(self) -> None:
        self._require_connected()
        if self.video_publisher is None:
            raise RuntimeError("LiveKit video publisher is not initialized")
        await self.video_publisher.first_frame_published.wait()

    def start_video(self, state_manager) -> asyncio.Task:
        self._require_connected()
        if self.video_publisher is None:
            raise RuntimeError("LiveKit video publisher is not initialized")
        if self.video_task is not None:
            return self.video_task
        self.video_task = asyncio.create_task(
            self.video_publisher.publish_from_state_manager(state_manager),
            name=f"video-egress-{self.session_id}",
        )
        return self.video_task

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("LiveKit egress is not connected")
