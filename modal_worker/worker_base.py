"""Shared Modal worker lifecycle for modal_app.py (production) and
modal_app_stress.py (stress testing).

WorkerBase carries the CPU-snapshot lifecycle (load/restore), the shared
serve() scaffolding (env, logging, handler import, batched engine init), and
the production shutdown flow. Subclasses are decorated with @app.cls in the
root entry files and provide tier-specific GPU/concurrency via class
attributes.

Diagnostics hooks are class attributes so the stress app can inject its
enhanced local versions without duplicating the lifecycle bodies.
"""

import asyncio
import contextlib
import os
import socket
import sys
import time
from pathlib import Path

import modal

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.modal_diagnostics import (  # noqa: E402
    ShortenLiveKitWebSocketUrlFilter,
    log_attention_backend,
    log_cuda_state,
    log_signal_handlers,
    reset_ucx_signal_handlers,
)

from modal_worker.gpu_config import GPU_CONFIG  # noqa: E402
from modal_worker.warmup import warm_pipeline  # noqa: E402


class WorkerBase:
    """Shared worker logic for all GPU tiers. Not registered with Modal directly.
    Subclasses (WorkerLow, WorkerHigh, stress Worker) are decorated with @app.cls
    and provide tier-specific GPU and concurrency settings via class attributes.
    """

    TARGET_GPU = "L4"
    WORKER_CONCURRENCY = 1
    LOG_PREFIX = "[modal]"

    # Diagnostics hooks - the stress app overrides these with its enhanced
    # local versions (crash handler with thread dumps, /proc/self/maps, ...).
    _diag_log_attention_backend = staticmethod(log_attention_backend)
    _diag_log_cuda_state = staticmethod(log_cuda_state)
    _diag_log_signal_handlers = staticmethod(log_signal_handlers)
    _diag_reset_ucx_signal_handlers = staticmethod(reset_ucx_signal_handlers)

    @modal.enter(snap=True)
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/SoulX-FlashHead")

        # Set env vars BEFORE any import that reads them.
        # FLASHHEAD_LOAD_DEVICE=cpu makes get_device() return "cpu" so
        # the pipeline loads to CPU with no CUDA calls during snapshot.
        os.environ["AIVATAR_WORKER_CONCURRENCY"] = str(self.WORKER_CONCURRENCY)
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["FLASHHEAD_LOAD_DEVICE"] = "cpu"
        os.environ.setdefault("LIVEKIT_RTC_DEBUG", "false")
        os.environ["ENGINE_PROFILE"] = "1"
        os.environ["AIVATAR_BATCHED_INFERENCE"] = "1"

        print(
            f"{self.LOG_PREFIX} CPU snapshot load with "
            f"WORKER_CONCURRENCY={self.WORKER_CONCURRENCY} TARGET_GPU={self.TARGET_GPU}",
            flush=True,
        )

        # Import ONLY flash_head (torch + model) - NOT handler/livekit.
        # handler.py imports streaming.orchestration.stream_processor which imports
        # livekit (Rust FFI). LiveKit spawns background threads that get captured
        # in the CRIU snapshot and corrupt CUDA state after restore.
        # handler.py is deferred to serve() (post-restore).
        from flash_head.inference import get_pipeline

        # Log which attention backend is available (CPU mode - no GPU calls)
        self._diag_log_attention_backend("SNAP_CREATE")

        ckpt_dir = os.getenv("FLASHHEAD_CKPT_DIR", "/app/models/SoulX-FlashHead-1_3B")
        wav2vec_dir = os.getenv("WAV2VEC_DIR", "/app/models/wav2vec2-base-960h")
        model_type = os.getenv("FLASHHEAD_MODEL_TYPE", "lite")

        print("[SNAP_CREATE] Loading FlashHead pipeline on CPU...", flush=True)
        load_t0 = time.monotonic()
        self._snap_pipeline = get_pipeline(1, ckpt_dir, model_type, wav2vec_dir)
        load_ms = (time.monotonic() - load_t0) * 1000
        print(f"[SNAP_CREATE] Pipeline loaded to CPU in {load_ms:.1f}ms", flush=True)

        # Preload avatar and idle video caches (CPU-only, no livekit dependency)
        from utils.default_avatar_cache import default_avatar_cache
        from utils.default_idle_video_cache import default_idle_video_cache
        cache_status = default_avatar_cache.preload()
        idle_cache_status = default_idle_video_cache.preload()
        if cache_status["manifestFound"]:
            print(
                f"[modal] Default avatar manifest loaded: cachedAvatarCount={cache_status['cachedAvatarCount']} "
                f"failed={len(cache_status['failedAvatarIds'])}",
                flush=True,
            )
        else:
            print(
                f"[modal] Default avatar manifest not found at {cache_status['manifestPath']} - "
                "falling back to per-session URL fetch",
                flush=True,
            )
        print(
            f"[modal] Default idle cache: cachedIdleVideoCount={idle_cache_status['cachedIdleVideoCount']} "
            f"failed={len(idle_cache_status['failedIdleVideoKeys'])}",
            flush=True,
        )

        # Reset UCX signal handlers to SIG_DFL before snapshot.
        # import torch loads UCX/NCCL libs which install signal handlers at
        # library load time. These can cause race conditions during CRIU restore.
        self._diag_log_signal_handlers("SNAP_PRE_RESET")
        self._diag_reset_ucx_signal_handlers()
        self._diag_log_signal_handlers("SNAP_POST_RESET")

        print(f"{self.LOG_PREFIX} Pipeline loaded to CPU for snapshot", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        import sys
        sys.path.insert(0, "/app")

        # Pop FLASHHEAD_LOAD_DEVICE so future get_pipeline() calls use GPU
        os.environ.pop("FLASHHEAD_LOAD_DEVICE", None)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[RESTORE] CPU snapshot restore - moving pipeline to {device}", flush=True)
        self._diag_log_cuda_state("RESTORE_START")

        # Move the snapshot pipeline from CPU to GPU
        assert self._snap_pipeline is not None, "Pipeline lost in snapshot!"
        move_t0 = time.monotonic()
        self._snap_pipeline.move_to_device(device)
        if device == "cuda":
            torch.cuda.synchronize()
        move_ms = (time.monotonic() - move_t0) * 1000
        print(f"[RESTORE] Pipeline moved to {device} in {move_ms:.1f}ms", flush=True)

        self._diag_log_cuda_state("RESTORE_POST_MOVE")

        # Warmup on GPU
        print("[RESTORE] Running warmup inference on GPU...", flush=True)
        warm_t0 = time.monotonic()
        warm_pipeline(self._snap_pipeline)
        warm_ms = (time.monotonic() - warm_t0) * 1000
        print(f"[RESTORE] Warmup completed in {warm_ms:.1f}ms", flush=True)

        self._diag_log_cuda_state("RESTORE_POST_WARMUP")
        self._diag_log_attention_backend("RESTORE_POST")

        # General netstack priming: send a UDP packet to a public DNS resolver
        # to warm the gVisor netstack's link state, route cache, and DNS
        # resolver. On freshly restored containers the gVisor netstack can
        # reject sendto() locally (ENETUNREACH) for tens of seconds while TCP
        # keeps working. This priming is best-effort and never blocks startup.
        self._prime_netstack()

        print("[RESTORE] CPU snapshot restore complete", flush=True)

    def _prime_netstack(self) -> None:
        """Best-effort general netstack priming after container restore.

        Sends a single UDP packet to a public DNS resolver (1.1.1.1:53) to
        warm the gVisor netstack's link state and route cache. This helps
        with the blanket UDP egress blackhole observed on freshly restored
        Modal containers (log2.log, Sep 3 2026) where ALL newly bound UDP
        sockets fail sendto() locally with ENETUNREACH.

        This does NOT help with per-destination blackholes (log3.log) where
        one media IP stays unreachable while others clear. The
        per-destination case is handled by egress warming plus TURN relay
        in the LiveKit egress adapter.

        All failures are logged at WARNING and never block startup.
        """
        import struct

        prime_host = os.environ.get("LIVEKIT_NET_PRIME_HOST", "1.1.1.1")
        try:
            prime_port = int(os.environ.get("LIVEKIT_NET_PRIME_PORT", "53"))
        except ValueError:
            prime_port = 53

        # Minimal DNS query packet (root NS, type A) for the priming sendto.
        dns_query = struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0) + b"\x00"

        prime_t0 = time.monotonic()
        try:
            infos = socket.getaddrinfo(prime_host, prime_port, socket.AF_UNSPEC, socket.SOCK_DGRAM)
            if not infos:
                print(f"[RESTORE] Netstack prime: no addresses for {prime_host}", flush=True)
                return
            # Prefer IPv4 for the priming probe.
            infos.sort(key=lambda info: 0 if info[0] == socket.AF_INET else 1)
            family, socktype, proto, _, address = infos[0]
            with socket.socket(family, socktype, proto) as sock:
                sock.sendto(dns_query, address)
            prime_ms = (time.monotonic() - prime_t0) * 1000
            print(
                f"[RESTORE] Netstack prime OK target={prime_host}:{prime_port} "
                f"elapsedMs={prime_ms:.1f}",
                flush=True,
            )
        except OSError as exc:
            prime_ms = (time.monotonic() - prime_t0) * 1000
            print(
                f"[RESTORE] Netstack prime FAILED target={prime_host}:{prime_port} "
                f"elapsedMs={prime_ms:.1f} error={exc.__class__.__name__}: {exc}",
                flush=True,
            )

    def get_nethealth(self) -> dict:
        """Return background UDP heartbeat health for routing/observability."""
        return {
            "lastUdpOkEpoch": getattr(self, "_nethealth_last_udp_ok", None),
            "consecutiveFailures": getattr(self, "_nethealth_failures", 0),
            "lastCheckEpoch": getattr(self, "_nethealth_last_check", None),
            "recentHosts": list(getattr(self, "_nethealth_recent_hosts", [])),
            "lastResponseGatedOkEpoch": getattr(self, "_nethealth_last_response_ok", None),
            "responseGatedFailures": getattr(self, "_nethealth_response_failures", 0),
            "lastResponseCheckEpoch": getattr(self, "_nethealth_last_response_check", None),
        }

    def start_udp_heartbeat(self) -> None:
        """Start the background UDP heartbeat thread (idempotent).

        Sends small UDP datagrams every few seconds from serve() post-restore
        through session life to keep the gVisor netstack warm. Pure stdlib
        socket, no rtc import, no FFI. Stops on stop_udp_heartbeat().
        """
        import threading

        if getattr(self, "_udp_heartbeat_thread", None) is not None:
            thread = self._udp_heartbeat_thread
            if thread.is_alive():
                return
        try:
            enabled = os.environ.get("LIVEKIT_HEARTBEAT_ENABLED", "true").strip().lower() not in {
                "0", "false", "no", "off",
            }
        except Exception:
            enabled = True
        if not enabled:
            return
        try:
            interval = float(os.environ.get("LIVEKIT_HEARTBEAT_INTERVAL_S", "3.0"))
        except ValueError:
            interval = 3.0
        interval = max(interval, 0.5)
        self._nethealth_last_udp_ok = None
        self._nethealth_failures = 0
        self._nethealth_last_check = None
        self._nethealth_was_ok = None
        self._nethealth_last_response_ok = None
        self._nethealth_response_failures = 0
        self._nethealth_last_response_check = None
        self._nethealth_gated_was_ok = None
        self._nethealth_recent_hosts = []
        self._nethealth_rr_index = 0
        try:
            response_check_s = float(os.environ.get("LIVEKIT_HEARTBEAT_RESPONSE_CHECK_S", "10.0"))
        except ValueError:
            response_check_s = 10.0
        self._heartbeat_response_check_s = response_check_s
        stop = threading.Event()
        self._udp_heartbeat_stop = stop
        thread = threading.Thread(
            target=self._udp_heartbeat_loop,
            args=(stop, interval),
            daemon=True,
            name="udp-heartbeat",
        )
        self._udp_heartbeat_thread = thread
        thread.start()
        print(f"[SERVE] UDP heartbeat started interval={interval:.1f}s", flush=True)

    def stop_udp_heartbeat(self) -> None:
        stop = getattr(self, "_udp_heartbeat_stop", None)
        thread = getattr(self, "_udp_heartbeat_thread", None)
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)

    def _udp_stun_response_ok(self, host: str, port: int, timeout: float) -> bool:
        """STUN binding request plus matching response check. Stdlib only."""
        import struct

        req = struct.pack("!HHI", 0x0001, 0, 0x2112A442) + b"\x00" * 12
        try:
            infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_DGRAM)
        except OSError:
            return False
        if not infos:
            return False
        infos.sort(key=lambda info: 0 if info[0] == socket.AF_INET else 1)
        for family, socktype, proto, _, address in infos[:2]:
            try:
                with socket.socket(family, socktype, proto) as sock:
                    sock.settimeout(timeout)
                    sock.sendto(req, address)
                    response, _source = sock.recvfrom(2048)
                    if (
                        len(response) >= 20
                        and response[:2] in {b"\x01\x01", b"\x01\x11"}
                        and response[4:8] == req[4:8]
                        and response[8:20] == req[8:20]
                    ):
                        return True
            except OSError:
                continue
        return False

    def _udp_heartbeat_loop(self, stop_event, interval: float) -> None:
        import struct

        stun = struct.pack("!HHI", 0x0001, 0, 0x2112A442) + b"\x00" * 12
        dns_query = struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0) + b"\x00"
        seeds: list[tuple[str, int, bytes]] = [
            ("1.1.1.1", 3478, stun),
            ("8.8.8.8", 53, dns_query),
            ("1.1.1.1", 53, dns_query),
        ]
        try:
            extra_host = os.environ.get("LIVEKIT_HEARTBEAT_EXTRA_HOST")
            if extra_host:
                try:
                    extra_port = int(os.environ.get("LIVEKIT_HEARTBEAT_EXTRA_PORT", "3478"))
                except ValueError:
                    extra_port = 3478
                seeds.append((extra_host, extra_port, stun))
        except Exception:
            pass
        try:
            resp_timeout = float(os.environ.get("LIVEKIT_NET_PROBE_RESPONSE_TIMEOUT_SECONDS", "0.8"))
        except ValueError:
            resp_timeout = 0.8
        resp_timeout = max(resp_timeout, 0.05)
        while not stop_event.is_set():
            recent: list[tuple[str, int]] = []
            try:
                from streaming.transport.adapters.livekit_egress import get_recent_egress_hosts

                recent = get_recent_egress_hosts() or []
            except Exception:
                recent = []
            self._nethealth_recent_hosts = list(recent)
            fast_ok = False
            send_targets = [(host, port, stun) for host, port in recent] + seeds
            for host, port, payload in send_targets:
                try:
                    infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_DGRAM)
                except OSError:
                    continue
                if not infos:
                    continue
                infos.sort(key=lambda info: 0 if info[0] == socket.AF_INET else 1)
                try:
                    family, socktype, proto, _, address = infos[0]
                    with socket.socket(family, socktype, proto) as sock:
                        sock.settimeout(0.5)
                        sock.sendto(payload, address)
                    fast_ok = True
                except OSError:
                    continue
            now = time.time()
            self._nethealth_last_check = now
            if fast_ok:
                self._nethealth_last_udp_ok = now
                self._nethealth_failures = 0
                if self._nethealth_was_ok is False:
                    print("[HEARTBEAT] UDP egress recovered", flush=True)
                self._nethealth_was_ok = True
            else:
                self._nethealth_failures = getattr(self, "_nethealth_failures", 0) + 1
                if self._nethealth_was_ok is not False:
                    print(
                        f"[HEARTBEAT] UDP egress blocked failures={self._nethealth_failures}",
                        flush=True,
                    )
                elif self._nethealth_failures % 10 == 0:
                    print(
                        f"[HEARTBEAT] UDP egress still blocked failures={self._nethealth_failures}",
                        flush=True,
                    )
                self._nethealth_was_ok = False
            check_s = getattr(self, "_heartbeat_response_check_s", 10.0)
            last_gated = self._nethealth_last_response_check
            if check_s > 0 and recent and (last_gated is None or (now - last_gated) >= check_s):
                target = recent[getattr(self, "_nethealth_rr_index", 0) % len(recent)]
                self._nethealth_rr_index = getattr(self, "_nethealth_rr_index", 0) + 1
                gated_ok = self._udp_stun_response_ok(target[0], target[1], resp_timeout)
                self._nethealth_last_response_check = now
                if gated_ok:
                    self._nethealth_last_response_ok = now
                    self._nethealth_response_failures = 0
                    if self._nethealth_gated_was_ok is False:
                        print("[HEARTBEAT-GATED] UDP response path recovered", flush=True)
                    self._nethealth_gated_was_ok = True
                else:
                    self._nethealth_response_failures = getattr(self, "_nethealth_response_failures", 0) + 1
                    if self._nethealth_gated_was_ok is not False:
                        print(
                            f"[HEARTBEAT-GATED] UDP response blocked failures={self._nethealth_response_failures}",
                            flush=True,
                        )
                    elif self._nethealth_response_failures % 10 == 0:
                        print(
                            f"[HEARTBEAT-GATED] UDP response still blocked failures={self._nethealth_response_failures}",
                            flush=True,
                        )
                    self._nethealth_gated_was_ok = False
            stop_event.wait(interval)

    async def _load_remaining_pipelines(self):
        """Fill configured capacity with the same shared pipeline object.

        handler.model_pool is created in serve() with INITIAL_PIPELINE_COUNT=WORKER_POOL_SIZE
        (FLASHHEAD_LOAD_DEVICE is popped in restore). The monkey-patched get_pipeline()
        returns self._snap_pipeline for every call, so the pool already has WORKER_POOL_SIZE
        references to the same pipeline. This method is a no-op but kept for compatibility.
        """
        pool = self._handler.model_pool
        print(
            f"{self.LOG_PREFIX} Pool status: current_size={pool.current_size} "
            f"available={pool.get_available_count()}/{pool.max_size}",
            flush=True,
        )

    def _serve_common(self, logger_name):
        """Shared serve() scaffolding: env vars, logging config. Returns logger."""
        import logging
        import sys

        sys.path.insert(0, "/app")

        # Re-assert env vars in case restore() cleared anything.
        os.environ["AIVATAR_BATCHED_INFERENCE"] = "1"

        # Video codec: driven by GPU_CONFIG. GPUs with incompatible NVENC (e.g.
        # RTX PRO 6000 Blackwell 9th-gen) use VP8 software encoding. Others use
        # H264 hardware encoding.
        os.environ.setdefault("AIVATAR_VIDEO_CODEC", GPU_CONFIG[self.TARGET_GPU]["video_codec"])

        # Relax WebSocket rate limits for multi-session serving.
        # Previous defaults (60 msg/s, 200 burst, 192KB/s, 10 conn/IP) were too
        # restrictive for multiple concurrent sessions on a single container.
        os.environ.setdefault("WS_AUDIO_RATE_PER_SEC", "600")
        os.environ.setdefault("WS_AUDIO_BURST", "2000")
        os.environ.setdefault("WS_MAX_BYTES_PER_SEC", "1920000")
        os.environ.setdefault("WS_MAX_CONNECTIONS_PER_IP", "100")

        # Configure root logger so all Python loggers (stream_processor,
        # VideoPublisher, AudioPublisher, etc.) output to stdout.
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )
        for handler in logging.getLogger().handlers:
            handler.addFilter(ShortenLiveKitWebSocketUrlFilter())
        return logging.getLogger(logger_name)

    def _init_handler_with_pipeline(self):
        """Monkey-patch get_pipeline, import handler, start BatchedStreamingEngine.

        handler.py creates model_pool at import time which calls get_pipeline()
        INITIAL_PIPELINE_COUNT times. Since FLASHHEAD_LOAD_DEVICE is popped in
        restore(), INITIAL_PIPELINE_COUNT=WORKER_POOL_SIZE, so the pool fills
        with WORKER_POOL_SIZE references to self._snap_pipeline.

        handler imports livekit (Rust FFI) which is unsafe during CRIU snapshot
        but safe after restore - only call this from serve() (post-restore).
        Returns the handler module.
        """
        import flash_head.inference as fhi
        _orig_get_pipeline = fhi.get_pipeline
        fhi.get_pipeline = lambda *a, **kw: self._snap_pipeline

        import handler
        self._handler = handler

        # Restore get_pipeline for any future calls
        fhi.get_pipeline = _orig_get_pipeline

        # Initialize BatchedStreamingEngine with the snapshot pipeline.
        # All concurrent sessions share this single engine for batched inference.
        from streaming.inference.batched import BatchedStreamingEngine
        handler.batched_engine = BatchedStreamingEngine(self._snap_pipeline)
        handler.batched_engine.start()
        print(
            f"BatchedStreamingEngine started (wait_window={handler.batched_engine.wait_window_ms}ms)",
            flush=True,
        )
        print(
            f"Pipeline in model_pool: "
            f"current_size={handler.model_pool.current_size} "
            f"available={handler.model_pool.get_available_count()}",
            flush=True,
        )
        return handler

    def _publish_serve_controls(self, loop):
        with self._serve_controls_lock:
            self._serve_stop_event = asyncio.Event()
            self._serve_loop = loop
            if self._serve_shutdown_requested:
                self._serve_stop_event.set()

    async def _shutdown_runtime(self):
        """Drain Modal ingress, processors, lifecycle callbacks, and warmup."""
        print("[EXIT] Shutting down Modal serve runtime", flush=True)
        handler = getattr(self, "_handler", None)
        try:
            # aiohttp owns the public /ws connections. Signal every session first
            # so each upgraded socket closes and its processor receives EndFrame.
            if handler is not None:
                signalled = await handler.request_active_session_shutdown(
                    reason="worker_shutdown"
                )
                if signalled:
                    print(
                        f"[EXIT] Signalled {signalled} active session(s)",
                        flush=True,
                    )

                # This listener is separate from aiohttp but may still be active in
                # compatibility deployments, so stop it after session signalling.
                if handler.ws_server.is_running:
                    try:
                        await asyncio.wait_for(handler.ws_server.stop(), timeout=8)
                    except asyncio.TimeoutError:
                        print(
                            "[EXIT] Timed out stopping WebSocket ingestion server",
                            flush=True,
                        )

            site = getattr(self, "_site", None)
            if site is not None:
                try:
                    await site.stop()
                except Exception as e:
                    print(f"[EXIT] Error stopping aiohttp site: {e}", flush=True)

            # Let signalled processors finish through their normal EndFrame path.
            # Cancel only tasks that exceed the shutdown bound. Completion callbacks
            # enqueue terminal lifecycle finalizers in either case.
            if handler is not None:
                tasks = [
                    task
                    for task in handler._active_sessions.values()
                    if not task.done()
                ]
                if tasks:
                    print(f"[EXIT] Draining {len(tasks)} active session task(s)", flush=True)
                    done, pending = await asyncio.wait(tasks, timeout=10)
                    if pending:
                        print(
                            f"[EXIT] Cancelling {len(pending)} session task(s) after drain timeout",
                            flush=True,
                        )
                        for task in pending:
                            task.cancel()
                        cancelled_done, stuck = await asyncio.wait(pending, timeout=3)
                        done.update(cancelled_done)
                        if stuck:
                            print(
                                f"[EXIT] {len(stuck)} session task(s) ignored cancellation",
                                flush=True,
                            )
                    for task in done:
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            task.result()

                # Let processor done callbacks register finalizers, then drain the
                # terminal callbacks before the loop and aiohttp runner close.
                await asyncio.sleep(0)
                finalizers_drained = await handler.drain_finalization_tasks(timeout=10)
                if not finalizers_drained:
                    print("[EXIT] Timed out draining worker finalization task(s)", flush=True)

                # Sweep reconnect-expiry watchdogs that outlived their session
                # tracking (sessions ended between disconnect and the reconnect
                # deadline). They must not be destroyed pending at loop teardown.
                expiry_tasks = [
                    task
                    for task in asyncio.all_tasks()
                    if task.get_name().startswith("reconnect-expiry-")
                    and not task.done()
                ]
                if expiry_tasks:
                    print(
                        f"[EXIT] Cancelling {len(expiry_tasks)} orphaned reconnect-expiry task(s)",
                        flush=True,
                    )
                    for task in expiry_tasks:
                        task.cancel()
                    await asyncio.wait(expiry_tasks, timeout=2)

            warmup_task = getattr(self, "_batched_warmup_task", None)
            warmup_stop = getattr(self, "_batched_warmup_stop", None)
            if warmup_stop is not None:
                warmup_stop.set()
            if warmup_task is not None and not warmup_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(warmup_task), timeout=12)
                except asyncio.TimeoutError:
                    print("[EXIT] Timed out waiting for batched warmup thread", flush=True)
                    warmup_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await warmup_task

            # Stop the UDP heartbeat (thread-safe Event, no loop needed).
            try:
                self.stop_udp_heartbeat()
            except Exception:
                pass

            runner = getattr(self, "_runner", None)
            if runner is not None:
                try:
                    await runner.cleanup()
                except Exception as e:
                    print(f"[EXIT] Error cleaning up aiohttp runner: {e}", flush=True)
        finally:
            print("[EXIT] Modal serve runtime shutdown complete", flush=True)

    @modal.exit()
    def cleanup(self):
        """Modal-only graceful shutdown.

        Stops the metrics logger, signals the aiohttp serve loop to cancel and
        await active session tasks, then stops the batched engine. Runs only
        from @modal.exit(); normal worker session/reconnect/provider
        semantics are unchanged.
        """
        print("[EXIT] Starting graceful shutdown", flush=True)

        # Stop the background metrics logger first so it does not continue
        # printing after shutdown begins.
        metrics_stop = getattr(self, "_metrics_stop_event", None)
        metrics_thread = getattr(self, "_metrics_thread", None)
        if metrics_stop is not None:
            metrics_stop.set()
        if metrics_thread is not None and metrics_thread.is_alive():
            metrics_thread.join(timeout=2)

        # Signal the serve loop to shut down. _shutdown_runtime runs on that
        # loop and cancels active session tasks, stops the WebSocket server,
        # and cleans up the aiohttp runner.
        serve_thread = getattr(self, "_serve_thread", None)
        with self._serve_controls_lock:
            self._serve_shutdown_requested = True
            serve_loop = getattr(self, "_serve_loop", None)
            stop_event = getattr(self, "_serve_stop_event", None)
            if serve_loop is not None and stop_event is not None and not serve_loop.is_closed():
                try:
                    serve_loop.call_soon_threadsafe(stop_event.set)
                except RuntimeError:
                    pass

        warmup_stop = getattr(self, "_batched_warmup_stop", None)
        if warmup_stop is not None:
            warmup_stop.set()

        try:
            self.stop_udp_heartbeat()
        except Exception:
            pass

        if serve_thread is not None:
            serve_thread.join(timeout=60)

        warmup_thread = getattr(self, "_batched_warmup_thread", None)
        if warmup_thread is not None and warmup_thread.is_alive():
            warmup_thread.join(timeout=15)

        serve_stopped = serve_thread is None or not serve_thread.is_alive()
        warmup_stopped = warmup_thread is None or not warmup_thread.is_alive()
        if not serve_stopped:
            print("[EXIT] Serve thread did not stop before shutdown deadline", flush=True)
        if not warmup_stopped:
            print("[EXIT] Warmup thread did not stop before shutdown deadline", flush=True)

        # Stop the shared engine only after all code that can access it has exited.
        # If either thread ignores its bound, process teardown is safer than racing
        # that thread against native engine destruction.
        try:
            if (
                serve_stopped
                and warmup_stopped
                and hasattr(self, "_handler")
                and getattr(self._handler, "batched_engine", None) is not None
            ):
                self._handler.batched_engine.stop()
                print("[EXIT] BatchedStreamingEngine stopped", flush=True)
            elif not (serve_stopped and warmup_stopped):
                print("[EXIT] Skipping engine stop while worker thread is active", flush=True)
        except Exception as e:
            print(f"[EXIT] Error stopping engine: {e}", flush=True)

        print("[EXIT] Graceful shutdown complete", flush=True)
