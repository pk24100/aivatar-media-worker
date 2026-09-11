"""
Modal STRESS TEST wrapper for the AiVatar media worker.

Shares the lifecycle modules with production (modal_app.py): WorkerBase from
modal_worker/worker_base.py provides load/restore/serve scaffolding; this file
adds stress-specific behavior on top.

Changes from production (modal_app.py):
- App name: aivatar-worker-stress (separate from production)
- WORKER_CONCURRENCY default: 15 (vs 3 in production)
- GPU: Set STRESS_TARGET_GPU env var or edit TARGET_GPU below (H100, RTX-PRO-6000, L40S, A100-40GB, A10)
- SageAttention: Built per-GPU CUDA arch from GPU_CONFIG dict (patched source
  build adding sm100/sm103 dispatch for B200/B300)
- Optimizations: SageAttention 2.2.0 (primary attention kernel)
- torch.compile DISABLED (causes FX symbolic tracing crashes with FlashHeadPipeline)
- STRESS_ENABLE_COMPILE env var controls torch.compile (default: "0" = disabled)
- min_containers: 1 (warm container for test start)
- Prewarm pool: 1 room only (test script mints rooms for sessions 2-N)
- /readyz: prewarm check relaxed (stress test mints rooms, prewarm not required)
- CPU memory snapshots: ENABLED (load() runs on CPU, restore() moves to GPU + warmup)
- GPU memory snapshots: DISABLED (corrupts NVENC hardware state on Blackwell sm120)
- Snapshot flow: load() -> CPU pipeline + caches -> snapshot -> restore() -> move_to_device(cuda) + warmup
- Enhanced crash diagnostics: local signal handlers dump Python + C backtraces,
  thread state, and UCX library maps (superset of utils.modal_diagnostics).

Deploy:
    modal deploy modal_app_stress.py

Test:
    python scripts/modal_concurrent_test.py \
        --modal-url https://<stress-app-url>.modal.run \
        --wav test_audio.wav --sessions 5 --stagger 1.0

    Incrementally increase --sessions: 3, 5, 7, 10, etc.
    Watch for FPS degradation in the verdict line.
"""

import faulthandler
import logging
import os
import re
import signal
import sys
import time

import modal

sys.path.insert(0, "/app")

from utils.prewarm_pool import PrewarmRoomPool

from modal_worker.gpu_config import resolve_gpu
from modal_worker.image import build_worker_image
from modal_worker.warmup import run_batched_warmup
from modal_worker.worker_base import WorkerBase

WORKER_APP_NAME = os.getenv("AIVATAR_MODAL_APP_NAME", "aivatar-worker-stress")

# ---------------------------------------------------------------------------
# GPU CONFIG - Change STRESS_TARGET_GPU to switch GPUs.
# ---------------------------------------------------------------------------
TARGET_GPU = os.getenv("STRESS_TARGET_GPU", "B300")
_gpu_cfg = resolve_gpu(TARGET_GPU)
print(f"[STRESS] Target GPU: {TARGET_GPU} (arch={_gpu_cfg['cuda_arch']}, codec={_gpu_cfg['video_codec']})", flush=True)

WORKER_CONCURRENCY = int(os.getenv("STRESS_WORKER_CONCURRENCY", "15"))

PREWARM_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("AIVATAR_PREWARM_CONNECT_TIMEOUT_SECONDS", "8")
)
PREWARM_MAX_ATTEMPTS = max(1, int(os.getenv("AIVATAR_PREWARM_MAX_ATTEMPTS", "3")))
PREWARM_RETRY_DELAY_SECONDS = float(
    os.getenv("AIVATAR_PREWARM_RETRY_DELAY_SECONDS", "1")
)

# Stress test: only 1 prewarm room. Test script mints rooms for sessions 2-N.
STRESS_PREWARM_SIZE = 1


class _ShortenLiveKitWebSocketUrlFilter(logging.Filter):
    """Keep LiveKit signaling logs useful without exposing long query payloads."""

    _url_pattern = re.compile(r"\bwss?://[^\s?]+(?:\?[^\s]*)?")

    def filter(self, record):
        if not record.name.startswith("livekit"):
            return True

        message = record.getMessage()

        def _shorten(match):
            return match.group(0).split("?", 1)[0]

        shortened = self._url_pattern.sub(_shorten, message)
        if shortened != message:
            record.msg = shortened
            record.args = ()
        return True


def _crash_handler(signum, frame):
    """Enhanced crash handler: dumps Python + C backtraces, thread state, UCX info."""
    import threading
    print(f"\n{'='*60}", flush=True)
    print(f"[CRASH] signal={signum} pid={os.getpid()} thread={threading.current_thread().name}", flush=True)
    print(f"[CRASH] Frame: {frame}", flush=True)
    print(f"\n[CRASH] === Python traceback (all threads) ===", flush=True)
    faulthandler.dump_traceback()
    print(f"\n[CRASH] === Thread enumeration ===", flush=True)
    for t in threading.enumerate():
        print(f"  thread: {t.name} ident={t.ident} daemon={t.daemon} alive={t.is_alive()}", flush=True)
    print(f"\n[CRASH] === Loaded shared libraries (UCX/NCCL/CUDA) ===", flush=True)
    try:
        with open("/proc/self/maps", "r") as f:
            for line in f:
                if any(k in line.lower() for k in ["libucs", "libucp", "libnccl", "libcuda", "libuct", "libtorch_cuda"]):
                    print(f"  {line.rstrip()}", flush=True)
    except Exception as e:
        print(f"  maps read failed: {e}", flush=True)
    print(f"\n{'='*60}", flush=True)
    os._exit(1)


def _reset_ucx_signal_handlers():
    """Reset signal handlers to SIG_DFL after UCX/NCCL libraries are loaded.

    import torch loads libtorch_cuda.so -> libnccl.so -> libucs.so -> libucp.so
    via shared library DT_NEEDED dependencies. UCX installs custom signal handlers
    at library load time (ELF constructors). These handlers corrupt GPU snapshot
    state during CRIU restore when @modal.concurrent(max_inputs > 1) creates
    additional threads, causing SIGSEGV in libucs.so.0.

    By resetting to SIG_DFL after all imports, we ensure no UCX signal handlers
    are active during CRIU restore.
    """
    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        try:
            current = signal.getsignal(sig)
            if current != signal.SIG_DFL:
                print(f"[SNAP] Resetting signal {sig} from {current} to SIG_DFL", flush=True)
                signal.signal(sig, signal.SIG_DFL)
        except (OSError, ValueError):
            pass


def _install_signal_handlers():
    faulthandler.enable()
    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        try:
            signal.signal(sig, _crash_handler)
        except (OSError, ValueError):
            pass


def _log_cuda_state(label):
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            snapshot = torch.cuda.memory_stats().get("snapshot.all.current", 0)
            print(
                f"[{label}] pid={os.getpid()} cuda_available=True "
                f"allocated={allocated} reserved={reserved} "
                f"memory_snapshot_segments={snapshot}",
                flush=True,
            )
        else:
            print(f"[{label}] pid={os.getpid()} cuda_available=False", flush=True)
    except Exception as e:
        print(f"[{label}] error logging CUDA state: {e}", flush=True)


def _log_attention_backend(label):
    """Log which attention backend is available for debugging."""
    try:
        import flash_attn
        print(f"[{label}] flash_attn version: {flash_attn.__version__}", flush=True)
    except ImportError:
        print(f"[{label}] flash_attn NOT installed - will use SDPA fallback", flush=True)
    try:
        import flash_attn_interface
        print(f"[{label}] flash_attn_interface (FA3) available", flush=True)
    except ImportError:
        print(f"[{label}] flash_attn_interface (FA3) NOT available", flush=True)
    try:
        from sageattention import sageattn
        print(f"[{label}] sageattention available", flush=True)
    except ImportError:
        print(f"[{label}] sageattention NOT installed", flush=True)


def _log_signal_handlers(label):
    """Log current signal handler state for debugging GPU snapshot issues."""
    for sig_name, sig_val in [
        ("SIGSEGV", signal.SIGSEGV),
        ("SIGABRT", signal.SIGABRT),
        ("SIGBUS", signal.SIGBUS),
        ("SIGFPE", signal.SIGFPE),
    ]:
        try:
            handler = signal.getsignal(sig_val)
            print(f"[{label}] {sig_name} handler: {handler}", flush=True)
        except Exception as e:
            print(f"[{label}] {sig_name} handler: error={e}", flush=True)


METRICS_INTERVAL_SECONDS = float(os.getenv("STRESS_METRICS_INTERVAL", "5"))


def _metrics_logger(handler_ref, interval=METRICS_INTERVAL_SECONDS):
    """Background thread that logs GPU/CPU/RAM/session metrics to stdout.

    Runs in a dedicated thread (not an asyncio task) so it is not affected
    by event loop blocking from LiveKit native SDK calls.
    Shows up in `modal app logs --follow` alongside other logs.
    """
    import torch
    import resource
    import time as _time

    while True:
        try:
            # GPU metrics
            gpu_util = "n/a"
            gpu_mem_alloc = 0
            gpu_mem_reserved = 0
            gpu_mem_total = 0
            gpu_temp = "n/a"

            if torch.cuda.is_available():
                gpu_mem_alloc = torch.cuda.memory_allocated()
                gpu_mem_reserved = torch.cuda.memory_reserved()
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory

                # Try nvidia-smi for utilization + temperature (subprocess, ~50ms)
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,temperature.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0:
                        parts = result.stdout.strip().split(", ")
                        gpu_util = f"{parts[0]}%"
                        gpu_temp = f"{parts[1]}C"
                except Exception:
                    pass

            # CPU + RAM (process-level)
            ru = resource.getrusage(resource.RUSAGE_SELF)
            cpu_user = ru.ru_utime
            cpu_sys = ru.ru_stime
            rss_mb = ru.ru_maxrss / 1024  # KB -> MB on Linux

            # Active sessions from handler
            active_sessions = len(handler_ref._active_sessions)
            pool_avail = handler_ref.model_pool.get_available_count()
            pool_size = handler_ref.model_pool.max_size

            # Format GPU memory in GB
            gpu_alloc_gb = gpu_mem_alloc / (1024 ** 3)
            gpu_res_gb = gpu_mem_reserved / (1024 ** 3)
            gpu_total_gb = gpu_mem_total / (1024 ** 3)

            print(
                f"[METRICS] GPU_util={gpu_util} GPU_temp={gpu_temp} "
                f"GPU_mem={gpu_alloc_gb:.1f}GB/{gpu_total_gb:.1f}GB "
                f"(reserved={gpu_res_gb:.1f}GB) "
                f"RSS={rss_mb:.0f}MB "
                f"CPU_user={cpu_user:.1f}s CPU_sys={cpu_sys:.1f}s "
                f"sessions={active_sessions} "
                f"pool={pool_avail}/{pool_size}",
                flush=True,
            )
        except Exception as e:
            print(f"[METRICS] error: {e}", flush=True)

        _time.sleep(interval)


# SageAttention 2.2.0 - primary attention kernel for stress test.
# Build for the target GPU's CUDA arch only. Mixing archs (e.g. 12.0 + 8.9)
# causes fat-binary generation issues and runtime cudaErrorNoKernelImageForDevice.
# flash_attn NOT installed - SageAttention is faster (INT8 QK quantization)
# and the code falls back to SDPA if SageAttention is unavailable.
#
# For B200/B300 (sm100/sm103): SageAttention 2.2.0's setup.py compiles the
# _qattn_sm89 binary WITH sm100 kernels, but core.py's sageattn() dispatch
# is missing an sm100/sm103 branch. We clone, patch core.py to add the
# dispatch, then install from the patched source.
#
# B300 (sm103) requires native sm103a build - compute_100a PTX is NOT
# forward-compatible to sm103 due to the 'a' suffix (NVIDIA Blackwell
# Compatibility Guide). We patch setup.py to add a 10.3 -> sm103a branch.
_SAGEATTENTION_CMD = (
    f"git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention && "
    f"cd /tmp/sageattention && "
    f"git checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 && "
    f"python -c \""
    f"path = 'sageattention/core.py';"
    f"content = open(path).read();"
    f"old = '    elif arch == \\\"sm121\\\":';"
    f"new = '    elif arch == \\\"sm100\\\" or arch == \\\"sm103\\\":\\n"
    f"        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, "
    f"is_causal=is_causal, qk_quant_gran=\\\"per_warp\\\", sm_scale=sm_scale, "
    f"return_lse=return_lse, pv_accum_dtype=\\\"fp32+fp16\\\")  "
    f"# sm100/sm103 (B200/B300) datacenter Blackwell\\n    elif arch == \\\"sm121\\\":';"
    f"content = content.replace(old, new, 1);"
    f"assert 'sm100' in content, 'core.py patch FAILED - sm100 dispatch not found';"
    f"open(path, 'w').write(content);"
    f"path = 'setup.py';"
    f"content = open(path).read();"
    f"old = '        elif capability.startswith(\\\"12.0\\\"):';"
    f"new = '        elif capability.startswith(\\\"10.3\\\"):\\n            HAS_SM100 = True\\n            num = \\\"103a\\\"\\n        elif capability.startswith(\\\"12.0\\\"):';"
    f"content = content.replace(old, new, 1);"
    f"assert '10.3' in content, 'setup.py patch FAILED - sm103 branch not found';"
    f"open(path, 'w').write(content);"
    f"print('core.py + setup.py patched for sm100/sm103')\" && "
    f"TORCH_CUDA_ARCH_LIST='{_gpu_cfg['cuda_arch']}' MAX_JOBS=8 NVCC_THREADS=4 "
    f"pip install /tmp/sageattention --no-build-isolation "
    f"|| echo 'SAGEATTN_INSTALL_FAILED'"
)

image = build_worker_image(
    sageattention_cmd=_SAGEATTENTION_CMD,
    extra_pip=("agora-python-server-sdk==2.4.9",),
)

app = modal.App(WORKER_APP_NAME, image=image)


@app.cls(
    gpu=_gpu_cfg["modal_gpu"],
    image=image,
    min_containers=0,
    scaledown_window=15,
    timeout=1800,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("livekit-secret"),
        modal.Secret.from_name("aivatar-worker-secret"),
        modal.Secret.from_name("aivatar-idle-video-r2"),
    ],
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=WORKER_CONCURRENCY, target_inputs=WORKER_CONCURRENCY)
class Worker(WorkerBase):
    """Stress test worker: single tier, env-driven GPU + concurrency."""

    LOG_PREFIX = "[STRESS]"
    TARGET_GPU = TARGET_GPU
    WORKER_CONCURRENCY = WORKER_CONCURRENCY

    # Inject the enhanced stress diagnostics into the shared lifecycle.
    _diag_log_attention_backend = staticmethod(_log_attention_backend)
    _diag_log_cuda_state = staticmethod(_log_cuda_state)
    _diag_log_signal_handlers = staticmethod(_log_signal_handlers)
    _diag_reset_ucx_signal_handlers = staticmethod(_reset_ucx_signal_handlers)

    async def _connect_prewarm_room(self, slot, require_connection=False):
        """Connect one room on the aiohttp loop that will later publish to it."""
        import asyncio
        import time
        import uuid

        livekit_url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")

        if not livekit_url or not api_key or not api_secret:
            raise RuntimeError("LIVEKIT_URL/API_KEY/API_SECRET not set")

        from livekit import rtc
        from livekit.api import AccessToken, VideoGrants

        prewarm_pool = self._handler.prewarm_pool

        attempt = 0
        while require_connection or attempt < PREWARM_MAX_ATTEMPTS:
            attempt += 1
            room_name = f"prewarm-{uuid.uuid4().hex[:8]}"
            token = (
                AccessToken(api_key, api_secret)
                .with_identity(f"prewarm-{room_name}")
                .with_name("Modal Pre-Warm (Stress)")
                .with_grants(
                    VideoGrants(
                        room_join=True,
                        room=room_name,
                        can_publish=True,
                        can_subscribe=True,
                    )
                )
            ).to_jwt()
            room = rtc.Room()
            started_at = time.monotonic()
            try:
                attempt_limit = "required" if require_connection else str(PREWARM_MAX_ATTEMPTS)
                print(
                    f"[prewarm] Connecting to {livekit_url} room={room_name} "
                    f"slot={slot + 1}/{STRESS_PREWARM_SIZE} attempt={attempt}/{attempt_limit}...",
                    flush=True,
                )
                await room.connect(
                    livekit_url,
                    token,
                    options=rtc.RoomOptions(
                        auto_subscribe=False,
                        single_peer_connection=True,
                        connect_timeout=PREWARM_CONNECT_TIMEOUT_SECONDS,
                    ),
                )
                elapsed_ms = (time.monotonic() - started_at) * 1000
                print(
                    f"[prewarm] Connected slot={slot + 1}/{STRESS_PREWARM_SIZE} in {elapsed_ms:.1f} ms, adding to pool",
                    flush=True,
                )
                await prewarm_pool.add(room_name, room, f"prewarm-{room_name}", token)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                elapsed_ms = (time.monotonic() - started_at) * 1000
                print(
                    f"[prewarm] Failed slot={slot + 1}/{STRESS_PREWARM_SIZE} attempt={attempt}/{attempt_limit} "
                    f"after {elapsed_ms:.1f} ms: {exc}",
                    flush=True,
                )
                if not require_connection and attempt >= PREWARM_MAX_ATTEMPTS:
                    return False
                retry_delay = min(PREWARM_RETRY_DELAY_SECONDS * attempt, 10.0)
                print(
                    f"[prewarm] Retrying slot={slot + 1}/{STRESS_PREWARM_SIZE} in {retry_delay:.1f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
        return False

    async def _prewarm_remaining_rooms(self):
        """Stress test: only 1 prewarm room total. No background fill needed."""
        print(
            f"[STRESS] Prewarm pool size={STRESS_PREWARM_SIZE}. "
            f"Test script will mint rooms for sessions 2-N.",
            flush=True,
        )

    @modal.web_server(8000, startup_timeout=600)
    def serve(self):
        import sys
        sys.path.insert(0, "/app")
        import asyncio
        import threading
        from aiohttp import web

        logger = self._serve_common("modal_app_stress")

        # Stress test: enhanced crash signal handlers (thread dumps, UCX maps).
        _install_signal_handlers()

        print(f"[SERVE] Starting serve() post-restore (STRESS TEST, concurrency={WORKER_CONCURRENCY})", flush=True)

        handler = self._init_handler_with_pipeline()

        # Suppress per-step denoise timing prints from flash_head_pipeline.py
        # (fires many times per session, floods logs during stress testing)
        import builtins
        _orig_print = builtins.print

        def _filtered_print(*args, **kwargs):
            try:
                msg = " ".join(str(a) for a in args)
                if "model denoise per step" in msg:
                    return
            except Exception:
                pass
            _orig_print(*args, **kwargs)

        builtins.print = _filtered_print

        # Stress test: relax /readyz to not require prewarm rooms.
        # The test script mints rooms for sessions 2-N, so prewarm pool
        # exhaustion (size=1) should not block readiness checks.
        _orig_pod_ready = handler.pod_ready

        async def _stress_pod_ready(request):
            from aiohttp import web
            server_ready = handler.ws_server.is_running
            models_ready = (
                handler._model_ready_event is None
                or (handler._model_ready_event.is_set()
                    and handler._model_init_error is None)
            )
            prewarm_rooms_available = handler.prewarm_pool.available_count()
            ready = server_ready and models_ready  # prewarm not required
            return web.json_response({
                "ready": ready,
                "status": "READY" if ready else "STARTING",
                "serverReady": server_ready,
                "modelsReady": models_ready,
                "prewarmReady": prewarm_rooms_available > 0,
                "runtimeMode": "load_balancer",
                "poolSize": handler.WORKER_POOL_SIZE,
                "availablePipelines": handler.model_pool.get_available_count(),
                "activeSessions": len(handler._active_sessions),
                "prewarmRoomsAvailable": prewarm_rooms_available,
            }, status=200 if ready else 503)

        handler.pod_ready = _stress_pod_ready
        print("[STRESS] /readyz patched: prewarm check relaxed", flush=True)

        # Log attention backend availability post-restore
        _log_attention_backend("SERVE_POST_RESTORE")

        # Import build_app (handler already imported above in serve())
        from app_factory import build_app

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _start():
                # Stress test: prewarm pool size = 1 (not WORKER_POOL_SIZE)
                self._handler.prewarm_pool = PrewarmRoomPool(
                    size=STRESS_PREWARM_SIZE
                )

                async def _stress_claim_room(_request):
                    entry = self._handler.prewarm_pool.claim()
                    if entry is None:
                        return web.json_response({
                            "roomName": None,
                            "workerToken": None,
                            "clientToken": None,
                        })
                    try:
                        worker_token = self._handler._mint_livekit_token(
                            entry["roomName"],
                            can_publish=True,
                            can_subscribe=True,
                            identity=f"facemode-worker-{entry['roomName']}",
                        )
                        client_token = self._handler._mint_livekit_token(
                            entry["roomName"],
                            can_publish=False,
                            can_subscribe=True,
                            identity=f"viewer-{entry['roomName']}",
                        )
                    except Exception as exc:
                        await self._handler.prewarm_pool.release(entry["roomName"])
                        return web.json_response({
                            "roomName": None,
                            "workerToken": None,
                            "clientToken": None,
                        }, status=500, reason=str(exc))
                    return web.json_response({
                        "roomName": entry["roomName"],
                        "workerToken": worker_token,
                        "clientToken": client_token,
                    })

                self._handler.configure_model_readiness(ready=False)
                application = await build_app()
                application.router.add_post("/room/claim", _stress_claim_room)

                print("[SERVE] Models on GPU after CPU snapshot restore - marking ready", flush=True)
                _log_cuda_state("SERVE_START")
                self._handler.mark_model_ready()

                first_prewarm_task = asyncio.create_task(
                    self._connect_prewarm_room(slot=0, require_connection=True)
                )
                await first_prewarm_task
                if self._handler._model_init_error is not None:
                    raise RuntimeError("Model init error") from self._handler._model_init_error

                runner = web.AppRunner(application)
                await runner.setup()
                site = web.TCPSite(runner, "0.0.0.0", 8000)
                await site.start()

                # Batched warmup: trigger CUDA kernel compilation for
                # run_pipeline_batch via asyncio.to_thread so the event loop
                # stays free and the HTTP server can accept /readyz checks.
                asyncio.create_task(
                    asyncio.to_thread(
                        run_batched_warmup,
                        self._handler.batched_engine,
                        None,
                        "[STRESS]",
                    )
                )

                # Load remaining pipelines in background (if any)
                self._pipeline_fill_task = asyncio.create_task(
                    self._load_remaining_pipelines()
                )
                # Stress test: no background prewarm fill
                self._prewarm_task = asyncio.create_task(self._prewarm_remaining_rooms())

                if getattr(self, "_prewarm_cleanup_task", None) is None:
                    async def _cleanup_loop():
                        while True:
                            await asyncio.sleep(10)
                            await self._handler.prewarm_pool.cleanup_expired()

                    self._prewarm_cleanup_task = asyncio.create_task(_cleanup_loop())

                # Stress test: metrics logger every 5s (background thread, not asyncio task)
                import threading as _threading
                self._metrics_thread = _threading.Thread(
                    target=_metrics_logger,
                    args=(self._handler,),
                    daemon=True,
                    name="stress-metrics",
                )
                self._metrics_thread.start()

                logger.info("aiohttp site started thread=%s event_loop=%s", threading.get_ident(), id(asyncio.get_event_loop()))
                await asyncio.Event().wait()

            loop.run_until_complete(_start())

        thread = threading.Thread(target=_run, daemon=True, name="modal-aiohttp-server")
        thread.start()
        logger.info("aiohttp background thread started ident=%s", thread.ident)

    @modal.exit()
    def cleanup(self):
        """Graceful shutdown: stop engine, drain sessions, clean up resources."""
        print("[EXIT] Starting graceful shutdown", flush=True)
        try:
            if hasattr(self, '_handler') and getattr(self._handler, 'batched_engine', None) is not None:
                self._handler.batched_engine.stop()
                print("[EXIT] BatchedStreamingEngine stopped", flush=True)
        except Exception as e:
            print(f"[EXIT] Error stopping engine: {e}", flush=True)
        try:
            if hasattr(self, '_handler') and self._handler.ws_server.is_running:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._handler.ws_server.stop())
                loop.close()
                print("[EXIT] WebSocket server stopped", flush=True)
        except Exception as e:
            print(f"[EXIT] Error stopping WS server: {e}", flush=True)
        print("[EXIT] Graceful shutdown complete", flush=True)
