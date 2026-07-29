"""
Modal STRESS TEST wrapper for the AiVatar media worker.

This is a copy of modal_app.py with high concurrency settings to find the
maximum concurrent sessions a single GPU can handle before FPS degrades.

Changes from production (modal_app.py):
- App name: aivatar-worker-stress (separate from production)
- WORKER_CONCURRENCY default: 10 (vs 3 in production)
- GPU: L4 (hardcoded, change to "L40S" for L40S testing)
- min_containers: 1 (warm container for test start)
- Prewarm pool: 1 room only (test script mints rooms for sessions 2-N)
- GPU memory snapshots: ENABLED (mimics production behavior)

Deploy:
    modal deploy modal_app_stress.py

Test:
    python scripts/modal_concurrent_test.py \
        --modal-url https://<stress-app-url>.modal.run \
        --wav test_audio.wav --sessions 5 --stagger 1.0

    Incrementally increase --sessions: 3, 5, 7, 10, etc.
    Watch for FPS degradation in the verdict line.
"""

import os
import logging
import re
import signal
import faulthandler
import time
import modal


WORKER_APP_NAME = "aivatar-worker-stress"
WORKER_CONCURRENCY = int(os.getenv("AIVATAR_WORKER_CONCURRENCY", "10"))
if WORKER_CONCURRENCY < 1:
    raise ValueError("AIVATAR_WORKER_CONCURRENCY must be at least 1")
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


image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")
    .env({
        "UCX_TLS": "self",
        "UCX_NET_DEVICES": "none",
        "UCX_UNIFIED_TLS_MODE": "y",
        "UCX_MEMTYPE_CACHE": "n",
        "NCCL_DEBUG": "INFO",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
    .apt_install("git", "git-lfs", "ffmpeg", "libsndfile1", "wget", "ca-certificates")
    .pip_install("ninja")
    .run_commands("pip install flash-attn --no-build-isolation || true")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "huggingface-cli download pkam24100/aivatar-flashhead-model --local-dir /app/models/SoulX-FlashHead-1_3B",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_dir("streaming", "/app/streaming", copy=True)
    .add_local_dir("utils", "/app/utils", copy=True)
    .add_local_dir("config", "/app/config", copy=True)
    .add_local_dir("SoulX-FlashHead", "/app/SoulX-FlashHead", copy=True)
    .add_local_file("flash_head_model_snapshot_patch.py", "/app/SoulX-FlashHead/flash_head/src/modules/flash_head_model.py", copy=True)
    .add_local_dir("models/wav2vec2-base-960h", "/app/models/wav2vec2-base-960h", copy=True)
    .add_local_file("handler.py", "/app/handler.py", copy=True)
    .add_local_file("app_factory.py", "/app/app_factory.py", copy=True)
)

app = modal.App(WORKER_APP_NAME, image=image)


worker_cls_config = {
    "gpu": "L4",
    "min_containers": 1,
    "scaledown_window": 15,
    "timeout": 1800,
    "secrets": [
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("livekit-secret"),
        modal.Secret.from_name("aivatar-worker-secret"),
        modal.Secret.from_name("aivatar-idle-video-r2"),
    ],
    "enable_memory_snapshot": True,
    "experimental_options": {"enable_gpu_snapshot": True},
}


@app.cls(**worker_cls_config)
@modal.concurrent(max_inputs=WORKER_CONCURRENCY)
class Worker:
    @modal.enter(snap=True)
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/SoulX-FlashHead")

        _install_signal_handlers()

        os.environ["AIVATAR_WORKER_CONCURRENCY"] = str(WORKER_CONCURRENCY)
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        print(f"[STRESS] Snapshot load with WORKER_CONCURRENCY={WORKER_CONCURRENCY}", flush=True)
        _log_cuda_state("SNAP_CREATE_START")

        from flash_head.inference import get_pipeline

        ckpt_dir = os.getenv("FLASHHEAD_CKPT_DIR", "/app/models/SoulX-FlashHead-1_3B")
        wav2vec_dir = os.getenv("WAV2VEC_DIR", "/app/models/wav2vec2-base-960h")
        model_type = os.getenv("FLASHHEAD_MODEL_TYPE", "lite")

        print("[SNAP_CREATE] Loading FlashHead pipeline on GPU...", flush=True)
        load_t0 = time.monotonic()
        self._snap_pipeline = get_pipeline(1, ckpt_dir, model_type, wav2vec_dir)
        load_ms = (time.monotonic() - load_t0) * 1000
        print(f"[SNAP_CREATE] Pipeline loaded in {load_ms:.1f}ms", flush=True)

        _log_cuda_state("SNAP_CREATE_POST_PIPELINE_LOAD")

        print("[SNAP_CREATE] Running warmup inference on GPU...", flush=True)
        warm_t0 = time.monotonic()
        self._warm_pipeline(self._snap_pipeline)
        warm_ms = (time.monotonic() - warm_t0) * 1000
        print(f"[SNAP_CREATE] Warmup completed in {warm_ms:.1f}ms", flush=True)

        _log_cuda_state("SNAP_CREATE_POST_WARMUP")

        import gc
        import torch
        p = self._snap_pipeline
        p.cond_image_tensor_dict = {}
        p.ref_img_latent_dict = {}
        p.original_color_reference = None
        p.ref_img_latent = None
        p.latent_motion_frames = None
        p.cond_image_dict = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        _log_cuda_state("SNAP_CREATE_AFTER_CLEANUP")

        _reset_ucx_signal_handlers()
        print("[SNAP_CREATE] Ready for GPU snapshot (memory optimized)", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        import sys
        sys.path.insert(0, "/app")

        print("[RESTORE] Starting GPU snapshot restore", flush=True)
        _log_cuda_state("RESTORE_START")

        import torch
        import torch.distributed as dist
        print(
            f"[RESTORE] cuda.is_available={torch.cuda.is_available()} "
            f"dist.is_initialized={dist.is_initialized()}",
            flush=True,
        )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            snapshot = torch.cuda.memory_stats().get("snapshot.all.current", 0)
            print(
                f"[RESTORE] allocated={allocated} reserved={reserved} "
                f"memory_snapshot_segments={snapshot}",
                flush=True,
            )

        assert self._snap_pipeline is not None, "Pipeline lost in snapshot!"
        print(
            f"[RESTORE] snap_pipeline exists: device={getattr(self._snap_pipeline, 'device', 'unknown')}",
            flush=True,
        )

        _log_cuda_state("RESTORE_FINAL")
        print("[RESTORE] GPU snapshot restore complete", flush=True)

    @staticmethod
    def _warm_pipeline(pipeline):
        import torch
        from flash_head.inference import get_base_data, get_audio_embedding, run_pipeline, get_infer_params
        from PIL import Image
        import numpy as np

        dummy_img_path = "/tmp/warmup_avatar.png"
        Image.new("RGB", (512, 512), color=(128, 128, 128)).save(dummy_img_path)
        get_base_data(pipeline, dummy_img_path, base_seed=42, use_face_crop=False)
        params = get_infer_params()
        sr = params["sample_rate"]
        cached_dur = params["cached_audio_duration"]
        frame_num = params["frame_num"]
        tgt_fps = params["tgt_fps"]
        audio_end_idx = cached_dur * tgt_fps
        audio_start_idx = audio_end_idx - frame_num
        dummy_audio = np.zeros(cached_dur * sr, dtype=np.float32)
        audio_emb = get_audio_embedding(pipeline, dummy_audio, audio_start_idx, audio_end_idx)
        run_pipeline(pipeline, audio_emb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    async def _load_remaining_pipelines(self):
        """Fill configured capacity after the first warmed pipeline is serving."""
        import time

        pool = self._handler.model_pool
        remaining = pool.max_size - pool.current_size
        if remaining <= 0:
            print(
                f"[STRESS] All {pool.max_size} pool slots filled from snapshot pipeline. "
                f"No additional pipelines to load.",
                flush=True,
            )
            return

        print(
            f"[STRESS] Loading {remaining} remaining pipeline(s) in the background "
            f"to reach configured capacity {pool.max_size}",
            flush=True,
        )
        started_at = time.monotonic()
        loaded_count = await pool.load_remaining()
        print(
            f"[STRESS] Background pipeline fill complete: loaded={loaded_count} "
            f"available={pool.get_available_count()}/{pool.max_size} "
            f"in {(time.monotonic() - started_at) * 1000:.1f} ms",
            flush=True,
        )

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
        from app_factory import build_app

        print(f"[SERVE] Starting serve() post-restore (STRESS TEST, concurrency={WORKER_CONCURRENCY})", flush=True)

        _install_signal_handlers()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )
        for handler in logging.getLogger().handlers:
            handler.addFilter(_ShortenLiveKitWebSocketUrlFilter())
        logger = logging.getLogger("modal_app_stress")

        import flash_head.inference as fhi
        _orig_get_pipeline = fhi.get_pipeline
        fhi.get_pipeline = lambda *a, **kw: self._snap_pipeline

        os.environ["AIVATAR_WORKER_CONCURRENCY"] = str(WORKER_CONCURRENCY)
        os.environ.setdefault("LIVEKIT_RTC_DEBUG", "false")
        os.environ["ENGINE_PROFILE"] = "1"

        # Suppress per-step denoise timing prints from flash_head_pipeline.py
        # (fires many times per session, floods logs during stress testing)
        # Kept active for the entire serve loop - restored only on shutdown.
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

        import handler
        self._handler = handler

        fhi.get_pipeline = _orig_get_pipeline

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

        print(
            f"[SERVE] Snapshot pipeline in model_pool: "
            f"current_size={handler.model_pool.current_size} "
            f"available={handler.model_pool.get_available_count()}",
            flush=True,
        )

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _start():
                # Stress test: prewarm pool size = 1 (not WORKER_POOL_SIZE)
                self._handler.prewarm_pool = self._handler.PrewarmRoomPool(
                    size=STRESS_PREWARM_SIZE
                )
                self._handler.configure_model_readiness(ready=False)
                application = await build_app()

                print("[SERVE] Models already on GPU from snapshot - marking ready immediately", flush=True)
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
