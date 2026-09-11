"""
Modal serving wrapper for the AiVatar media worker.
Uses the same aiohttp app (app_factory.build_app) for Modal and local runs.

CPU memory snapshots (production stable):
- load() (@modal.enter(snap=True)): Loads pipeline to CPU (FLASHHEAD_LOAD_DEVICE=cpu),
  preloads avatar/idle caches, resets UCX signal handlers. NO handler/livekit import.
- restore() (@modal.enter(snap=False)): Pops FLASHHEAD_LOAD_DEVICE, moves pipeline
  from CPU to GPU via move_to_device(), runs warmup inference.
- serve() (post-restore): Monkey-patches get_pipeline to return snapshot pipeline,
  then imports handler. BatchedStreamingEngine started. Batched warmup in background.
- GPU_CONFIG dict drives GPU selection, CUDA arch, and video codec per target GPU.
- Two-tier architecture: WorkerLow (L40S, max_inputs=3) + WorkerHigh (RTX PRO 6000, max_inputs=7).
- target_inputs = max_inputs - 1 pre-warms containers to handle cold boot latency.
- update_autoscaler_fn: CPU-only HTTP function for dynamic max_containers control.

Shared lifecycle logic lives in modal_worker/ (worker_base.py, gpu_config.py,
image.py, warmup.py). modal_app_stress.py reuses the same modules.

Manual prerequisites:
1. modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxx
2. modal secret create livekit-secret LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... LIVEKIT_URL=wss://...
3. modal secret create aivatar-worker-secret LIVEKIT_URL=... WORKER_AUTH_SECRET=... BACKEND_INTERNAL_URL=https://api.example.com
4. modal secret create aivatar-idle-video-r2 IDLE_VIDEO_R2_ENDPOINT=... IDLE_VIDEO_R2_ACCESS_KEY_ID=... IDLE_VIDEO_R2_SECRET_ACCESS_KEY=...
5. modal deploy modal_app.py

FlashHead model weights are baked into the image at build time via
snapshot_download from pkam24100/aivatar-flashhead-model.
No Modal Volume or pre-download script needed.
"""

import asyncio
import os
import sys
import threading

import modal

sys.path.insert(0, "/app")

from utils.modal_diagnostics import (
    install_signal_handlers,
    install_denoise_filter,
    log_attention_backend,
    log_cuda_state,
    metrics_logger,
    _ENABLE_METRICS_LOG,
)

from modal_worker.image import build_autoscaler_image, build_worker_image
from modal_worker.warmup import run_batched_warmup
from modal_worker.worker_base import WorkerBase

# Backward-compatible alias (tests and tooling reference _WorkerBase).
_WorkerBase = WorkerBase

WORKER_APP_NAME = os.getenv("AIVATAR_MODAL_APP_NAME", "aivatar-worker")

# SageAttention 2.2.0 - primary attention kernel (INT8 QK quantization, faster
# than flash-attn). Source build with TORCH_CUDA_ARCH_LIST covering both
# production tiers (L40S sm89 + RTX PRO 6000 sm120). Code falls back to SDPA
# if SageAttention is unavailable.
_SAGEATTENTION_CMD = (
    "TORCH_CUDA_ARCH_LIST=8.9,12.0 MAX_JOBS=8 NVCC_THREADS=4 "
    "pip install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation "
    "|| echo 'SAGEATTN_INSTALL_FAILED'"
)
_SAGEATTENTION_VERIFY_CMD = (
    "python -c \"from sageattention import sageattn; print('sageattention OK')\" "
    "2>/dev/null || echo 'sageattention NOT available'"
)

image = (
    build_worker_image(
        sageattention_cmd=_SAGEATTENTION_CMD,
        extra_env={"LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"},
    )
    .run_commands(_SAGEATTENTION_VERIFY_CMD)
)

app = modal.App(WORKER_APP_NAME, image=image)

_worker_secrets = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("livekit-secret"),
    modal.Secret.from_name("aivatar-worker-secret"),
    modal.Secret.from_name("aivatar-idle-video-r2"),
    modal.Secret.from_name("sentry-secret"),
]


class ProductionWorker(WorkerBase):
    """Production serve() flow: aiohttp background thread with serve controls,
    batched warmup thread, metrics logger, and graceful shutdown draining.
    Both GPU tiers share this serve() implementation."""

    @modal.web_server(8000, startup_timeout=600)
    def serve(self):
        import sys
        sys.path.insert(0, "/app")
        import asyncio
        from aiohttp import web

        self._serve_controls_lock = threading.Lock()
        self._serve_shutdown_requested = False

        print("[SERVE] Starting serve() post-restore", flush=True)

        # Install production signal handlers for crash diagnostics.
        install_signal_handlers()
        try:
            from utils.errors import init_sentry, install_error_hooks
            init_sentry()
            install_error_hooks()
        except Exception:
            pass

        logger = self._serve_common("modal_app")

        # Suppress per-step denoise timing prints from flash_head_pipeline.py
        # (fires many times per session, floods logs during concurrent sessions)
        # Set AIVATAR_LOG_DENOISE_STEP=1 to re-enable denoise step timing.
        install_denoise_filter()

        # Monkey-patch get_pipeline, import handler, start batched engine.
        self._init_handler_with_pipeline()

        # Log attention backend availability post-restore
        log_attention_backend("SERVE_POST_RESTORE")

        from app_factory import build_app

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._publish_serve_controls(loop)

            async def _start():
                try:
                    self._handler.configure_model_readiness(ready=False)
                    application = await build_app()

                    # Narrow health order: build_app -> runner.setup -> site.start -> mark_ready.
                    # /health stays always ok, /readyz gates ONLY on ws_server +
                    # model_ready + batched thread. warmupDone/pool are fields only.
                    runner = web.AppRunner(application, shutdown_timeout=3.0)
                    await runner.setup()
                    site = web.TCPSite(runner, "0.0.0.0", 8000)
                    await site.start()
                    self._runner = runner
                    self._site = site

                    print("[SERVE] Models on GPU after CPU snapshot restore - marking ready", flush=True)
                    log_cuda_state("SERVE_START")
                    self._handler.mark_model_ready()
                    if self._handler._model_init_error is not None:
                        raise RuntimeError("Model init error") from self._handler._model_init_error

                    # Do not create a dummy LiveKit room here. Cancelling Room.connect()
                    # can leave native retries alive and later panic the shared FFI process.

                    # Batched warmup: trigger CUDA kernel compilation in a background
                    # thread so the event loop stays free and the HTTP server can
                    # accept /readyz checks immediately.
                    self._batched_warmup_stop = threading.Event()
                    warmup_thread = threading.Thread(
                        target=run_batched_warmup,
                        args=(self._handler.batched_engine, self._batched_warmup_stop),
                        kwargs={"log_prefix": "[SERVE]"},
                        daemon=True,
                        name="batched-warmup",
                    )
                    self._batched_warmup_thread = warmup_thread
                    # Expose warmup thread on handler so /readyz can report
                    # warmupDone as a field only (never gates readiness).
                    try:
                        self._handler._batched_warmup_thread = warmup_thread
                    except Exception:
                        pass
                    warmup_thread.start()

                    # Background UDP heartbeat: keeps gVisor netstack warm from
                    # serve() post-restore through session life. Stdlib only.
                    try:
                        self.start_udp_heartbeat()
                    except Exception as _hb_err:
                        print(f"[SERVE] UDP heartbeat failed to start: {_hb_err}", flush=True)

                    async def _wait_for_batched_warmup():
                        while warmup_thread.is_alive():
                            await asyncio.sleep(0.1)

                    self._batched_warmup_task = asyncio.create_task(
                        _wait_for_batched_warmup(),
                        name="batched-warmup-waiter",
                    )
                    try:
                        from utils.errors import make_task_guard
                        import logging as _logging
                        _gw = make_task_guard(_logging.getLogger("worker.errors"), "-", "BATCHED_WARMUP_WAITER")
                        self._batched_warmup_task.add_done_callback(_gw)
                    except Exception:
                        pass

                    # Load remaining pipelines in background (no-op with shared pipeline)
                    self._pipeline_fill_task = asyncio.create_task(
                        self._load_remaining_pipelines()
                    )
                    try:
                        from utils.errors import make_task_guard as _mtg
                        import logging as _logging2
                        _gp = _mtg(_logging2.getLogger("worker.errors"), "-", "PIPELINE_FILL")
                        self._pipeline_fill_task.add_done_callback(_gp)
                    except Exception:
                        pass
                    # Metrics logger every 5s (background thread, not asyncio task)
                    # Set AIVATAR_LOG_METRICS=0 to disable.
                    if _ENABLE_METRICS_LOG:
                        import threading as _threading
                        self._metrics_stop_event = _threading.Event()
                        self._metrics_thread = _threading.Thread(
                            target=metrics_logger,
                            args=(self._handler,),
                            kwargs={"stop_event": self._metrics_stop_event},
                            daemon=True,
                            name="metrics-logger",
                        )
                        self._metrics_thread.start()

                    logger.info("aiohttp site started thread=%s event_loop=%s", threading.get_ident(), id(asyncio.get_event_loop()))
                    await self._serve_stop_event.wait()
                finally:
                    # Modal-only cleanup: stop the server, cancel/await active
                    # session tasks, and clean up the aiohttp runner on the
                    # existing serve loop. Engine teardown is handled in cleanup().
                    try:
                        await self._shutdown_runtime()
                    except Exception as _shutdown_err:
                        print(f"[EXIT] Error in _shutdown_runtime: {_shutdown_err}", flush=True)

            try:
                loop.run_until_complete(_start())
            finally:
                # Python 3.12 compatibility: shutdown_default_executor does not
                # accept a timeout argument. Use wait_for so we never hang here.
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                try:
                    loop.run_until_complete(asyncio.wait_for(loop.shutdown_default_executor(), timeout=3))
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_run, daemon=True, name="modal-aiohttp-server")
        self._serve_thread = thread
        thread.start()
        logger.info("aiohttp background thread started ident=%s", thread.ident)


@app.cls(
    gpu="L40S",
    image=image,
    min_containers=0,
    scaledown_window=15,
    timeout=1800,
    secrets=_worker_secrets,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=3, target_inputs=2)
class WorkerLow(ProductionWorker):
    """Low-traffic tier: L40S GPU, 3 concurrent sessions per container."""

    TARGET_GPU = "L40S"
    WORKER_CONCURRENCY = 3


@app.cls(
    gpu="RTX-PRO-6000",
    image=image,
    min_containers=0,
    scaledown_window=30,
    timeout=1800,
    secrets=_worker_secrets,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=7, target_inputs=6)
class WorkerHigh(ProductionWorker):
    """High-traffic tier: RTX PRO 6000 GPU, 7 concurrent sessions per container."""

    TARGET_GPU = "RTX-PRO-6000"
    WORKER_CONCURRENCY = 7


# ---------------------------------------------------------------------------
# Autoscaler HTTP function - CPU-only, no GPU needed.
# Called by the backend orchestrator to dynamically set max_containers per tier.
# Uses modal.Cls.from_name() + update_autoscaler() to adjust caps at runtime.
# ---------------------------------------------------------------------------
_autoscaler_image = build_autoscaler_image()


@app.function(image=_autoscaler_image, scaledown_window=2)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def update_autoscaler_fn(data: dict):
    """Dynamically set max_containers for a worker tier.

    Body: {"tier": "low"|"high", "max_containers": int}
    Returns: {"tier": str, "max_containers": int}
    """
    tier = data["tier"]
    max_containers = int(data["max_containers"])
    class_name = f"Worker{tier.title()}"
    cls = modal.Cls.from_name(WORKER_APP_NAME, class_name)
    instance = cls()
    instance.update_autoscaler(max_containers=max_containers)
    print(f"[AUTOSCALER] {class_name} max_containers={max_containers}", flush=True)
    return {"tier": tier, "max_containers": max_containers}
