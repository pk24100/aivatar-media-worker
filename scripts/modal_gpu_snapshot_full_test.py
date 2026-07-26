"""
Phase 3: Full worker GPU snapshot test.

Based on modal_app.py but uses GPU memory snapshots instead of CPU snapshots.
- Models loaded directly to GPU in @modal.enter(snap=True)
- Warmup inference run before snapshot (CUDA kernels captured in snapshot)
- On restore: verify CUDA state, start serving immediately (no model move/warmup)
- Lazy xfuser patch applied via flash_head_model_snapshot_patch.py overlay
- UCX/NCCL env vars set at image level to prevent SIGSEGV

Deploy: modal deploy modal_gpu_snapshot_full_test.py
"""

import os
import logging
import re
import signal
import faulthandler
import time

import modal


APP_NAME = "aivatar-gpu-snapshot-full-test"
WORKER_CONCURRENCY = 1
PREWARM_CONNECT_TIMEOUT_SECONDS = 8.0
PREWARM_MAX_ATTEMPTS = 3
PREWARM_RETRY_DELAY_SECONDS = 1.0


class _ShortenLiveKitWebSocketUrlFilter(logging.Filter):
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


def _install_signal_handlers():
    faulthandler.enable()
    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        try:
            signal.signal(sig, lambda signum, frame: (
                print(f"\n[CRASH] signal={signum} pid={os.getpid()}", flush=True),
                faulthandler.dump_traceback(),
                os._exit(1),
            ))
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


image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")
    .env({
        "AIVATAR_WORKER_CONCURRENCY": str(WORKER_CONCURRENCY),
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

app = modal.App(APP_NAME, image=image)


worker_cls_config = {
    "gpu": "L40S",
    "min_containers": 0,
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

        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        print("[SNAP_CREATE] Starting GPU snapshot load", flush=True)
        _log_cuda_state("SNAP_CREATE_START")

        # Import ONLY flash_head (torch + model) - NOT handler/livekit.
        # livekit's Rust FFI spawns background threads that corrupt GPU
        # memory snapshot state. handler.py is deferred to serve() (post-restore).
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

        # Release warmup artifacts and reserved-but-unused CUDA memory.
        # This reduces snapshot size by ~1.6GB (reserved vs allocated gap).
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
        print("[SNAP_CREATE] Ready for GPU snapshot (memory optimized)", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        import sys
        sys.path.insert(0, "/app")

        _install_signal_handlers()

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

        # Verify pre-loaded pipeline survived the snapshot/restore
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

    async def _connect_prewarm_room(self, slot, require_connection=False):
        import asyncio
        import time
        import uuid

        livekit_url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")

        if not livekit_url or not api_key or not api_secret:
            raise RuntimeError("LIVEKIT_URL/API_KEY/API_SECRET not set")

        pool_size = self._handler.WORKER_POOL_SIZE
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
                .with_name("Modal GPU Snapshot Pre-Warm")
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
                    f"slot={slot + 1}/{pool_size} attempt={attempt}/{attempt_limit}...",
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
                    f"[prewarm] Connected slot={slot + 1}/{pool_size} in {elapsed_ms:.1f} ms, adding to pool",
                    flush=True,
                )
                await prewarm_pool.add(room_name, room, f"prewarm-{room_name}", token)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                elapsed_ms = (time.monotonic() - started_at) * 1000
                print(
                    f"[prewarm] Failed slot={slot + 1}/{pool_size} attempt={attempt}/{attempt_limit} "
                    f"after {elapsed_ms:.1f} ms: {exc}",
                    flush=True,
                )
                if not require_connection and attempt >= PREWARM_MAX_ATTEMPTS:
                    return False
                retry_delay = min(PREWARM_RETRY_DELAY_SECONDS * attempt, 10.0)
                print(
                    f"[prewarm] Retrying slot={slot + 1}/{pool_size} in {retry_delay:.1f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
        return False

    async def _prewarm_remaining_rooms(self):
        import asyncio

        pool_size = self._handler.WORKER_POOL_SIZE
        prewarm_pool = self._handler.prewarm_pool
        connected = await asyncio.gather(
            *(self._connect_prewarm_room(slot) for slot in range(1, pool_size))
        )
        print(
            f"[prewarm] Background fill finished: {1 + sum(connected)}/{pool_size} rooms connected, "
            f"{prewarm_pool.available_count()}/{pool_size} rooms available",
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

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )
        for hdlr in logging.getLogger().handlers:
            hdlr.addFilter(_ShortenLiveKitWebSocketUrlFilter())
        logger = logging.getLogger("modal_gpu_snapshot_full_test")

        # Import handler post-restore. livekit's Rust FFI threads won't be
        # in the snapshot. Monkey-patch get_pipeline to return our GPU
        # snapshot pipeline so handler's model_pool loads it directly -
        # zero wasted CPU pipeline load.
        import flash_head.inference as fhi
        _orig_get_pipeline = fhi.get_pipeline
        fhi.get_pipeline = lambda *a, **kw: self._snap_pipeline

        os.environ["AIVATAR_WORKER_CONCURRENCY"] = str(WORKER_CONCURRENCY)
        import handler
        self._handler = handler

        # Restore original get_pipeline for any future calls
        fhi.get_pipeline = _orig_get_pipeline

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
                self._handler.prewarm_pool = self._handler.PrewarmRoomPool(
                    size=WORKER_CONCURRENCY
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

                self._prewarm_task = asyncio.create_task(self._prewarm_remaining_rooms())

                if getattr(self, "_prewarm_cleanup_task", None) is None:
                    async def _cleanup_loop():
                        while True:
                            await asyncio.sleep(10)
                            await self._handler.prewarm_pool.cleanup_expired()

                    self._prewarm_cleanup_task = asyncio.create_task(_cleanup_loop())

                logger.info("aiohttp site started thread=%s event_loop=%s", threading.get_ident(), id(asyncio.get_event_loop()))
                await asyncio.Event().wait()

            loop.run_until_complete(_start())

        thread = threading.Thread(target=_run, daemon=True, name="modal-aiohttp-server")
        thread.start()
        logger.info("aiohttp background thread started ident=%s", thread.ident)
