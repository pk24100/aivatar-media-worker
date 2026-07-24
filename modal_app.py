"""
Modal serving wrapper for the AiVatar media worker.
Uses the same aiohttp app (app_factory.build_app) as RunPod.

Manual prerequisites:
1. modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxx
2. modal secret create livekit-secret LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... LIVEKIT_URL=wss://...
3. modal secret create aivatar-worker-secret LIVEKIT_URL=...
4. modal secret create aivatar-idle-video-r2 IDLE_VIDEO_R2_ENDPOINT=... IDLE_VIDEO_R2_ACCESS_KEY_ID=... IDLE_VIDEO_R2_SECRET_ACCESS_KEY=...
5. modal deploy modal_app.py

FlashHead model weights are baked into the image at build time via
snapshot_download from pkam24100/aivatar-flashhead-model.
No Modal Volume or pre-download script needed.
"""

import os
import logging
import re
import modal


WORKER_APP_NAME = os.getenv("AIVATAR_MODAL_APP_NAME", "aivatar-worker")
WORKER_CONCURRENCY = int(os.getenv("AIVATAR_WORKER_CONCURRENCY", "3"))
if WORKER_CONCURRENCY < 1:
    raise ValueError("AIVATAR_WORKER_CONCURRENCY must be at least 1")
PREWARM_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("AIVATAR_PREWARM_CONNECT_TIMEOUT_SECONDS", "8")
)
PREWARM_MAX_ATTEMPTS = max(1, int(os.getenv("AIVATAR_PREWARM_MAX_ATTEMPTS", "3")))
PREWARM_RETRY_DELAY_SECONDS = float(
    os.getenv("AIVATAR_PREWARM_RETRY_DELAY_SECONDS", "1")
)


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

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")
    .env({"AIVATAR_WORKER_CONCURRENCY": str(WORKER_CONCURRENCY)})
    .apt_install("git", "git-lfs", "ffmpeg", "libsndfile1", "wget", "ca-certificates")
    .pip_install("ninja")
    .run_commands("pip install flash-attn --no-build-isolation || true")
    .pip_install_from_requirements("requirements.txt")
    # Download FlashHead model weights from HuggingFace into the image layer.
    # This bakes ~6GB of weights directly into the image, eliminating the
    # ~38s snapshot_download from Modal Volume at every container boot.
    # Modal caches this layer, so it's only downloaded once (on first deploy
    # or when the model changes). The huggingface-secret provides the HF token.
    .run_commands(
        "huggingface-cli download pkam24100/aivatar-flashhead-model --local-dir /app/models/SoulX-FlashHead-1_3B",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_dir("streaming", "/app/streaming", copy=True)
    .add_local_dir("utils", "/app/utils", copy=True)
    .add_local_dir("config", "/app/config", copy=True)
    .add_local_dir("SoulX-FlashHead", "/app/SoulX-FlashHead", copy=True)
    .add_local_dir("models/wav2vec2-base-960h", "/app/models/wav2vec2-base-960h", copy=True)
    .add_local_file("handler.py", "/app/handler.py", copy=True)
    .add_local_file("app_factory.py", "/app/app_factory.py", copy=True)
)

app = modal.App(WORKER_APP_NAME, image=image)
#models_volume = modal.Volume.from_name("aivatar-models", create_if_missing=True)


worker_cls_config = {
    "gpu": "L4",
    "min_containers": 1,
    "scaledown_window": 15,
    "timeout": 1800,
    #volumes={"/models": models_volume},
    "secrets": [
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("livekit-secret"),
        modal.Secret.from_name("aivatar-worker-secret"),
        modal.Secret.from_name("aivatar-idle-video-r2"),
    ],
    "enable_memory_snapshot": True,
}


@app.cls(**worker_cls_config)
@modal.concurrent(max_inputs=WORKER_CONCURRENCY)
class Worker:
    @modal.enter(snap=True)
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        # Keep the model pool aligned with the deployment's input concurrency.
        os.environ["AIVATAR_WORKER_CONCURRENCY"] = str(WORKER_CONCURRENCY)
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        # CPU snapshots are stable and intentionally make no CUDA calls.
        os.environ["FLASHHEAD_LOAD_DEVICE"] = "cpu"
        # Enable Rust FFI debug logs (ICE, DTLS) before handler import so the
        # native livekit.rtc library picks it up at initialization time.
        os.environ.setdefault("LIVEKIT_RTC_DEBUG", "false")
        # Trigger the snapshot bootstrap pipeline preload on CPU.
        import handler
        from utils.default_avatar_cache import default_avatar_cache
        from utils.default_idle_video_cache import default_idle_video_cache
        self._handler = handler
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
        print("[modal] Bootstrap pipeline loaded to CPU for snapshot", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        # The CPU-loaded bootstrap pipeline remains in the stable memory snapshot.
        # GPU transfer and CUDA warmup run after the snapshot is restored.
        os.environ.pop("FLASHHEAD_LOAD_DEVICE", None)

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

    async def _initialize_models(self):
        import asyncio
        import torch
        import time

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            print(f"[modal] Initializing bootstrap pipeline on {device} after snapshot restore", flush=True)
            move_started_at = time.monotonic()

            def _move_models():
                self._handler.model_pool.move_to_device(device)
                if device == "cuda":
                    torch.cuda.synchronize()

            await asyncio.to_thread(_move_models)
            print(
                f"[modal] Bootstrap pipeline moved to {device} in {(time.monotonic() - move_started_at) * 1000:.1f} ms",
                flush=True,
            )

            pipeline = await self._handler.model_pool.acquire()
            try:
                warm_started_at = time.monotonic()
                await asyncio.to_thread(self._warm_pipeline, pipeline)
                warm_elapsed_ms = (time.monotonic() - warm_started_at) * 1000
                print(f"[modal] CUDA warmup completed in {warm_elapsed_ms:.1f} ms", flush=True)
                print("[modal] warmup generate() completed - CUDA kernels pre-warmed", flush=True)
            finally:
                self._handler.model_pool.release(pipeline)
            self._handler.mark_model_ready()
        except Exception as exc:
            self._handler.mark_model_ready(exc)
            logging.getLogger("modal_app").exception("Model initialization failed")

    async def _load_remaining_pipelines(self):
        """Fill configured capacity after the first warmed pipeline is serving."""
        import time

        pool = self._handler.model_pool
        remaining = pool.max_size - pool.current_size
        if remaining <= 0:
            return

        print(
            f"[modal] Loading {remaining} remaining pipeline(s) in the background "
            f"to reach configured capacity {pool.max_size}",
            flush=True,
        )
        started_at = time.monotonic()
        loaded_count = await pool.load_remaining()
        print(
            f"[modal] Background pipeline fill complete: loaded={loaded_count} "
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
                .with_name("Modal Pre-Warm")
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
                # Let the native SDK own its connection timeout and retry
                # lifecycle. Cancelling Room.connect() mid-flight races its FFI
                # callback and caused the observed FFI panic.
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
        """Fill remaining capacity after the required first room is claimable."""
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

        # Configure root logger so all Python loggers (stream_processor,
        # VideoPublisher, AudioPublisher, etc.) output to stdout.
        # Without this, _logger.info() calls are silently dropped.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )
        for handler in logging.getLogger().handlers:
            handler.addFilter(_ShortenLiveKitWebSocketUrlFilter())
        logger = logging.getLogger("modal_app")

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _start():
                # Recreate the pool on the aiohttp server loop so claimed rooms
                # and session publishers share the same event-loop ownership.
                self._handler.prewarm_pool = self._handler.PrewarmRoomPool(
                    size=self._handler.WORKER_POOL_SIZE
                )
                self._handler.configure_model_readiness(ready=False)
                application = await build_app()
                self._model_init_task = asyncio.create_task(self._initialize_models())

                # Do not bind HTTP until a session can reuse a connected room.
                # CPU-model warmup and the first room connect are independent,
                # so run them concurrently on the session-owning aiohttp loop.
                first_prewarm_task = asyncio.create_task(
                    self._connect_prewarm_room(slot=0, require_connection=True)
                )
                await asyncio.gather(self._model_init_task, first_prewarm_task)
                if self._handler._model_init_error is not None:
                    raise RuntimeError("FlashHead model initialization failed") from self._handler._model_init_error

                runner = web.AppRunner(application)
                await runner.setup()
                site = web.TCPSite(runner, "0.0.0.0", 8000)
                await site.start()
                self._pipeline_fill_task = asyncio.create_task(
                    self._load_remaining_pipelines()
                )
                self._prewarm_task = asyncio.create_task(self._prewarm_remaining_rooms())

                if getattr(self, "_prewarm_cleanup_task", None) is None:
                    async def _cleanup_loop():
                        while True:
                            await asyncio.sleep(10)
                            await self._handler.prewarm_pool.cleanup_expired()

                    self._prewarm_cleanup_task = asyncio.create_task(_cleanup_loop())
                import asyncio as _aio
                logger.info("aiohttp site started thread=%s event_loop=%s", threading.get_ident(), id(_aio.get_event_loop()))
                await asyncio.Event().wait()

            loop.run_until_complete(_start())

        thread = threading.Thread(target=_run, daemon=True, name="modal-aiohttp-server")
        thread.start()
        logger.info("aiohttp background thread started ident=%s", thread.ident)
