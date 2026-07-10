"""
Modal serving wrapper for the AiVatar media worker.
Uses the same aiohttp app (app_factory.build_app) as RunPod.

Manual prerequisites:
1. modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxx
2. modal secret create livekit-secret LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... LIVEKIT_URL=wss://...
3. modal secret create aivatar-worker-secret LIVEKIT_URL=...
# 4. modal volume create aivatar-models
# 5. modal deploy modal_app.py
4. modal deploy modal_app.py

FlashHead model weights are baked into the image at build time via
snapshot_download from pkam24100/aivatar-flashhead-model.
No Modal Volume or pre-download script needed.
"""

import os
import modal

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")
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

app = modal.App("aivatar-worker", image=image)
#models_volume = modal.Volume.from_name("aivatar-models", create_if_missing=True)


@app.cls(
    gpu="L4",
    min_containers=1,
    scaledown_window=15,
    timeout=600,
    #volumes={"/models": models_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("livekit-secret"),
        modal.Secret.from_name("aivatar-worker-secret"),
    ],
    enable_memory_snapshot=True,
    # NOTE: GPU snapshots are alpha. Using CPU-only snapshots (stable) which
    # skip disk I/O + deserialization but still pay CPU->GPU transfer.
    # Modal auto-invalidates snapshots when code or image changes (e.g. model
    # weight updates create a new image layer). No manual key needed.
)
@modal.concurrent(max_inputs=3)
class Worker:
    @modal.enter(snap=True)
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        # Worker concurrency is driven by env var (default 3 for L40S)
        worker_concurrency = os.getenv("AIVATAR_WORKER_CONCURRENCY", "3")
        os.environ["AIVATAR_WORKER_CONCURRENCY"] = worker_concurrency
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        # Load models to CPU for snapshotting (no CUDA calls before snapshot)
        os.environ["FLASHHEAD_LOAD_DEVICE"] = "cpu"
        # Enable Rust FFI debug logs (ICE, DTLS) before handler import so the
        # native livekit.rtc library picks it up at initialization time.
        os.environ.setdefault("LIVEKIT_RTC_DEBUG", "false")
        # Trigger FlashHeadModelPool preload at import - loads to CPU
        import handler
        from utils.default_avatar_cache import default_avatar_cache
        self._handler = handler
        cache_status = default_avatar_cache.preload()
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
        print("[modal] Models loaded to CPU for snapshot", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        import sys
        sys.path.insert(0, "/app")
        import torch

        # Move models from CPU to GPU after snapshot restore
        os.environ.pop("FLASHHEAD_LOAD_DEVICE", None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[modal] Restoring from snapshot, moving models to {device}", flush=True)
        self._handler.model_pool.move_to_device(device)
        print("[modal] Models moved to GPU", flush=True)

        # Warm-up: run dummy inference to pre-compile CUDA kernels
        from flash_head.inference import get_base_data, get_audio_embedding, run_pipeline, get_infer_params
        from PIL import Image
        import numpy as np

        dummy_img_path = "/tmp/warmup_avatar.png"
        Image.new("RGB", (512, 512), color=(128, 128, 128)).save(dummy_img_path)

        import asyncio

        async def _warmup():
            pipeline = await self._handler.model_pool.acquire()
            try:
                get_base_data(pipeline, dummy_img_path, base_seed=42, use_face_crop=False)
                print("[modal] pipeline params warmed up, running dummy generate() to pre-warm CUDA kernels...")

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
                print("[modal] warmup generate() completed — CUDA kernels pre-warmed, first session will be fast")
            finally:
                self._handler.model_pool.release(pipeline)

        asyncio.run(_warmup())

        # Pre-warm rooms must be created on the same aiohttp event loop that
        # later publishes tracks on them. The server loop handles that in serve().

    async def _prewarm_webrtc_routes(self):
        """Create N pre-warm LiveKit rooms and add them to the prewarm pool."""
        import asyncio
        import time
        import uuid

        livekit_url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")

        if not livekit_url or not api_key or not api_secret:
            print("[prewarm] Skipping — LIVEKIT_URL/API_KEY/API_SECRET not set", flush=True)
            return

        # Pre-warm one LiveKit room per available model pipeline so /readyz
        # prewarmRoomsAvailable matches the worker's configured concurrency.
        pool_size = self._handler.WORKER_POOL_SIZE

        from livekit import rtc
        from livekit.api import AccessToken, VideoGrants

        prewarm_pool = self._handler.prewarm_pool

        for i in range(pool_size):
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
            t0 = time.monotonic()
            try:
                print(f"[prewarm] Connecting to {livekit_url} room={room_name} ({i+1}/{pool_size})...", flush=True)
                await room.connect(
                    livekit_url,
                    token,
                    options=rtc.RoomOptions(
                        auto_subscribe=False,
                        single_peer_connection=True,
                        connect_timeout=30.0,
                    ),
                )
                elapsed = round((time.monotonic() - t0) * 1000, 1)
                print(f"[prewarm] Connected in {elapsed} ms, adding to pool", flush=True)
                await prewarm_pool.add(room_name, room, f"prewarm-{room_name}", token)
            except Exception as exc:
                print(f"[prewarm] Failed for room {room_name} (non-fatal): {exc}", flush=True)

        print(f"[prewarm] Pool ready: {prewarm_pool.available_count()}/{pool_size} rooms available", flush=True)

        if getattr(self, "_prewarm_cleanup_task", None) is None:
            async def _cleanup_loop():
                while True:
                    await asyncio.sleep(10)
                    await prewarm_pool.cleanup_expired()

            self._prewarm_cleanup_task = asyncio.create_task(_cleanup_loop())

    @modal.web_server(8000, startup_timeout=600)
    def serve(self):
        import sys
        sys.path.insert(0, "/app")
        import asyncio
        import logging
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
                application = await build_app()
                await self._prewarm_webrtc_routes()
                runner = web.AppRunner(application)
                await runner.setup()
                site = web.TCPSite(runner, "0.0.0.0", 8000)
                await site.start()
                import asyncio as _aio
                logger.info("aiohttp site started thread=%s event_loop=%s", threading.get_ident(), id(_aio.get_event_loop()))
                await asyncio.Event().wait()

            loop.run_until_complete(_start())

        thread = threading.Thread(target=_run, daemon=True, name="modal-aiohttp-server")
        thread.start()
        logger.info("aiohttp background thread started ident=%s", thread.ident)
