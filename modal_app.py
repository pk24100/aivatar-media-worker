"""
Modal serving wrapper for the AiVatar media worker.
Uses the same aiohttp app (app_factory.build_app) as RunPod.

Manual prerequisites (see migration plan Section 5):
1. modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxx
2. modal secret create livekit-secret LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... LIVEKIT_URL=wss://...
3. modal secret create aivatar-worker-secret LIVEKIT_URL=...
4. modal volume create aivatar-models
5. modal deploy modal_app.py
"""

import os
import modal

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")
    .apt_install("git", "git-lfs", "ffmpeg", "libsndfile1", "wget", "ca-certificates")
    .pip_install("ninja")
    .run_commands("pip install flash-attn --no-build-isolation || true")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("streaming", "/app/streaming", copy=True)
    .add_local_dir("utils", "/app/utils", copy=True)
    .add_local_dir("SoulX-FlashHead", "/app/SoulX-FlashHead", copy=True)
    .add_local_dir("models/wav2vec2-base-960h", "/app/models/wav2vec2-base-960h", copy=True)
    .add_local_file("handler.py", "/app/handler.py", copy=True)
    .add_local_file("app_factory.py", "/app/app_factory.py", copy=True)
)

app = modal.App("aivatar-worker", image=image)
models_volume = modal.Volume.from_name("aivatar-models", create_if_missing=True)


@app.cls(
    gpu="L4",
    min_containers=1,
    scaledown_window=15,
    timeout=600,
    volumes={"/models": models_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("livekit-secret"),
        modal.Secret.from_name("aivatar-worker-secret"),
    ],
    # NOTE: Snapshots disabled — UCX/libucs segfaults on restore (signal 11).
    # Re-enable once the NGC base image / CUDA driver compat issue is resolved.
    # enable_memory_snapshot=True,
    # experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=3)
class Worker:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        # Worker concurrency is driven by env var (default 1 on Modal L4, 3 on RunPod)
        worker_concurrency = os.getenv("AIVATAR_WORKER_CONCURRENCY", "1")
        os.environ["AIVATAR_WORKER_CONCURRENCY"] = worker_concurrency
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ.setdefault("FLASHHEAD_HF_CACHE_DIR", "/models/huggingface-cache/hub")
        # Use the native livekit.rtc SDK (Rust FFI) on Modal. In web_server runtime,
        # UDP to LiveKit's IP range is blocked (error 101) but TCP (port 7881) works.
        # single_peer_connection=True in RoomOptions makes publisher share the
        # subscriber's TCP-connected PC, avoiding the 15s UDP-to-TCP fallback race.
        os.environ["AIVATAR_WEBRTC_BACKEND"] = "native"
        # Enable Rust FFI debug logs (ICE, DTLS) before handler import so the
        # native livekit.rtc library picks it up at initialization time.
        os.environ.setdefault("LIVEKIT_RTC_DEBUG", "true")
        # Trigger FlashHeadModelPool preload at import
        import handler
        self._handler = handler

        # Warm-up: model weights are already loaded by import handler above.
        # We call get_base_data to trigger any lazy pipeline initialization.
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

        # Pre-warm WebRTC routes to LiveKit media servers.
        # In Modal's web_server runtime, TCP routes to LiveKit's media IP range
        # (161.115.x.x) can take ~15s to become routable on cold containers.
        # We create N pre-warm rooms (N = AIVATAR_WORKER_CONCURRENCY) that stay
        # connected, warming the route. When a real session arrives, it claims
        # a pre-warmed room via /room/claim — same SFU node → warm route → <2s connect.
        # Pre-warm rooms auto-expire after 60s if not claimed.
        self._prewarm_webrtc_routes()

    def _prewarm_webrtc_routes(self):
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

        pool_size = int(os.getenv("AIVATAR_WORKER_CONCURRENCY", "1"))

        async def _prewarm():
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
                            can_publish=False,
                            can_subscribe=False,
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

            # Start background cleanup task for expired pre-warm rooms.
            async def _cleanup_loop():
                while True:
                    await asyncio.sleep(10)
                    await prewarm_pool.cleanup_expired()

            asyncio.create_task(_cleanup_loop())

        try:
            asyncio.run(_prewarm())
        except Exception as exc:
            print(f"[prewarm] Event loop error (non-fatal): {exc}", flush=True)

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
                application = await build_app()
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
