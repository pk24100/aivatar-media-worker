"""
Modal app for asynchronous idle video generation.
Uses L4 GPU, min_containers=0 (scale-to-zero), CPU memory snapshots.

Manual prerequisites:
1. modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxx
   (shared with streaming worker)
2. modal deploy modal_idle_video_generator.py

FlashHead model weights are baked into the image at build time via
huggingface-cli download, same approach as modal_app.py.
No Dockerfile needed.
"""

import os
import logging
import modal


image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")
    .apt_install("git", "git-lfs", "ffmpeg", "libsndfile1", "wget", "ca-certificates")
    .pip_install("ninja")
    .run_commands("pip install flash-attn --no-build-isolation || true")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "huggingface-cli download pkam24100/aivatar-flashhead-model --local-dir /app/models/SoulX-FlashHead-1_3B",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_dir("SoulX-FlashHead", "/app/SoulX-FlashHead", copy=True)
    .add_local_dir("models/wav2vec2-base-960h", "/app/models/wav2vec2-base-960h", copy=True)
    .add_local_file("idle_generator.py", "/app/idle_generator.py", copy=True)
)

app = modal.App("aivatar-idle-video-generator", image=image)


@app.cls(
    gpu="L4",
    min_containers=0,
    scaledown_window=15,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=2)
class IdleGenerator:
    @modal.enter(snap=True)
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        # Load models to CPU for snapshotting (no CUDA calls before snapshot)
        os.environ["FLASHHEAD_LOAD_DEVICE"] = "cpu"
        import idle_generator
        from flash_head.inference import get_pipeline
        ckpt_dir = os.getenv("FLASHHEAD_CKPT_DIR", "/app/models/SoulX-FlashHead-1_3B")
        wav2vec_dir = os.getenv("WAV2VEC_DIR", "/app/models/wav2vec2-base-960h")
        pipeline = get_pipeline(1, ckpt_dir, "lite", wav2vec_dir)
        idle_generator.set_pipeline(pipeline)
        self._idle_generator = idle_generator
        print("[modal-idle] Pipeline loaded to CPU for snapshot", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        import sys
        sys.path.insert(0, "/app")
        os.environ.pop("FLASHHEAD_LOAD_DEVICE", None)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from flash_head.inference import get_pipeline
        ckpt_dir = os.getenv("FLASHHEAD_CKPT_DIR", "/app/models/SoulX-FlashHead-1_3B")
        wav2vec_dir = os.getenv("WAV2VEC_DIR", "/app/models/wav2vec2-base-960h")
        pipeline = get_pipeline(1, ckpt_dir, "lite", wav2vec_dir)
        pipeline.move_to_device(device)
        self._idle_generator.set_pipeline(pipeline)
        print(f"[modal-idle] Pipeline moved to {device}", flush=True)

    @modal.web_server(8000, startup_timeout=300)
    def serve(self):
        import sys
        sys.path.insert(0, "/app")
        import asyncio
        import threading
        from aiohttp import web

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )
        logger = logging.getLogger("modal_idle_video_generator")

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _start():
                application = self._idle_generator.app
                runner = web.AppRunner(application)
                await runner.setup()
                site = web.TCPSite(runner, "0.0.0.0", 8000)
                await site.start()
                logger.info("idle generator aiohttp site started")
                await asyncio.Event().wait()

            loop.run_until_complete(_start())

        thread = threading.Thread(target=_run, daemon=True, name="modal-idle-aiohttp")
        thread.start()
        logger.info("idle generator aiohttp background thread started ident=%s", thread.ident)
