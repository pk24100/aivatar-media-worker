"""
Modal STRESS TEST wrapper for the AiVatar media worker.

This is a copy of modal_app.py with high concurrency settings to find the
maximum concurrent sessions a single GPU can handle before FPS degrades.

Changes from production (modal_app.py):
- App name: aivatar-worker-stress (separate from production)
- WORKER_CONCURRENCY default: 15 (vs 3 in production)
- GPU: Set STRESS_TARGET_GPU env var or edit TARGET_GPU below (H100, RTX-PRO-6000, L40S, A100-40GB, A10)
- SageAttention: Built per-GPU CUDA arch from GPU_CONFIG dict
- Optimizations: SageAttention 2.2.0 (primary attention kernel)
- torch.compile DISABLED (causes FX symbolic tracing crashes with FlashHeadPipeline)
- STRESS_ENABLE_COMPILE env var controls torch.compile (default: "0" = disabled)
- min_containers: 1 (warm container for test start)
- Prewarm pool: 1 room only (test script mints rooms for sessions 2-N)
- /readyz: prewarm check relaxed (stress test mints rooms, prewarm not required)
- CPU memory snapshots: ENABLED (load() runs on CPU, restore() moves to GPU + warmup)
- GPU memory snapshots: DISABLED (corrupts NVENC hardware state on Blackwell sm120)
- Snapshot flow: load() -> CPU pipeline + caches -> snapshot -> restore() -> move_to_device(cuda) + warmup

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
WORKER_CONCURRENCY = int(os.getenv("AIVATAR_WORKER_CONCURRENCY", "15"))
if WORKER_CONCURRENCY < 1:
    raise ValueError("AIVATAR_WORKER_CONCURRENCY must be at least 1")

# ---------------------------------------------------------------------------
# GPU CONFIG - Change TARGET_GPU to switch GPUs. All GPU-specific settings
# (Modal gpu string, SageAttention CUDA arch, video codec, NVENC) are driven
# from this dict. Add new GPUs here as needed.
# ---------------------------------------------------------------------------
TARGET_GPU = os.getenv("STRESS_TARGET_GPU", "B300")

GPU_CONFIG = {
    "H100": {
        "modal_gpu": "H100",
        "cuda_arch": "9.0",
        "video_codec": "h264_hw",
        "nvenc_ok": True,
        "notes": "Hopper sm90. NVENC 7th-gen works with LiveKit.",
    },
    "RTX-PRO-6000": {
        "modal_gpu": "RTX-PRO-6000",
        "cuda_arch": "12.0",
        "video_codec": "vp8",
        "nvenc_ok": False,
        "notes": "Blackwell sm120. NVENC 9th-gen incompatible with LiveKit, use VP8 software.",
    },
    "L40S": {
        "modal_gpu": "L40S",
        "cuda_arch": "8.9",
        "video_codec": "h264_hw",
        "nvenc_ok": True,
        "notes": "Ada sm89. NVENC works with LiveKit.",
    },
    "A100-40GB": {
        "modal_gpu": "A100-40GB",
        "cuda_arch": "8.0",
        "video_codec": "h264_hw",
        "nvenc_ok": True,
        "notes": "Ampere sm80. NVENC works with LiveKit.",
    },
    "A10": {
        "modal_gpu": "A10",
        "cuda_arch": "8.6",
        "video_codec": "h264_hw",
        "nvenc_ok": True,
        "notes": "Ampere sm86. NVENC works with LiveKit.",
    },
    "B200": {
        "modal_gpu": "B200",
        "cuda_arch": "10.0",
        "video_codec": "vp8",
        "nvenc_ok": False,
        "notes": "Blackwell GB100 sm100. Same arch family as RTX PRO 6000 (sm120). Use VP8 software - NVENC compatibility unverified with LiveKit.",
    },
    "H200": {
        "modal_gpu": "H200",
        "cuda_arch": "9.0",
        "video_codec": "h264_hw",
        "nvenc_ok": True,
        "notes": "Hopper sm90. Same compute as H100, more VRAM (141GB). NVENC 7th-gen works with LiveKit.",
    },
    "B300": {
        "modal_gpu": "B300",
        "cuda_arch": "10.3",
        "video_codec": "vp8",
        "nvenc_ok": False,
        "notes": "Blackwell Ultra GB300 sm103. Build natively with sm103a (compute_100a PTX is NOT forward-compatible to sm103 due to 'a' suffix per NVIDIA Blackwell Compatibility Guide). Most SMs (~240). Use VP8 software.",
    },
}

if TARGET_GPU not in GPU_CONFIG:
    raise ValueError(
        f"Unknown TARGET_GPU='{TARGET_GPU}'. Valid options: {list(GPU_CONFIG.keys())}"
    )

_gpu_cfg = GPU_CONFIG[TARGET_GPU]
print(f"[STRESS] Target GPU: {TARGET_GPU} (arch={_gpu_cfg['cuda_arch']}, codec={_gpu_cfg['video_codec']})", flush=True)
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


image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.05-py3")
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
    .run_commands(
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
        f"|| echo 'SAGEATTN_INSTALL_FAILED'",
    )
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "huggingface-cli download pkam24100/aivatar-flashhead-model --local-dir /app/models/SoulX-FlashHead-1_3B",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_dir("streaming", "/app/streaming", copy=True)
    .add_local_dir("utils", "/app/utils", copy=True)
    .add_local_dir("config", "/app/config", copy=True)
    .add_local_dir("SoulX-FlashHead", "/app/SoulX-FlashHead", copy=True)
    .add_local_file("SoulX-FlashHead/flash_head/src/modules/flash_head_model_snapshot_patch.py", "/app/SoulX-FlashHead/flash_head/src/modules/flash_head_model.py", copy=True)
    .add_local_dir("models/wav2vec2-base-960h", "/app/models/wav2vec2-base-960h", copy=True)
    .add_local_file("handler.py", "/app/handler.py", copy=True)
    .add_local_file("app_factory.py", "/app/app_factory.py", copy=True)
)

app = modal.App(WORKER_APP_NAME, image=image)


worker_cls_config = {
    "gpu": _gpu_cfg["modal_gpu"],
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
}


@app.cls(**worker_cls_config)
@modal.concurrent(max_inputs=WORKER_CONCURRENCY, target_inputs=WORKER_CONCURRENCY)
class Worker:
    @modal.enter(snap=True)
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/SoulX-FlashHead")

        # Set env vars BEFORE any import that reads them.
        # FLASHHEAD_LOAD_DEVICE=cpu makes get_device() return "cpu" so
        # the pipeline loads to CPU with no CUDA calls during snapshot.
        os.environ["AIVATAR_WORKER_CONCURRENCY"] = str(WORKER_CONCURRENCY)
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["FLASHHEAD_LOAD_DEVICE"] = "cpu"
        os.environ.setdefault("LIVEKIT_RTC_DEBUG", "false")
        os.environ["ENGINE_PROFILE"] = "1"
        os.environ["AIVATAR_BATCHED_INFERENCE"] = "1"

        print(f"[STRESS] CPU snapshot load with WORKER_CONCURRENCY={WORKER_CONCURRENCY}", flush=True)

        # Import ONLY flash_head (torch + model) - NOT handler/livekit.
        # handler.py imports streaming.stream_processor which imports livekit
        # (Rust FFI). LiveKit spawns background threads that get captured in
        # the CRIU snapshot and corrupt CUDA state after restore.
        # handler.py is deferred to serve() (post-restore).
        from flash_head.inference import get_pipeline

        # Log attention backend availability (CPU mode - no GPU calls)
        _log_attention_backend("SNAP_CREATE")

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
        _log_signal_handlers("SNAP_PRE_RESET")
        _reset_ucx_signal_handlers()
        _log_signal_handlers("SNAP_POST_RESET")

        print("[STRESS] Pipeline loaded to CPU for snapshot", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        import sys
        sys.path.insert(0, "/app")

        # Pop FLASHHEAD_LOAD_DEVICE so future get_pipeline() calls use GPU
        os.environ.pop("FLASHHEAD_LOAD_DEVICE", None)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[RESTORE] CPU snapshot restore - moving pipeline to {device}", flush=True)
        _log_cuda_state("RESTORE_START")

        # Move the snapshot pipeline from CPU to GPU
        assert self._snap_pipeline is not None, "Pipeline lost in snapshot!"
        move_t0 = time.monotonic()
        self._snap_pipeline.move_to_device(device)
        if device == "cuda":
            torch.cuda.synchronize()
        move_ms = (time.monotonic() - move_t0) * 1000
        print(f"[RESTORE] Pipeline moved to {device} in {move_ms:.1f}ms", flush=True)

        _log_cuda_state("RESTORE_POST_MOVE")

        # Warmup on GPU
        print("[RESTORE] Running warmup inference on GPU...", flush=True)
        warm_t0 = time.monotonic()
        self._warm_pipeline(self._snap_pipeline)
        warm_ms = (time.monotonic() - warm_t0) * 1000
        print(f"[RESTORE] Warmup completed in {warm_ms:.1f}ms", flush=True)

        _log_cuda_state("RESTORE_POST_WARMUP")
        _log_attention_backend("RESTORE_POST")
        print("[RESTORE] CPU snapshot restore complete", flush=True)

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
        """Fill configured capacity with the same shared pipeline object.

        handler.model_pool is created in serve() with INITIAL_PIPELINE_COUNT=WORKER_POOL_SIZE
        (FLASHHEAD_LOAD_DEVICE is popped in restore). The monkey-patched get_pipeline()
        returns self._snap_pipeline for every call, so the pool already has WORKER_POOL_SIZE
        references to the same pipeline. This method is a no-op but kept for compatibility.
        """
        pool = self._handler.model_pool
        print(
            f"[STRESS] Pool status: current_size={pool.current_size} "
            f"available={pool.get_available_count()}/{pool.max_size}",
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

        # Env vars already set in load() (survive CPU snapshot).
        # Re-assert AIVATAR_BATCHED_INFERENCE in case restore() cleared it.
        os.environ["AIVATAR_BATCHED_INFERENCE"] = "1"

        # Video codec: driven by GPU_CONFIG. GPUs with incompatible NVENC (e.g. RTX PRO 6000
        # Blackwell 9th-gen) use VP8 software encoding. Others use H264 hardware encoding.
        os.environ.setdefault("AIVATAR_VIDEO_CODEC", _gpu_cfg["video_codec"])

        # Stress test: relax WebSocket rate limits and IP connection throttling.
        # Test script sends audio at 5x real-time (send_interval=0.02) and all
        # sessions originate from the same IP (test machine or Modal proxy).
        # These must be set before `import handler` so the module-level constants
        # pick up the overridden values.
        os.environ.setdefault("WS_AUDIO_RATE_PER_SEC", "600")
        os.environ.setdefault("WS_AUDIO_BURST", "2000")
        os.environ.setdefault("WS_MAX_BYTES_PER_SEC", "1920000")
        os.environ.setdefault("WS_MAX_CONNECTIONS_PER_IP", "100")

        print(f"[SERVE] Starting serve() post-restore (STRESS TEST, concurrency={WORKER_CONCURRENCY})", flush=True)

        _install_signal_handlers()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )
        for h in logging.getLogger().handlers:
            h.addFilter(_ShortenLiveKitWebSocketUrlFilter())
        logger = logging.getLogger("modal_app_stress")

        # Monkey-patch get_pipeline to return our snapshot pipeline.
        # handler.py creates model_pool at import time which calls get_pipeline()
        # INITIAL_PIPELINE_COUNT times. Since FLASHHEAD_LOAD_DEVICE is popped
        # in restore(), INITIAL_PIPELINE_COUNT=WORKER_POOL_SIZE, so the pool
        # fills with WORKER_POOL_SIZE references to self._snap_pipeline.
        import flash_head.inference as fhi
        _orig_get_pipeline = fhi.get_pipeline
        fhi.get_pipeline = lambda *a, **kw: self._snap_pipeline

        # Now import handler (post-restore) - creates model_pool with shared pipeline.
        # handler imports livekit (Rust FFI) which is unsafe during CRIU snapshot
        # but safe after restore.
        import handler
        self._handler = handler

        # Restore get_pipeline for any future calls
        fhi.get_pipeline = _orig_get_pipeline

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

        # Initialize BatchedStreamingEngine with the snapshot pipeline.
        # All concurrent sessions share this single engine for batched inference.
        from streaming.batched_engine import BatchedStreamingEngine
        handler.batched_engine = BatchedStreamingEngine(self._snap_pipeline)
        handler.batched_engine.start()
        print(f"[STRESS] BatchedStreamingEngine started (wait_window={handler.batched_engine.wait_window_ms}ms)", flush=True)

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

        print(
            f"[SERVE] Pipeline in model_pool: "
            f"current_size={handler.model_pool.current_size} "
            f"available={handler.model_pool.get_available_count()}",
            flush=True,
        )

        # Import build_app (handler already imported above in serve())
        from app_factory import build_app

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

                # Batched warmup: trigger CUDA kernel compilation for run_pipeline_batch
                # in a background thread so the event loop stays free and the HTTP
                # server can accept /readyz checks immediately. The first batched call
                # JIT-compiles kernels (~2s on Blackwell); running it here absorbs
                # that cost so session 1 doesn't see a warmup spike.
                def _run_batched_warmup():
                    import flash_head.inference as _fhi_warm
                    import numpy as _np_warm
                    from PIL import Image as _PILImage
                    print("[STRESS] Running batched warmup cycle (background)...", flush=True)
                    try:
                        _warm_t0 = time.monotonic()
                        _warm_params = _fhi_warm.get_infer_params()
                        _warm_sid = "_warmup_"
                        _warm_img = "/tmp/warmup_avatar.png"
                        if not os.path.exists(_warm_img):
                            _PILImage.new("RGB", (512, 512), color=(128, 128, 128)).save(_warm_img)
                        handler.batched_engine.add_session(_warm_sid, _warm_img, 42, _warm_params)
                        _warm_sr = _warm_params["sample_rate"]
                        _warm_slice = (_warm_params["frame_num"] - _warm_params["motion_frames_num"]) * _warm_sr // _warm_params["tgt_fps"]
                        handler.batched_engine.feed_audio(_warm_sid, _np_warm.zeros(_warm_slice, dtype=_np_warm.float32))
                        for _ in range(100):
                            _wsess = handler.batched_engine.get_session(_warm_sid)
                            if _wsess is None or _wsess.slices_processed > 0:
                                break
                            time.sleep(0.1)
                        handler.batched_engine.remove_session(_warm_sid)
                        for _ in range(50):
                            if handler.batched_engine.get_session(_warm_sid) is None:
                                break
                            time.sleep(0.1)
                        _warm_ms = (time.monotonic() - _warm_t0) * 1000
                        print(f"[STRESS] Batched warmup completed in {_warm_ms:.1f}ms", flush=True)
                    except Exception as _warm_err:
                        print(f"[STRESS] Batched warmup failed (non-fatal): {_warm_err}", flush=True)

                asyncio.create_task(asyncio.to_thread(_run_batched_warmup))

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
