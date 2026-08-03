"""
Diagnostic and logging utilities for Modal worker (modal_app.py).

All functions are controlled by environment variables, defaulting to enabled:
- AIVATAR_LOG_ATTENTION_BACKEND (default "0"): Log available attention kernels
- AIVATAR_LOG_SIGNAL_HANDLERS (default "1"): Log OS signal handler state
- AIVATAR_LOG_METRICS (default "1"): Background GPU/CPU/RAM metrics thread
- AIVATAR_LOG_DENOISE_STEP (default "0"): Per-step denoise timing (filtered by default)
- METRICS_INTERVAL (default "5"): Interval in seconds for metrics logging
"""

import os
import re
import signal
import faulthandler
import logging


class ShortenLiveKitWebSocketUrlFilter(logging.Filter):
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


def crash_handler(signum, frame):
    """Enhanced crash handler: dumps Python + C backtraces, thread state, UCX info."""
    import threading
    print(f"\n{'='*60}", flush=True)
    print(f"[CRASH] signal={signum} pid={os.getpid()} thread={threading.current_thread().name}", flush=True)
    print(f"[CRASH] Frame: {frame}", flush=True)
    print(f"\n[CRASH] === Python traceback (all threads) ===", flush=True)
    faulthandler.dump_traceback(limit=50)
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


def install_signal_handlers():
    """Install crash_handler for SIGSEGV/SIGABRT/SIGBUS/SIGFPE."""
    faulthandler.enable()
    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        try:
            signal.signal(sig, crash_handler)
        except (OSError, ValueError):
            pass


def reset_ucx_signal_handlers():
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


def log_cuda_state(label):
    """Log CUDA memory allocated/reserved and snapshot segment count."""
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


_ENABLE_ATTENTION_LOG = os.getenv("AIVATAR_LOG_ATTENTION_BACKEND", "0") == "1"


def log_attention_backend(label):
    """Log which attention backend is available for debugging."""
    if not _ENABLE_ATTENTION_LOG:
        return
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


_ENABLE_SIGNAL_LOG = os.getenv("AIVATAR_LOG_SIGNAL_HANDLERS", "1") == "1"


def log_signal_handlers(label):
    """Log current signal handler state for debugging GPU snapshot issues."""
    if not _ENABLE_SIGNAL_LOG:
        return
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


_ENABLE_METRICS_LOG = os.getenv("AIVATAR_LOG_METRICS", "1") == "1"
METRICS_INTERVAL_SECONDS = float(os.getenv("METRICS_INTERVAL", "5"))


def metrics_logger(handler_ref, interval=METRICS_INTERVAL_SECONDS):
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
            gpu_util = "n/a"
            gpu_mem_alloc = 0
            gpu_mem_reserved = 0
            gpu_mem_total = 0
            gpu_temp = "n/a"

            if torch.cuda.is_available():
                gpu_mem_alloc = torch.cuda.memory_allocated()
                gpu_mem_reserved = torch.cuda.memory_reserved()
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory

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

            ru = resource.getrusage(resource.RUSAGE_SELF)
            cpu_user = ru.ru_utime
            cpu_sys = ru.ru_stime
            rss_mb = ru.ru_maxrss / 1024

            active_sessions = len(handler_ref._active_sessions)
            pool_avail = handler_ref.model_pool.get_available_count()
            pool_size = handler_ref.model_pool.max_size

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


def install_denoise_filter():
    """Monkey-patch builtins.print to suppress per-step denoise timing logs.

    flash_head_pipeline.py prints "model denoise per step" timing many times
    per session, which floods logs during concurrent batched inference.
    Set AIVATAR_LOG_DENOISE_STEP=1 to keep denoise step timing visible.

    Returns a no-op if the filter is disabled (AIVATAR_LOG_DENOISE_STEP=1).
    """
    if os.getenv("AIVATAR_LOG_DENOISE_STEP", "0") == "1":
        return

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
