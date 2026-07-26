"""
Phase 2: FlashHead + xfuser GPU Memory Snapshot Isolation Test.

Tests whether the xfuser/UCX import chain survives GPU snapshot restore.
This is the most likely failure point based on Jul 21 SIGSEGV analysis.

Deploy: modal deploy modal_gpu_snapshot_flashhead_test.py
App name: aivatar-gpu-snapshot-flashhead-test

Image-level env vars disable UCX network transports and NCCL P2P/IB
before any Python library loads, since the Jul 21 crash happened in
libucs.so.0 before application code ran.
"""

import modal
import time
import os
import signal
import faulthandler
import json
import subprocess

app_name = "aivatar-gpu-snapshot-flashhead-test"

WORKER_CONCURRENCY = 1

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")
    .env({
        "AIVATAR_WORKER_CONCURRENCY": str(WORKER_CONCURRENCY),
        # UCX: restrict to shared memory + self transport only, no network.
        # Must be set at image level so libucs.so.0 sees them before Python runs.
        "UCX_TLS": "self,sm",
        "UCX_NET_DEVICES": "none",
        "UCX_UNIFIED_TLS_MODE": "y",
        # NCCL: disable P2P and InfiniBand (single GPU doesn't need them).
        "NCCL_DEBUG": "INFO",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        # torch.compile compatibility with GPU snapshots.
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

app = modal.App(app_name, image=image)


def _install_signal_handlers():
    """Install signal handlers that dump diagnostics before the process dies."""
    faulthandler.enable(all_threads=True)

    def _crash_handler(signum, frame):
        try:
            import torch
            cuda_info = "unavailable"
            try:
                if torch.cuda.is_available():
                    cuda_info = json.dumps({
                        "device": torch.cuda.get_device_name(0),
                        "allocated": torch.cuda.memory_allocated(),
                        "reserved": torch.cuda.memory_reserved(),
                    })
            except Exception:
                cuda_info = "cuda query failed"

            dist_info = "n/a"
            try:
                import torch.distributed as dist
                dist_info = str(dist.is_initialized())
            except Exception:
                dist_info = "dist query failed"

            print(f"[CRASH] signal={signum} cuda={cuda_info} dist_initialized={dist_info}", flush=True)
        except Exception:
            print(f"[CRASH] signal={signum} (diagnostics failed)", flush=True)
        faulthandler.dump_traceback(all_threads=True)

    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        try:
            signal.signal(sig, _crash_handler)
        except (OSError, ValueError):
            pass


def _log_ucx_state(label):
    """Log UCX library state."""
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True, text=True, timeout=5,
        )
        ucs_lines = [l for l in result.stdout.splitlines() if "ucs" in l.lower()]
        print(f"[{label}] ucx_libraries={ucs_lines}", flush=True)
    except Exception as e:
        print(f"[{label}] ucx_libraries_query_failed={e}", flush=True)


def _log_env(label):
    """Dump relevant env vars to verify image-level settings are present."""
    relevant = [
        "UCX_TLS", "UCX_NET_DEVICES", "UCX_UNIFIED_TLS_MODE",
        "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
        "TORCHINDUCTOR_COMPILE_THREADS", "FLASHHEAD_LOAD_DEVICE",
    ]
    env_dump = {k: os.environ.get(k, "<not set>") for k in relevant}
    print(f"[{label}] env={json.dumps(env_dump)}", flush=True)


def _log_cuda_state(label):
    """Log CUDA state at a given phase boundary."""
    import torch
    print(f"[{label}] pid={os.getpid()}", flush=True)
    print(f"[{label}] torch.cuda.is_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[{label}] device={props.name} total_memory={props.total_memory}", flush=True)
        print(f"[{label}] allocated={torch.cuda.memory_allocated()} reserved={torch.cuda.memory_reserved()}", flush=True)
        try:
            snapshot = torch.cuda.memory_snapshot()
            print(f"[{label}] memory_snapshot_segments={len(snapshot)}", flush=True)
        except Exception as e:
            print(f"[{label}] memory_snapshot_failed={e}", flush=True)


def _log_dist_state(label):
    """Log torch.distributed initialization state."""
    try:
        import torch.distributed as dist
        print(f"[{label}] dist.is_initialized={dist.is_initialized()}", flush=True)
        if dist.is_initialized():
            print(f"[{label}] dist.backend={dist.get_backend()}", flush=True)
    except Exception as e:
        print(f"[{label}] dist_query_failed={e}", flush=True)


def _try_import_xfuser(label):
    """Attempt xfuser import with full error logging."""
    print(f"[{label}] Attempting xfuser import...", flush=True)
    try:
        import xfuser
        print(f"[{label}] xfuser imported successfully, version={getattr(xfuser, '__version__', 'unknown')}", flush=True)
    except Exception as e:
        print(f"[{label}] xfuser import FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False
    return True


@app.cls(
    gpu="L40S",
    min_containers=0,
    scaledown_window=15,
    timeout=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=1)
class FlashHeadSnapshotTest:
    @modal.enter(snap=True)
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/SoulX-FlashHead")

        _install_signal_handlers()
        _log_env("SNAP_CREATE")
        _log_ucx_state("SNAP_CREATE")
        _log_cuda_state("SNAP_CREATE_PRE_IMPORT")

        self._snap_create_start = time.monotonic()
        print(f"[SNAP_CREATE] start t={self._snap_create_start:.3f}", flush=True)

        # Step 1: Try xfuser import (the suspected SIGSEGV trigger)
        xfuser_ok = _try_import_xfuser("SNAP_CREATE")
        _log_cuda_state("SNAP_CREATE_POST_XFUSER")
        _log_dist_state("SNAP_CREATE_POST_XFUSER")

        # Step 2: Import flash_head inference
        print("[SNAP_CREATE] Importing flash_head.inference...", flush=True)
        try:
            from flash_head.inference import get_pipeline, get_base_data, get_audio_embedding, run_pipeline, get_infer_params
            print("[SNAP_CREATE] flash_head.inference imported successfully", flush=True)
        except Exception as e:
            print(f"[SNAP_CREATE] flash_head.inference import FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        _log_cuda_state("SNAP_CREATE_POST_FLASHHEAD_IMPORT")

        # Step 3: Load pipeline on GPU (no FLASHHEAD_LOAD_DEVICE=cpu)
        print("[SNAP_CREATE] Loading FlashHead pipeline on GPU...", flush=True)
        ckpt_dir = os.getenv("FLASHHEAD_CKPT_DIR", "/app/models/SoulX-FlashHead-1_3B")
        wav2vec_dir = os.getenv("WAV2VEC_DIR", "/app/models/wav2vec2-base-960h")
        model_type = os.getenv("FLASHHEAD_MODEL_TYPE", "lite")

        load_start = time.monotonic()
        try:
            self._pipeline = get_pipeline(
                world_size=1,
                ckpt_dir=ckpt_dir,
                model_type=model_type,
                wav2vec_dir=wav2vec_dir,
            )
            load_elapsed = (time.monotonic() - load_start) * 1000
            print(f"[SNAP_CREATE] Pipeline loaded in {load_elapsed:.1f}ms", flush=True)
        except Exception as e:
            print(f"[SNAP_CREATE] Pipeline load FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        _log_cuda_state("SNAP_CREATE_POST_PIPELINE_LOAD")

        # Step 4: Warmup inference (dummy data)
        print("[SNAP_CREATE] Running warmup inference...", flush=True)
        from PIL import Image
        import numpy as np

        warm_start = time.monotonic()
        try:
            dummy_img_path = "/tmp/warmup_avatar.png"
            Image.new("RGB", (512, 512), color=(128, 128, 128)).save(dummy_img_path)
            get_base_data(self._pipeline, dummy_img_path, base_seed=42, use_face_crop=False)
            params = get_infer_params()
            sr = params["sample_rate"]
            cached_dur = params["cached_audio_duration"]
            frame_num = params["frame_num"]
            tgt_fps = params["tgt_fps"]
            audio_end_idx = cached_dur * tgt_fps
            audio_start_idx = audio_end_idx - frame_num
            dummy_audio = np.zeros(cached_dur * sr, dtype=np.float32)
            audio_emb = get_audio_embedding(self._pipeline, dummy_audio, audio_start_idx, audio_end_idx)
            run_pipeline(self._pipeline, audio_emb)

            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            warm_elapsed = (time.monotonic() - warm_start) * 1000
            print(f"[SNAP_CREATE] Warmup inference completed in {warm_elapsed:.1f}ms", flush=True)
        except Exception as e:
            print(f"[SNAP_CREATE] Warmup inference FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        self._snap_create_end = time.monotonic()
        total_ms = (self._snap_create_end - self._snap_create_start) * 1000
        print(f"[SNAP_CREATE] Complete total={total_ms:.1f}ms", flush=True)
        _log_cuda_state("SNAP_CREATE_FINAL")
        print("[SNAP_CREATE] Ready for GPU snapshot", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        import sys
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/SoulX-FlashHead")

        _install_signal_handlers()
        import torch

        self._restore_start = time.monotonic()
        print(f"[RESTORE] start t={self._restore_start:.3f}", flush=True)
        _log_env("RESTORE")
        _log_ucx_state("RESTORE")
        _log_cuda_state("RESTORE_PRE_CHECK")
        _log_dist_state("RESTORE_PRE_CHECK")

        cuda_ok = torch.cuda.is_available()
        print(f"[RESTORE] cuda.is_available={cuda_ok}", flush=True)

        if not cuda_ok:
            self._restore_success = False
            self._restore_error = "CUDA not available after restore"
            self._restore_end = time.monotonic()
            print(f"[RESTORE] FAILED: {self._restore_error}", flush=True)
            return

        # Verify the pipeline tensor survived
        try:
            assert hasattr(self, "_pipeline"), "Pipeline object missing after restore"
            print(f"[RESTORE] Pipeline object present: {type(self._pipeline).__name__}", flush=True)
        except AssertionError as e:
            self._restore_success = False
            self._restore_error = str(e)
            self._restore_end = time.monotonic()
            print(f"[RESTORE] FAILED: {e}", flush=True)
            return

        # Run a post-restore inference to verify CUDA kernels work
        print("[RESTORE] Running post-restore inference...", flush=True)
        from PIL import Image
        import numpy as np

        try:
            from flash_head.inference import get_base_data, get_audio_embedding, run_pipeline, get_infer_params

            dummy_img_path = "/tmp/restore_warmup_avatar.png"
            Image.new("RGB", (512, 512), color=(128, 128, 128)).save(dummy_img_path)
            get_base_data(self._pipeline, dummy_img_path, base_seed=42, use_face_crop=False)
            params = get_infer_params()
            sr = params["sample_rate"]
            cached_dur = params["cached_audio_duration"]
            frame_num = params["frame_num"]
            tgt_fps = params["tgt_fps"]
            audio_end_idx = cached_dur * tgt_fps
            audio_start_idx = audio_end_idx - frame_num
            dummy_audio = np.zeros(cached_dur * sr, dtype=np.float32)
            audio_emb = get_audio_embedding(self._pipeline, dummy_audio, audio_start_idx, audio_end_idx)
            run_pipeline(self._pipeline, audio_emb)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self._restore_end = time.monotonic()
            elapsed = (self._restore_end - self._restore_start) * 1000
            print(f"[RESTORE] Post-restore inference succeeded in {elapsed:.1f}ms", flush=True)
            _log_cuda_state("RESTORE_POST_INFERENCE")
            self._restore_success = True
            self._restore_error = None
        except Exception as e:
            self._restore_end = time.monotonic()
            elapsed = (self._restore_end - self._restore_start) * 1000
            print(f"[RESTORE] Post-restore inference FAILED in {elapsed:.1f}ms: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            _log_cuda_state("RESTORE_POST_FAILURE")
            self._restore_success = False
            self._restore_error = f"{type(e).__name__}: {e}"

    @modal.method()
    def run_test(self):
        import torch
        results = {
            "phase": "flashhead_xfuser_gpu_snapshot",
            "cuda_available": torch.cuda.is_available(),
            "restore_success": getattr(self, "_restore_success", None),
            "restore_error": getattr(self, "_restore_error", None),
            "restore_elapsed_ms": round((getattr(self, "_restore_end", 0) - getattr(self, "_restore_start", 0)) * 1000, 1),
            "snap_create_elapsed_ms": round((getattr(self, "_snap_create_end", 0) - getattr(self, "_snap_create_start", 0)) * 1000, 1),
        }

        if torch.cuda.is_available() and getattr(self, "_restore_success", False):
            try:
                t0 = time.monotonic()
                from PIL import Image
                import numpy as np
                from flash_head.inference import get_base_data, get_audio_embedding, run_pipeline, get_infer_params

                dummy_img_path = "/tmp/http_test_avatar.png"
                Image.new("RGB", (512, 512), color=(128, 128, 128)).save(dummy_img_path)
                get_base_data(self._pipeline, dummy_img_path, base_seed=99, use_face_crop=False)
                params = get_infer_params()
                sr = params["sample_rate"]
                cached_dur = params["cached_audio_duration"]
                frame_num = params["frame_num"]
                tgt_fps = params["tgt_fps"]
                audio_end_idx = cached_dur * tgt_fps
                audio_start_idx = audio_end_idx - frame_num
                dummy_audio = np.zeros(cached_dur * sr, dtype=np.float32)
                audio_emb = get_audio_embedding(self._pipeline, dummy_audio, audio_start_idx, audio_end_idx)
                run_pipeline(self._pipeline, audio_emb)
                torch.cuda.synchronize()
                elapsed = (time.monotonic() - t0) * 1000
                results["http_inference_ms"] = round(elapsed, 1)
                results["status"] = "PASS"
            except Exception as e:
                results["http_inference_error"] = str(e)
                results["status"] = "FAIL"
        else:
            results["status"] = "FAIL"

        _log_cuda_state("HTTP_TEST")
        _log_dist_state("HTTP_TEST")
        print(f"[HTTP_TEST] results={json.dumps(results)}", flush=True)
        return results

    @modal.method()
    def health(self):
        return {"status": "ok", "app": app_name}
