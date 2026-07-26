"""
Phase 1: Bare CUDA GPU Memory Snapshot Test.

Minimal test - no FlashHead, no xfuser, no LiveKit.
Confirms GPU snapshots work at all on L40S with our NGC base image.

Deploy: modal deploy modal_gpu_snapshot_bare_test.py
App name: aivatar-gpu-snapshot-bare-test
"""

import modal
import time
import os
import signal
import faulthandler
import json

app_name = "aivatar-gpu-snapshot-bare-test"

image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.02-py3")

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
            print(f"[CRASH] signal={signum} cuda={cuda_info}", flush=True)
        except Exception:
            print(f"[CRASH] signal={signum} (diagnostics failed)", flush=True)
        faulthandler.dump_traceback(all_threads=True)

    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
        try:
            signal.signal(sig, _crash_handler)
        except (OSError, ValueError):
            pass


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


@app.cls(
    gpu="L40S",
    min_containers=0,
    scaledown_window=15,
    timeout=300,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class BareCudaTest:
    @modal.enter(snap=True)
    def load(self):
        _install_signal_handlers()
        import torch

        self._snap_create_start = time.monotonic()
        print(f"[SNAP_CREATE] start t={self._snap_create_start:.3f}", flush=True)
        _log_cuda_state("SNAP_CREATE_PRE_ALLOC")

        self._test_tensor = torch.randn(2048, 2048, device="cuda")
        result = self._test_tensor @ self._test_tensor.T
        torch.cuda.synchronize()

        self._snap_create_end = time.monotonic()
        print(f"[SNAP_CREATE] matmul done t={self._snap_create_end:.3f} elapsed={((self._snap_create_end - self._snap_create_start) * 1000):.1f}ms", flush=True)
        _log_cuda_state("SNAP_CREATE_POST_MATMUL")
        print("[SNAP_CREATE] GPU tensor allocated and matmul completed - ready for snapshot", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        _install_signal_handlers()
        import torch

        self._restore_start = time.monotonic()
        print(f"[RESTORE] start t={self._restore_start:.3f}", flush=True)
        _log_cuda_state("RESTORE_PRE_CHECK")

        cuda_ok = torch.cuda.is_available()
        print(f"[RESTORE] cuda.is_available={cuda_ok}", flush=True)

        if cuda_ok:
            try:
                result = self._test_tensor @ self._test_tensor.T
                torch.cuda.synchronize()
                self._restore_end = time.monotonic()
                print(f"[RESTORE] matmul survived t={self._restore_end:.3f} elapsed={((self._restore_end - self._restore_start) * 1000):.1f}ms", flush=True)
                _log_cuda_state("RESTORE_POST_MATMUL")
                self._restore_success = True
            except Exception as e:
                self._restore_end = time.monotonic()
                print(f"[RESTORE] matmul FAILED t={self._restore_end:.3f} elapsed={((self._restore_end - self._restore_start) * 1000):.1f}ms error={e}", flush=True)
                self._restore_success = False
                self._restore_error = str(e)
        else:
            self._restore_end = time.monotonic()
            self._restore_success = False
            self._restore_error = "CUDA not available after restore"

    @modal.method()
    def run_test(self):
        import torch
        results = {
            "phase": "post_restore_inference",
            "cuda_available": torch.cuda.is_available(),
            "restore_success": getattr(self, "_restore_success", None),
            "restore_error": getattr(self, "_restore_error", None),
            "restore_elapsed_ms": round((getattr(self, "_restore_end", 0) - getattr(self, "_restore_start", 0)) * 1000, 1),
        }

        if torch.cuda.is_available():
            try:
                t0 = time.monotonic()
                x = torch.randn(4096, 4096, device="cuda")
                y = x @ x.T
                torch.cuda.synchronize()
                elapsed = (time.monotonic() - t0) * 1000
                results["post_restore_matmul_ms"] = round(elapsed, 1)
                results["post_restore_matmul_shape"] = str(y.shape)
                results["status"] = "PASS"
            except Exception as e:
                results["post_restore_matmul_error"] = str(e)
                results["status"] = "FAIL"
        else:
            results["status"] = "FAIL"

        _log_cuda_state("HTTP_TEST")
        print(f"[HTTP_TEST] results={json.dumps(results)}", flush=True)
        return results

    @modal.method()
    def health(self):
        return {"status": "ok", "app": app_name}
