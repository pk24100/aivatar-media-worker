"""Warmup helpers shared by modal_app.py and modal_app_stress.py.

warm_pipeline(): single-session GPU warmup after CPU->GPU restore.
run_batched_warmup(): background-thread warmup that JIT-compiles the batched
inference kernels so the first real session does not see a warmup spike.
"""

import contextlib
import os
import time


def warm_pipeline(pipeline):
    """Run one dummy inference to warm the freshly restored GPU pipeline."""
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


def run_batched_warmup(batched_engine, stop_event=None, log_prefix="[SERVE]"):
    """Trigger CUDA kernel compilation for run_pipeline_batch.

    Runs in a background thread so the event loop stays free and the HTTP
    server can accept /readyz checks immediately. The first batched call
    JIT-compiles kernels (~2s); running it here absorbs that cost so session 1
    does not see a warmup spike.

    stop_event: optional threading.Event; when set, the warmup loop exits
        early (used by production shutdown).
    """
    import flash_head.inference as fhi
    import numpy as np
    from PIL import Image

    session_id = "_warmup_"
    added = False
    print(f"{log_prefix} Running batched warmup cycle (background)...", flush=True)
    try:
        t0 = time.monotonic()
        params = fhi.get_infer_params()
        img_path = "/tmp/warmup_avatar.png"
        if not os.path.exists(img_path):
            Image.new("RGB", (512, 512), color=(128, 128, 128)).save(img_path)
        try:
            batched_engine.add_session(session_id, img_path, 42, params)
        except ValueError:
            # Already warming/warmed; reuse existing placeholder.
            pass
        added = True
        sr = params["sample_rate"]
        slice_samples = (
            (params["frame_num"] - params["motion_frames_num"]) * sr // params["tgt_fps"]
        )
        batched_engine.feed_audio(session_id, np.zeros(slice_samples, dtype=np.float32))
        for _ in range(100):
            if stop_event is not None and stop_event.is_set():
                break
            sess = batched_engine.get_session(session_id)
            if sess is None or sess.slices_processed > 0:
                break
            time.sleep(0.1)
        elapsed_ms = (time.monotonic() - t0) * 1000
        if stop_event is not None and stop_event.is_set():
            print(f"{log_prefix} Batched warmup stopped during shutdown", flush=True)
        else:
            print(f"{log_prefix} Batched warmup completed in {elapsed_ms:.1f}ms", flush=True)
    except Exception as err:
        print(f"{log_prefix} Batched warmup failed (non-fatal): {err}", flush=True)
    finally:
        if added:
            with contextlib.suppress(Exception):
                batched_engine.remove_session(session_id)
            for _ in range(50):
                if batched_engine.get_session(session_id) is None:
                    break
                time.sleep(0.1)
