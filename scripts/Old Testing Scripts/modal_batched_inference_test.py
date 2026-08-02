"""
Modal test script for batched inference benchmarking on L40S GPU.

Tests batched inference (generate_batch) vs sequential inference (run_pipeline)
to determine if L40S's 142 SMs provide sub-linear scaling for batch > 1.

Usage:
    modal run modal_batched_inference_test.py

The script:
1. Loads 1 FlashHead pipeline on L40S
2. Prepares N avatars with different seeds
3. Runs sequential baseline: N x run_pipeline() (batch=1)
4. Runs batched: 1 x run_pipeline_batch() (batch=N)
5. Logs DENOISE_STEPS and GPU_BREAKDOWN for both
6. Verifies output shapes match
"""

import os
import time
import modal

BATCH_SIZES_TO_TEST = [1, 2, 3]
NUM_SEQUENTIAL_RUNS = 3
NUM_BATCHED_RUNS = 3

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.05-py3")
    .env({
        "UCX_TLS": "self",
        "UCX_NET_DEVICES": "none",
        "UCX_UNIFIED_TLS_MODE": "y",
        "UCX_MEMTYPE_CACHE": "n",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
    .apt_install("git", "git-lfs", "ffmpeg", "libsndfile1", "wget", "ca-certificates")
    .pip_install("ninja")
    .run_commands(
        "TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 NVCC_THREADS=4 pip install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation || echo 'SAGEATTN_INSTALL_FAILED'",
    )
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "huggingface-cli download pkam24100/aivatar-flashhead-model --local-dir /app/models/SoulX-FlashHead-1_3B",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_dir("SoulX-FlashHead", "/app/SoulX-FlashHead", copy=True)
    .add_local_file(
        "flash_head_model_snapshot_patch.py",
        "/app/SoulX-FlashHead/flash_head/src/modules/flash_head_model.py",
        copy=True,
    )
    .add_local_dir("models/wav2vec2-base-960h", "/app/models/wav2vec2-base-960h", copy=True)
)

app = modal.App("aivatar-batched-inference-test", image=image)


@app.function(gpu="L40S", timeout=1800, secrets=[modal.Secret.from_name("huggingface-secret")])
def run_batched_inference_test():
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/SoulX-FlashHead")

    import torch
    import numpy as np
    from PIL import Image

    os.environ["ENGINE_PROFILE"] = "1"

    from flash_head.inference import (
        get_pipeline, get_base_data, get_audio_embedding,
        run_pipeline, run_pipeline_batch, get_infer_params,
    )

    ckpt_dir = "/app/models/SoulX-FlashHead-1_3B"
    wav2vec_dir = "/app/models/wav2vec2-base-960h"
    model_type = "lite"

    print("=" * 80, flush=True)
    print("[TEST] Batched Inference Benchmark - L40S", flush=True)
    print("=" * 80, flush=True)

    # Log attention backend
    try:
        from sageattention import sageattn
        print("[TEST] Attention backend: SageAttention", flush=True)
    except ImportError:
        pass
    try:
        import flash_attn
        print(f"[TEST] Attention backend: flash_attn {flash_attn.__version__}", flush=True)
    except ImportError:
        pass

    # Load pipeline
    print("[TEST] Loading pipeline...", flush=True)
    load_t0 = time.monotonic()
    pipeline = get_pipeline(1, ckpt_dir, model_type, wav2vec_dir)
    load_ms = (time.monotonic() - load_t0) * 1000
    print(f"[TEST] Pipeline loaded in {load_ms:.1f}ms", flush=True)

    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"[TEST] GPU: {gpu_name} ({gpu_props.multi_processor_count} SMs)", flush=True)

    # Prepare N avatars with different colors (simulating different sessions)
    max_batch = max(BATCH_SIZES_TO_TEST)
    avatar_colors = [
        (128, 128, 128),
        (200, 100, 50),
        (50, 150, 200),
        (180, 50, 180),
    ]

    print(f"\n[TEST] Preparing {max_batch} avatar sessions...", flush=True)
    avatar_paths = []
    for i in range(max_batch):
        img_path = f"/tmp/avatar_{i}.png"
        color = avatar_colors[i % len(avatar_colors)]
        Image.new("RGB", (512, 512), color=color).save(img_path)
        avatar_paths.append(img_path)

    # Prepare per-session state
    params = get_infer_params()
    sr = params["sample_rate"]
    cached_dur = params["cached_audio_duration"]
    frame_num = params["frame_num"]
    tgt_fps = params["tgt_fps"]
    audio_end_idx = cached_dur * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    # Generate different audio for each session (different frequencies)
    audio_embeddings = []
    for i in range(max_batch):
        t = np.linspace(0, cached_dur, cached_dur * sr, dtype=np.float32)
        freq = 200 + i * 100  # 200Hz, 300Hz, 400Hz...
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        emb = get_audio_embedding(pipeline, audio, audio_start_idx, audio_end_idx)
        audio_embeddings.append(emb)

    # Prepare per-session pipeline state by calling prepare_params + reset_person_name
    # for each avatar, then extracting the state
    ref_img_latent_list = []
    latent_motion_frames_list = []
    generators = []
    original_color_refs = []
    color_correction_strengths = []

    for i in range(max_batch):
        get_base_data(pipeline, avatar_paths[i], base_seed=42 + i, use_face_crop=False)
        ref_img_latent_list.append(pipeline.ref_img_latent.clone())
        latent_motion_frames_list.append(pipeline.latent_motion_frames.clone())
        generators.append(torch.Generator(device=pipeline.device).manual_seed(42 + i))
        original_color_refs.append(pipeline.original_color_reference.clone())
        color_correction_strengths.append(pipeline.color_correction_strength)

    print(f"[TEST] Per-session state prepared for {max_batch} sessions", flush=True)
    print(f"[TEST] ref_img_latent shape: {ref_img_latent_list[0].shape}", flush=True)
    print(f"[TEST] latent_motion_frames shape: {latent_motion_frames_list[0].shape}", flush=True)
    print(f"[TEST] audio_embedding shape: {audio_embeddings[0].shape}", flush=True)

    # Warmup
    print("\n[TEST] Warming up pipeline (1 sequential run)...", flush=True)
    _ = run_pipeline(pipeline, audio_embeddings[0])
    torch.cuda.synchronize()
    print("[TEST] Warmup complete", flush=True)

    results = {}

    # ==========================================
    # Phase 1: Sequential baseline (batch=1, N times)
    # ==========================================
    print("\n" + "=" * 80, flush=True)
    print("[TEST] Phase 1: Sequential baseline (batch=1, N sequential runs)", flush=True)
    print("=" * 80, flush=True)

    for batch_size in BATCH_SIZES_TO_TEST:
        if batch_size == 1 and NUM_SEQUENTIAL_RUNS == 0:
            continue

        print(f"\n[TEST] --- Sequential batch_size={batch_size} ---", flush=True)

        # Reset motion frames for fair comparison
        for i in range(batch_size):
            get_base_data(pipeline, avatar_paths[i], base_seed=42 + i, use_face_crop=False)

        seq_times = []
        seq_outputs = []

        for run_idx in range(NUM_SEQUENTIAL_RUNS):
            torch.cuda.synchronize()
            run_t0 = time.monotonic()

            for i in range(batch_size):
                # Reset state for this session
                pipeline.ref_img_latent = ref_img_latent_list[i].clone()
                pipeline.latent_motion_frames = latent_motion_frames_list[i].clone()
                pipeline.generator = torch.Generator(device=pipeline.device).manual_seed(42 + i)
                pipeline.original_color_reference = original_color_refs[i].clone()
                pipeline.color_correction_strength = color_correction_strengths[i]

                frames = run_pipeline(pipeline, audio_embeddings[i])
                seq_outputs.append(frames)

                # Update motion frames for next iteration
                latent_motion_frames_list[i] = pipeline.latent_motion_frames.clone()

            torch.cuda.synchronize()
            run_ms = (time.monotonic() - run_t0) * 1000
            seq_times.append(run_ms)
            print(f"[TEST] Sequential run {run_idx+1}/{NUM_SEQUENTIAL_RUNS}: "
                  f"{run_ms:.1f}ms total ({run_ms/batch_size:.1f}ms/session)", flush=True)

        avg_seq = sum(seq_times) / len(seq_times)
        avg_per_session = avg_seq / batch_size
        results[f"seq_b{batch_size}"] = {
            "total_ms": avg_seq,
            "per_session_ms": avg_per_session,
            "all_runs": seq_times,
        }
        print(f"[TEST] Sequential batch={batch_size} AVG: {avg_seq:.1f}ms total, "
              f"{avg_per_session:.1f}ms/session", flush=True)

        # Log GPU memory
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[TEST] GPU mem after sequential: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)

    # ==========================================
    # Phase 2: Batched inference (batch=N, 1 run)
    # ==========================================
    print("\n" + "=" * 80, flush=True)
    print("[TEST] Phase 2: Batched inference (batch=N, 1 batched run)", flush=True)
    print("=" * 80, flush=True)

    for batch_size in BATCH_SIZES_TO_TEST:
        print(f"\n[TEST] --- Batched batch_size={batch_size} ---", flush=True)

        # Reset motion frames for fair comparison
        for i in range(batch_size):
            get_base_data(pipeline, avatar_paths[i], base_seed=42 + i, use_face_crop=False)
            latent_motion_frames_list[i] = pipeline.latent_motion_frames.clone()

        batch_times = []
        batch_outputs = []

        for run_idx in range(NUM_BATCHED_RUNS):
            torch.cuda.synchronize()
            run_t0 = time.monotonic()

            # Prepare per-session state for this run
            current_mf = [latent_motion_frames_list[i].clone() for i in range(batch_size)]
            current_gens = [torch.Generator(device=pipeline.device).manual_seed(42 + i) for i in range(batch_size)]
            current_refs = [original_color_refs[i].clone() for i in range(batch_size)]
            current_ccs = [color_correction_strengths[i] for i in range(batch_size)]
            current_audio = [audio_embeddings[i] for i in range(batch_size)]
            current_ref_latents = [ref_img_latent_list[i].clone() for i in range(batch_size)]

            frames_list, updated_mf_list = run_pipeline_batch(
                pipeline,
                current_audio,
                current_mf,
                current_ref_latents,
                current_gens,
                current_refs,
                current_ccs,
            )

            # Update motion frames for next run
            for i in range(batch_size):
                latent_motion_frames_list[i] = updated_mf_list[i]

            batch_outputs.append(frames_list)

            torch.cuda.synchronize()
            run_ms = (time.monotonic() - run_t0) * 1000
            batch_times.append(run_ms)
            print(f"[TEST] Batched run {run_idx+1}/{NUM_BATCHED_RUNS}: "
                  f"{run_ms:.1f}ms total ({run_ms/batch_size:.1f}ms/session)", flush=True)

        avg_batch = sum(batch_times) / len(batch_times)
        avg_per_session = avg_batch / batch_size
        results[f"batch_b{batch_size}"] = {
            "total_ms": avg_batch,
            "per_session_ms": avg_per_session,
            "all_runs": batch_times,
        }
        print(f"[TEST] Batched batch={batch_size} AVG: {avg_batch:.1f}ms total, "
              f"{avg_per_session:.1f}ms/session", flush=True)

        # Log GPU memory
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[TEST] GPU mem after batched: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)

    # ==========================================
    # Phase 3: Verify output shapes
    # ==========================================
    print("\n" + "=" * 80, flush=True)
    print("[TEST] Phase 3: Output verification", flush=True)
    print("=" * 80, flush=True)

    # Check that batched output has same shape as sequential
    if seq_outputs and batch_outputs:
        seq_shape = seq_outputs[0].shape
        batch_shape = batch_outputs[0][0].shape if batch_outputs[0] else None
        print(f"[TEST] Sequential output shape: {seq_shape}", flush=True)
        print(f"[TEST] Batched output shape: {batch_shape}", flush=True)
        if seq_shape == batch_shape:
            print("[TEST] PASS: Output shapes match", flush=True)
        else:
            print(f"[TEST] WARN: Output shapes differ! seq={seq_shape} vs batch={batch_shape}", flush=True)

    # ==========================================
    # Phase 4: Summary
    # ==========================================
    print("\n" + "=" * 80, flush=True)
    print("[TEST] SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n{'Batch':>6} | {'Seq Total':>12} | {'Batch Total':>12} | {'Seq/Sess':>10} | {'Batch/Sess':>12} | {'Speedup':>8}")
    print("-" * 75)

    for batch_size in BATCH_SIZES_TO_TEST:
        seq_key = f"seq_b{batch_size}"
        batch_key = f"batch_b{batch_size}"

        if seq_key in results and batch_key in results:
            seq_total = results[seq_key]["total_ms"]
            batch_total = results[batch_key]["total_ms"]
            seq_per = results[seq_key]["per_session_ms"]
            batch_per = results[batch_key]["per_session_ms"]
            speedup = seq_total / batch_total if batch_total > 0 else 0

            print(f"{batch_size:>6} | {seq_total:>10.1f}ms | {batch_total:>10.1f}ms | "
                  f"{seq_per:>8.1f}ms | {batch_per:>10.1f}ms | {speedup:>7.2f}x")

            if speedup > 1.0:
                print(f"       -> Batched is {speedup:.2f}x FASTER than sequential", flush=True)
            else:
                print(f"       -> Batched is {1/speedup:.2f}x SLOWER than sequential", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("[TEST] Test complete", flush=True)
    print("=" * 80, flush=True)

    return "done"
