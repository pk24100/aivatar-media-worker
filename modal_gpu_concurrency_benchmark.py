"""
Modal GPU Concurrency Benchmark - Phase 1

Inference-only benchmark that tests run_pipeline_batch() at various batch sizes
on different GPU types to determine maximum viable concurrency.

No LiveKit, no streaming, no HTTP server - just raw GPU inference timing.

Usage:
    modal run modal_gpu_concurrency_benchmark.py --gpu L40S
    modal run modal_gpu_concurrency_benchmark.py --gpu A100-40GB
    modal run modal_gpu_concurrency_benchmark.py --gpu H100
    modal run modal_gpu_concurrency_benchmark.py --gpu RTX-PRO-6000

The benchmark:
1. Loads FlashHead pipeline on the specified GPU
2. Warms up with 1 inference run
3. For each batch_size in [1, 2, 3, 4, 5, 8, 10, 12, 15]:
   a. Prepares N avatar sessions with different seeds/audio
   b. Runs 10 batched inference cycles
   c. Logs: inference_ms, per_session_ms, GPU util, VRAM
   d. Marks PASS/FAIL against 960ms budget
4. Reports max viable batch size for the GPU

PASS criteria: avg_inference_ms < 960ms (the real-time budget for 25 FPS)
"""

import os
import time
import json
import modal

BATCH_SIZES_TO_TEST = [3, 4, 5, 8, 10, 12, 15]
NUM_WARMUP_RUNS = 3
NUM_BENCHMARK_RUNS = 10
INFERENCE_BUDGET_MS = 960

# Multi-arch SageAttention build supporting all target GPUs:
# 8.0  = A100 (Ampere)
# 8.6  = A10 (Ampere)
# 8.9  = L40S, L4 (Ada)
# 9.0  = H100, H200 (Hopper)
# 10.0 = B200 (Blackwell GB100)
# 12.0 = RTX PRO 6000 (Blackwell GB202)
SAGEATTN_CUDA_ARCH_LIST = "8.0 8.6 8.9 9.0 10.0 12.0"

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
        f"TORCH_CUDA_ARCH_LIST='{SAGEATTN_CUDA_ARCH_LIST}' MAX_JOBS=8 NVCC_THREADS=4 "
        f"pip install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation "
        f"|| echo 'SAGEATTN_INSTALL_FAILED'",
    )
    .run_commands(
        "python -c \"from sageattention import sageattn; print('sageattention OK')\" 2>/dev/null "
        "|| echo 'sageattention NOT available'",
    )
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "huggingface-cli download pkam24100/aivatar-flashhead-model --local-dir /app/models/SoulX-FlashHead-1_3B",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_dir("SoulX-FlashHead", "/app/SoulX-FlashHead", copy=True)
    .add_local_file(
        "SoulX-FlashHead/flash_head/src/modules/flash_head_model_snapshot_patch.py",
        "/app/SoulX-FlashHead/flash_head/src/modules/flash_head_model.py",
        copy=True,
    )
    .add_local_dir("models/wav2vec2-base-960h", "/app/models/wav2vec2-base-960h", copy=True)
)

app = modal.App("aivatar-gpu-concurrency-benchmark", image=image)


def _log_gpu_info():
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        total_mem_gb = gpu_props.total_memory / (1024 ** 3)
        print(
            f"[BENCH] GPU: {gpu_name} | SMs={gpu_props.multi_processor_count} | "
            f"VRAM={total_mem_gb:.1f}GB",
            flush=True,
        )
        return gpu_name, gpu_props.multi_processor_count, total_mem_gb
    return "unknown", 0, 0.0


def _log_gpu_memory(label=""):
    import torch
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[BENCH] GPU mem {label}: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)
        return alloc, reserved
    return 0.0, 0.0


def _log_gpu_util():
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            util_pct = int(parts[0])
            temp_c = int(parts[1])
            mem_used_mb = int(parts[2])
            mem_total_mb = int(parts[3])
            print(
                f"[BENCH] GPU util={util_pct}% temp={temp_c}C "
                f"mem_used={mem_used_mb}MB/{mem_total_mb}MB",
                flush=True,
            )
            return util_pct, temp_c
    except Exception:
        pass
    return 0, 0


def _log_attention_backend():
    try:
        from sageattention import sageattn
        print("[BENCH] Attention backend: SageAttention", flush=True)
    except ImportError:
        print("[BENCH] Attention backend: SageAttention NOT available (will use SDPA fallback)", flush=True)
    try:
        import flash_attn
        print(f"[BENCH] Attention backend: flash_attn {flash_attn.__version__}", flush=True)
    except ImportError:
        pass


def run_benchmark(
    gpu_type: str = "L40S",
    batch_sizes: list = None,
    num_runs: int = NUM_BENCHMARK_RUNS,
):
    """Run batched inference benchmark on the specified GPU.

    Args:
        gpu_type: GPU type string (L4, A10, L40S, A100-40GB, H100, etc.)
        batch_sizes: List of batch sizes to test. Defaults to BATCH_SIZES_TO_TEST.
        num_runs: Number of benchmark runs per batch size.

    Returns:
        JSON string with full results.
    """
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

    if batch_sizes is None:
        batch_sizes = BATCH_SIZES_TO_TEST

    max_batch = max(batch_sizes)

    print("=" * 80, flush=True)
    print(f"[BENCH] GPU Concurrency Benchmark - {gpu_type}", flush=True)
    print(f"[BENCH] Batch sizes: {batch_sizes}")
    print(f"[BENCH] Runs per batch size: {num_runs}")
    print(f"[BENCH] Inference budget: {INFERENCE_BUDGET_MS}ms")
    print("=" * 80, flush=True)

    _log_attention_backend()

    # Load pipeline
    print(f"\n[BENCH] Loading FlashHead pipeline on {gpu_type}...", flush=True)
    load_t0 = time.monotonic()
    ckpt_dir = "/app/models/SoulX-FlashHead-1_3B"
    wav2vec_dir = "/app/models/wav2vec2-base-960h"
    model_type = "lite"
    pipeline = get_pipeline(1, ckpt_dir, model_type, wav2vec_dir)
    load_ms = (time.monotonic() - load_t0) * 1000
    print(f"[BENCH] Pipeline loaded in {load_ms:.1f}ms", flush=True)

    gpu_name, gpu_sms, gpu_vram_gb = _log_gpu_info()
    _log_gpu_memory("after pipeline load")

    # Prepare N avatar sessions with different colors and audio
    avatar_colors = [
        (128, 128, 128),
        (200, 100, 50),
        (50, 150, 200),
        (180, 50, 180),
        (100, 200, 100),
        (200, 200, 50),
        (50, 200, 200),
        (200, 50, 50),
        (150, 100, 200),
        (100, 50, 200),
        (200, 150, 100),
        (50, 100, 200),
        (200, 100, 200),
        (100, 200, 200),
        (200, 200, 200),
    ]

    print(f"\n[BENCH] Preparing {max_batch} avatar sessions...", flush=True)
    avatar_paths = []
    for i in range(max_batch):
        img_path = f"/tmp/avatar_{i}.png"
        color = avatar_colors[i % len(avatar_colors)]
        Image.new("RGB", (512, 512), color=color).save(img_path)
        avatar_paths.append(img_path)

    # Get inference params
    params = get_infer_params()
    sr = params["sample_rate"]
    cached_dur = params["cached_audio_duration"]
    frame_num = params["frame_num"]
    tgt_fps = params["tgt_fps"]
    motion_frames_num = params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num
    audio_end_idx = cached_dur * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    print(f"[BENCH] Params: frame_num={frame_num}, motion_frames={motion_frames_num}, "
          f"slice_len={slice_len}, tgt_fps={tgt_fps}, sample_rate={sr}", flush=True)

    # Generate different audio embeddings for each session
    print(f"[BENCH] Generating audio embeddings for {max_batch} sessions...", flush=True)
    audio_embeddings = []
    for i in range(max_batch):
        t = np.linspace(0, cached_dur, cached_dur * sr, dtype=np.float32)
        freq = 200 + i * 100
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        emb = get_audio_embedding(pipeline, audio, audio_start_idx, audio_end_idx)
        audio_embeddings.append(emb)

    # Prepare per-session pipeline state
    ref_img_latent_list = []
    latent_motion_frames_list = []
    original_color_refs = []
    color_correction_strengths = []

    for i in range(max_batch):
        get_base_data(pipeline, avatar_paths[i], base_seed=42 + i, use_face_crop=False)
        ref_img_latent_list.append(pipeline.ref_img_latent.clone())
        latent_motion_frames_list.append(pipeline.latent_motion_frames.clone())
        original_color_refs.append(pipeline.original_color_reference.clone())
        color_correction_strengths.append(pipeline.color_correction_strength)

    print(f"[BENCH] Per-session state prepared for {max_batch} sessions", flush=True)
    _log_gpu_memory("after session prep")

    # Warmup
    print(f"\n[BENCH] Warming up pipeline ({NUM_WARMUP_RUNS} runs)...", flush=True)
    for w in range(NUM_WARMUP_RUNS):
        pipeline.ref_img_latent = ref_img_latent_list[0].clone()
        pipeline.latent_motion_frames = latent_motion_frames_list[0].clone()
        pipeline.generator = torch.Generator(device=pipeline.device).manual_seed(42)
        pipeline.original_color_reference = original_color_refs[0].clone()
        pipeline.color_correction_strength = color_correction_strengths[0]
        _ = run_pipeline(pipeline, audio_embeddings[0])
        torch.cuda.synchronize()
    print(f"[BENCH] Warmup complete", flush=True)
    _log_gpu_memory("after warmup")

    # Benchmark each batch size
    all_results = {
        "gpu_type": gpu_type,
        "gpu_name": gpu_name,
        "gpu_sms": gpu_sms,
        "gpu_vram_gb": round(gpu_vram_gb, 1),
        "inference_budget_ms": INFERENCE_BUDGET_MS,
        "num_runs_per_batch": num_runs,
        "pipeline_load_ms": round(load_ms, 1),
        "frame_num": frame_num,
        "slice_len": slice_len,
        "tgt_fps": tgt_fps,
        "batch_results": [],
    }

    print(f"[BENCH] FPS calculation: per_session_fps = slice_len / (inference_ms / 1000)", flush=True)
    print(f"[BENCH] Real-time target: per_session_fps >= {tgt_fps} (slice_len={slice_len} frames per cycle)", flush=True)

    for batch_size in batch_sizes:
        print(f"\n{'=' * 80}", flush=True)
        print(f"[BENCH] --- Batch size = {batch_size} ---", flush=True)
        print(f"{'=' * 80}", flush=True)

        # Reset motion frames for fair comparison
        for i in range(batch_size):
            get_base_data(pipeline, avatar_paths[i], base_seed=42 + i, use_face_crop=False)
            latent_motion_frames_list[i] = pipeline.latent_motion_frames.clone()

        batch_times = []

        for run_idx in range(num_runs):
            # Prepare per-session state for this run
            current_mf = [latent_motion_frames_list[i].clone() for i in range(batch_size)]
            current_gens = [torch.Generator(device=pipeline.device).manual_seed(42 + i) for i in range(batch_size)]
            current_refs = [original_color_refs[i].clone() for i in range(batch_size)]
            current_ccs = [color_correction_strengths[i] for i in range(batch_size)]
            current_audio = [audio_embeddings[i] for i in range(batch_size)]
            current_ref_latents = [ref_img_latent_list[i].clone() for i in range(batch_size)]

            torch.cuda.synchronize()
            run_t0 = time.monotonic()

            frames_list, updated_mf_list = run_pipeline_batch(
                pipeline,
                current_audio,
                current_mf,
                current_ref_latents,
                current_gens,
                current_refs,
                current_ccs,
            )

            torch.cuda.synchronize()
            run_ms = (time.monotonic() - run_t0) * 1000
            batch_times.append(run_ms)

            # Update motion frames for next run (chaining like production)
            for i in range(batch_size):
                latent_motion_frames_list[i] = updated_mf_list[i]

            per_session_ms = run_ms / batch_size
            print(
                f"[BENCH] Run {run_idx + 1}/{num_runs}: {run_ms:.1f}ms total "
                f"({per_session_ms:.1f}ms/session)",
                flush=True,
            )

        # Calculate statistics
        avg_ms = sum(batch_times) / len(batch_times)
        min_ms = min(batch_times)
        max_ms = max(batch_times)
        p50_ms = float(np.percentile(batch_times, 50))
        p95_ms = float(np.percentile(batch_times, 95))
        per_session_avg = avg_ms / batch_size
        per_session_p95 = p95_ms / batch_size

        # FPS calculation: each inference cycle generates slice_len new frames per session
        # per_session_generated_fps = slice_len / (inference_ms / 1000)
        # For real-time, this must be >= tgt_fps (25)
        per_session_fps_avg = slice_len / (avg_ms / 1000) if avg_ms > 0 else 0
        per_session_fps_p95 = slice_len / (p95_ms / 1000) if p95_ms > 0 else 0
        total_fps = (batch_size * slice_len) / (avg_ms / 1000) if avg_ms > 0 else 0

        # GPU memory and utilization after this batch size
        gpu_alloc_gb, gpu_reserved_gb = _log_gpu_memory(f"after batch={batch_size}")
        gpu_util, gpu_temp = _log_gpu_util()

        # PASS/FAIL against budget
        passes = avg_ms < INFERENCE_BUDGET_MS
        p95_passes = p95_ms < INFERENCE_BUDGET_MS

        status = "PASS" if passes else "FAIL"
        p95_status = "PASS" if p95_passes else "FAIL"

        print(
            f"\n[BENCH] Batch={batch_size} SUMMARY: avg={avg_ms:.1f}ms "
            f"p50={p50_ms:.1f}ms p95={p95_ms:.1f}ms "
            f"per_session={per_session_avg:.1f}ms "
            f"per_session_fps={per_session_fps_avg:.1f} (p95={per_session_fps_p95:.1f}) "
            f"total_fps={total_fps:.1f} "
            f"GPU_mem={gpu_alloc_gb:.2f}GB "
            f"STATUS={status} (p95={p95_status})",
            flush=True,
        )

        batch_result = {
            "batch_size": batch_size,
            "avg_ms": round(avg_ms, 1),
            "min_ms": round(min_ms, 1),
            "max_ms": round(max_ms, 1),
            "p50_ms": round(p50_ms, 1),
            "p95_ms": round(p95_ms, 1),
            "per_session_avg_ms": round(per_session_avg, 1),
            "per_session_p95_ms": round(per_session_p95, 1),
            "per_session_generated_fps_avg": round(per_session_fps_avg, 1),
            "per_session_generated_fps_p95": round(per_session_fps_p95, 1),
            "total_generated_fps": round(total_fps, 1),
            "all_runs": [round(t, 1) for t in batch_times],
            "gpu_alloc_gb": round(gpu_alloc_gb, 2),
            "gpu_reserved_gb": round(gpu_reserved_gb, 2),
            "gpu_util_pct": gpu_util,
            "gpu_temp_c": gpu_temp,
            "status": status,
            "p95_status": p95_status,
        }
        all_results["batch_results"].append(batch_result)

        # Early exit if we're well over budget - no point testing higher batches
        if not passes and avg_ms > INFERENCE_BUDGET_MS * 1.5:
            print(
                f"\n[BENCH] Early exit: batch={batch_size} exceeded budget by 1.5x "
                f"({avg_ms:.1f}ms > {INFERENCE_BUDGET_MS * 1.5:.0f}ms)",
                flush=True,
            )
            remaining = [bs for bs in batch_sizes if bs > batch_size]
            for bs in remaining:
                all_results["batch_results"].append({
                    "batch_size": bs,
                    "status": "SKIPPED",
                    "reason": f"Early exit after batch={batch_size} exceeded budget by 1.5x",
                })
            break

    # Find max viable batch size
    max_viable = 0
    for br in all_results["batch_results"]:
        if br.get("status") == "PASS":
            max_viable = br["batch_size"]

    all_results["max_viable_batch_size"] = max_viable

    # Print final summary
    print(f"\n{'=' * 80}", flush=True)
    print(f"[BENCH] FINAL SUMMARY - {gpu_type} ({gpu_name})", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(
        f"\n{'Batch':>6} | {'Avg (ms)':>10} | {'P95 (ms)':>10} | {'Per/Sess':>10} | "
        f"{'Sess FPS':>10} | {'Total FPS':>10} | {'GPU Mem':>10} | {'Status':>8}",
        flush=True,
    )
    print("-" * 95)

    for br in all_results["batch_results"]:
        if br.get("status") == "SKIPPED":
            print(f"{br['batch_size']:>6} | {'-':>10} | {'-':>10} | {'-':>10} | {'-':>10} | {'-':>10} | {'-':>10} | {'SKIP':>8}")
            continue
        print(
            f"{br['batch_size']:>6} | {br['avg_ms']:>8.1f}ms | {br['p95_ms']:>8.1f}ms | "
            f"{br['per_session_avg_ms']:>8.1f}ms | {br['per_session_generated_fps_avg']:>8.1f} | "
            f"{br['total_generated_fps']:>8.1f} | {br['gpu_alloc_gb']:>6.2f}GB | {br['status']:>8}",
            flush=True,
        )

    print(f"\n[BENCH] Max viable batch size: {max_viable}", flush=True)
    print(f"[BENCH] Inference budget: {INFERENCE_BUDGET_MS}ms", flush=True)
    print(f"[BENCH] Target FPS: {tgt_fps} (slice_len={slice_len} new frames per cycle)", flush=True)
    print(f"{'=' * 80}", flush=True)

    # Return JSON results (also printed for log capture)
    results_json = json.dumps(all_results, indent=2)
    print(f"\n[BENCH_JSON] {results_json}", flush=True)
    return all_results


@app.local_entrypoint()
def main(
    gpu: str = "L40S",
    batch_sizes: str = "",
    num_runs: int = NUM_BENCHMARK_RUNS,
):
    """Local entrypoint for `modal run` command.

    Args:
        gpu: GPU type (L4, A10, L40S, A100-40GB, H100, RTX-PRO-6000, B200, etc.)
        batch_sizes: Comma-separated batch sizes (e.g. "1,3,5"). Empty = default.
        num_runs: Number of benchmark runs per batch size.
    """
    from datetime import datetime
    from pathlib import Path

    bs_list = None
    if batch_sizes:
        bs_list = [int(x.strip()) for x in batch_sizes.split(",")]

    print(f"[ENTRYPOINT] Starting GPU concurrency benchmark for {gpu}")
    print(f"[ENTRYPOINT] Batch sizes: {bs_list or BATCH_SIZES_TO_TEST}")
    print(f"[ENTRYPOINT] Runs per batch: {num_runs}")

    fn = app.function(
        gpu=gpu,
        timeout=3600,
        secrets=[modal.Secret.from_name("huggingface-secret")],
        cpu=4,
        memory=16384,
    )(run_benchmark)

    result = fn.remote(
        gpu_type=gpu,
        batch_sizes=bs_list,
        num_runs=num_runs,
    )

    # Save JSON results to test_logs directory
    log_dir = Path("scripts/test_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_safe = gpu.replace(" ", "_").replace("-", "_")
    results_path = log_dir / f"benchmark_{gpu_safe}_{timestamp}.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[ENTRYPOINT] Results saved to: {results_path}")
    print(f"[ENTRYPOINT] Max viable batch size: {result.get('max_viable_batch_size', 'N/A')}")

    # Also suggest how to capture full Modal logs:
    print(f"\n[ENTRYPOINT] To capture full Modal logs, re-run with:")
    print(f"  modal run modal_gpu_concurrency_benchmark.py --gpu {gpu} "
          f"| Set-Content -Path 'scripts/test_logs/benchmark_{gpu_safe}_{timestamp}.log' -Encoding UTF8")
