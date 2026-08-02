"""
Realistic batched inference streaming benchmark on L40S GPU.

Simulates the actual streaming engine conditions:
- 3 independent sessions with staggered audio arrival (jitter)
- Audio embedding generation per session (get_audio_embedding)
- GPU-to-CPU transfer (.cpu().numpy())
- Continuous slice-by-slice processing for a fixed duration
- Per-session wall-clock latency: audio_ready -> frames_in_hand

Compares three modes:
  MODE A (sequential): Run run_pipeline 3 times back-to-back (simulates
    current production without green contexts, single thread, all 142 SMs)
  MODE B (fixed batched): Wait up to WAIT_WINDOW_MS for sessions to be ready,
    then run run_pipeline_batch once for all ready sessions
  MODE C (adaptive batched): Wait for FIRST session to be ready, then wait
    up to ADAPTIVE_WINDOWS_MS more for additional sessions to arrive

Logs are written to both terminal and a local file (benchmark_results.log)

Usage:
    modal run modal_batched_streaming_benchmark.py
"""

import os
import time
import modal

# --- Config ---
NUM_SESSIONS = 3
DURATION_SECONDS = 30  # Per mode (reduced since we run multiple wait windows)
WAIT_WINDOWS_MS = [20, 50]  # Fixed wait windows to test
ADAPTIVE_WINDOWS_MS = [20, 30, 40]  # Adaptive wait windows (wait X ms after FIRST session ready)
AUDIO_JITTER_MS = 50  # Simulated audio arrival jitter per session per slice
NUM_WARMUP_SLICES = 3
LOG_FILE = "/tmp/benchmark_results.log"

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

app = modal.App("aivatar-batched-streaming-benchmark", image=image)


class SimulatedSession:
    """Simulates a single streaming session with audio context and per-session state."""
    def __init__(self, session_id, pipeline, avatar_path, seed, audio_freq,
                 infer_params):
        self.id = session_id
        self.pipeline = pipeline
        self.seed = seed
        self.audio_freq = audio_freq

        self.frame_num = infer_params["frame_num"]
        self.motion_frames_num = infer_params["motion_frames_num"]
        self.slice_len = self.frame_num - self.motion_frames_num
        self.sample_rate = infer_params["sample_rate"]
        self.tgt_fps = infer_params["tgt_fps"]
        self.cached_audio_duration = infer_params["cached_audio_duration"]
        self.slice_samples = self.slice_len * self.sample_rate // self.tgt_fps
        self.cached_audio_samples = self.cached_audio_duration * self.sample_rate
        self.audio_end_idx = self.cached_audio_duration * self.tgt_fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num

        from collections import deque
        self.audio_context = deque(
            [0.0] * self.cached_audio_samples,
            maxlen=self.cached_audio_samples,
        )

        # Prepare avatar and extract per-session state
        from flash_head.inference import get_base_data
        get_base_data(pipeline, avatar_path, base_seed=seed, use_face_crop=False)
        self.ref_img_latent = pipeline.ref_img_latent.clone()
        self.latent_motion_frames = pipeline.latent_motion_frames.clone()
        self.original_color_reference = pipeline.original_color_reference.clone()
        self.color_correction_strength = pipeline.color_correction_strength

        # Generate continuous audio stream (different per session)
        import numpy as np
        total_samples = DURATION_SECONDS * self.sample_rate + self.cached_audio_samples
        t = np.linspace(0, total_samples / self.sample_rate, total_samples, dtype=np.float32)
        # Mix of frequencies to simulate speech-like patterns
        self.audio_stream = (
            0.3 * np.sin(2 * np.pi * audio_freq * t)
            + 0.2 * np.sin(2 * np.pi * (audio_freq * 1.5) * t)
            + 0.1 * np.sin(2 * np.pi * (audio_freq * 0.5) * t)
        ).astype(np.float32)
        self.audio_pos = 0

        # Metrics
        self.latencies = []
        self.slices_processed = 0

    def get_audio_chunk(self):
        """Get next slice of audio samples and extend context."""
        import numpy as np
        chunk = self.audio_stream[self.audio_pos:self.audio_pos + self.slice_samples]
        self.audio_pos += self.slice_samples
        self.audio_context.extend(chunk.tolist())
        return np.array(self.audio_context, dtype=np.float32)

    def get_audio_embedding(self):
        """Generate audio embedding from current context."""
        from flash_head.inference import get_audio_embedding as _get_emb
        audio_array = self.get_audio_chunk()
        return _get_emb(self.pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)

    def update_motion_frames(self, new_mf):
        self.latent_motion_frames = new_mf


@app.function(gpu="L40S", timeout=1800, secrets=[modal.Secret.from_name("huggingface-secret")])
def run_streaming_benchmark():
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/SoulX-FlashHead")

    import torch
    import numpy as np
    from PIL import Image

    os.environ["ENGINE_PROFILE"] = "1"

    # --- File + terminal logging setup ---
    class DualLogger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log_file = open(filepath, "w", buffering=1)
        def write(self, message):
            self.terminal.write(message)
            self.log_file.write(message)
        def flush(self):
            self.terminal.flush()
            self.log_file.flush()

    sys.stdout = DualLogger(LOG_FILE)
    print(f"[BENCH] Logging to file: {LOG_FILE}", flush=True)

    from flash_head.inference import (
        get_pipeline, get_audio_embedding, run_pipeline, run_pipeline_batch,
        get_infer_params,
    )

    ckpt_dir = "/app/models/SoulX-FlashHead-1_3B"
    wav2vec_dir = "/app/models/wav2vec2-base-960h"
    model_type = "lite"

    print("=" * 80, flush=True)
    print("[BENCH] Realistic Batched Streaming Benchmark - L40S", flush=True)
    print(f"[BENCH] Sessions={NUM_SESSIONS}, Duration={DURATION_SECONDS}s/mode, "
          f"FixedWindows={WAIT_WINDOWS_MS}ms, AdaptiveWindows={ADAPTIVE_WINDOWS_MS}ms, "
          f"Jitter={AUDIO_JITTER_MS}ms", flush=True)
    print("=" * 80, flush=True)

    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"[BENCH] GPU: {gpu_name} ({gpu_props.multi_processor_count} SMs)", flush=True)

    try:
        from sageattention import sageattn
        print("[BENCH] Attention: SageAttention", flush=True)
    except ImportError:
        pass
    try:
        import flash_attn
        print(f"[BENCH] Attention: flash_attn {flash_attn.__version__}", flush=True)
    except ImportError:
        pass

    # Load pipeline
    print("[BENCH] Loading pipeline...", flush=True)
    load_t0 = time.monotonic()
    pipeline = get_pipeline(1, ckpt_dir, model_type, wav2vec_dir)
    print(f"[BENCH] Pipeline loaded in {(time.monotonic()-load_t0)*1000:.1f}ms", flush=True)

    params = get_infer_params()

    # Create avatar images
    avatar_colors = [(128, 128, 128), (200, 100, 50), (50, 150, 200)]
    avatar_paths = []
    for i in range(NUM_SESSIONS):
        p = f"/tmp/avatar_{i}.png"
        Image.new("RGB", (512, 512), color=avatar_colors[i]).save(p)
        avatar_paths.append(p)

    # Create sessions
    sessions = []
    for i in range(NUM_SESSIONS):
        s = SimulatedSession(
            session_id=i,
            pipeline=pipeline,
            avatar_path=avatar_paths[i],
            seed=42 + i,
            audio_freq=200 + i * 100,
            infer_params=params,
        )
        sessions.append(s)
    print(f"[BENCH] {NUM_SESSIONS} sessions prepared", flush=True)

    # Warmup
    print("[BENCH] Warming up...", flush=True)
    for _ in range(NUM_WARMUP_SLICES):
        emb = sessions[0].get_audio_embedding()
        _ = run_pipeline(pipeline, emb)
    torch.cuda.synchronize()
    print("[BENCH] Warmup complete", flush=True)

    # ==========================================
    # MODE A: Sequential (simulates current production without green contexts)
    # ==========================================
    print("\n" + "=" * 80, flush=True)
    print("[BENCH] MODE A: Sequential (3x run_pipeline, all 142 SMs each)", flush=True)
    print("=" * 80, flush=True)

    # Reset sessions
    for s in sessions:
        from flash_head.inference import get_base_data
        get_base_data(pipeline, avatar_paths[s.id], base_seed=s.seed, use_face_crop=False)
        s.ref_img_latent = pipeline.ref_img_latent.clone()
        s.latent_motion_frames = pipeline.latent_motion_frames.clone()
        s.original_color_reference = pipeline.original_color_reference.clone()
        s.audio_pos = 0
        s.latencies = []
        s.slices_processed = 0

    mode_a_start = time.monotonic()
    slice_count = 0
    while (time.monotonic() - mode_a_start) < DURATION_SECONDS:
        slice_t0 = time.monotonic()

        for s in sessions:
            # Simulate audio arrival jitter
            jitter = np.random.uniform(0, AUDIO_JITTER_MS) / 1000.0
            time.sleep(jitter)

            # Per-session processing (like _process_available_audio)
            t_embed = time.monotonic()
            audio_emb = s.get_audio_embedding()
            embed_ms = (time.monotonic() - t_embed) * 1000

            # Set pipeline state for this session
            pipeline.ref_img_latent = s.ref_img_latent.clone()
            pipeline.latent_motion_frames = s.latent_motion_frames.clone()
            pipeline.generator = torch.Generator(device=pipeline.device).manual_seed(s.seed)
            pipeline.original_color_reference = s.original_color_reference.clone()
            pipeline.color_correction_strength = s.color_correction_strength

            # Inference
            t_infer = time.monotonic()
            video = run_pipeline(pipeline, audio_emb)
            infer_ms = (time.monotonic() - t_infer) * 1000

            # Transfer to CPU (like streaming engine)
            t_xfer = time.monotonic()
            video = video[s.motion_frames_num:]
            frames_np = video.cpu().numpy().astype(np.uint8)
            xfer_ms = (time.monotonic() - t_xfer) * 1000

            # Update motion frames
            s.update_motion_frames(pipeline.latent_motion_frames.clone())

            # Per-session latency: audio ready -> frames in hand
            latency_ms = (time.monotonic() - slice_t0) * 1000
            s.latencies.append(latency_ms)
            s.slices_processed += 1

        slice_count += 1
        if slice_count % 10 == 0:
            elapsed = time.monotonic() - mode_a_start
            avg_lat = np.mean([np.mean(s.latencies[-10:]) for s in sessions])
            print(f"[BENCH-A] slice={slice_count} elapsed={elapsed:.1f}s "
                  f"avg_latency_10={avg_lat:.1f}ms", flush=True)

    mode_a_elapsed = time.monotonic() - mode_a_start

    # Collect Mode A stats
    all_latencies_a = []
    for s in sessions:
        all_latencies_a.extend(s.latencies)

    print(f"\n[BENCH-A] DONE: {slice_count} slices in {mode_a_elapsed:.1f}s", flush=True)
    print(f"[BENCH-A] Total slices per session: {[s.slices_processed for s in sessions]}", flush=True)
    print(f"[BENCH-A] Latency stats (ms): "
          f"avg={np.mean(all_latencies_a):.1f} "
          f"p50={np.percentile(all_latencies_a, 50):.1f} "
          f"p95={np.percentile(all_latencies_a, 95):.1f} "
          f"max={np.max(all_latencies_a):.1f}", flush=True)

    # GPU memory
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"[BENCH-A] GPU mem: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)

    # ==========================================
    # MODE B: Batched - test multiple wait windows
    # ==========================================
    all_mode_b_results = {}

    for wait_window_ms in WAIT_WINDOWS_MS:
        print("\n" + "=" * 80, flush=True)
        print(f"[BENCH] MODE B: Batched (wait_window={wait_window_ms}ms)", flush=True)
        print("=" * 80, flush=True)

        # Reset sessions
        for s in sessions:
            from flash_head.inference import get_base_data
            get_base_data(pipeline, avatar_paths[s.id], base_seed=s.seed, use_face_crop=False)
            s.ref_img_latent = pipeline.ref_img_latent.clone()
            s.latent_motion_frames = pipeline.latent_motion_frames.clone()
            s.original_color_reference = pipeline.original_color_reference.clone()
            s.audio_pos = 0
            s.latencies = []
            s.slices_processed = 0

        batch_sizes_seen = []
        mode_b_start = time.monotonic()
        slice_count_b = 0

        while (time.monotonic() - mode_b_start) < DURATION_SECONDS:
            cycle_t0 = time.monotonic()

            # Simulate audio arrival: each session has audio ready at different times
            ready_times = []
            for s in sessions:
                jitter = np.random.uniform(0, AUDIO_JITTER_MS) / 1000.0
                ready_times.append(time.monotonic() + jitter)

            # Wait for sessions to be ready (with wait window)
            ready_sessions = []
            wait_deadline = time.monotonic() + (wait_window_ms / 1000.0)

            # Sort sessions by ready time
            sorted_indices = sorted(range(NUM_SESSIONS), key=lambda i: ready_times[i])

            for idx in sorted_indices:
                now = time.monotonic()
                if ready_times[idx] <= wait_deadline:
                    # Wait until this session is ready
                    if ready_times[idx] > now:
                        time.sleep(ready_times[idx] - now)
                    ready_sessions.append(idx)
                else:
                    # Session won't be ready within window, skip it
                    break

            # If no sessions ready, process the first one anyway
            if not ready_sessions:
                ready_sessions = [sorted_indices[0]]
                time.sleep(max(0, ready_times[sorted_indices[0]] - time.monotonic()))

            batch_size = len(ready_sessions)
            batch_sizes_seen.append(batch_size)

            # Generate audio embeddings for ready sessions
            t_embed = time.monotonic()
            audio_embs = []
            for idx in ready_sessions:
                s = sessions[idx]
                audio_embs.append(s.get_audio_embedding())
            embed_ms = (time.monotonic() - t_embed) * 1000

            # Prepare per-session state for batched call
            current_mf = [sessions[idx].latent_motion_frames.clone() for idx in ready_sessions]
            current_gens = [torch.Generator(device=pipeline.device).manual_seed(sessions[idx].seed) for idx in ready_sessions]
            current_refs = [sessions[idx].original_color_reference.clone() for idx in ready_sessions]
            current_ccs = [sessions[idx].color_correction_strength for idx in ready_sessions]
            current_ref_latents = [sessions[idx].ref_img_latent.clone() for idx in ready_sessions]

            # Batched inference
            t_infer = time.monotonic()
            frames_list, updated_mf_list = run_pipeline_batch(
                pipeline,
                audio_embs,
                current_mf,
                current_ref_latents,
                current_gens,
                current_refs,
                current_ccs,
            )
            infer_ms = (time.monotonic() - t_infer) * 1000

            # Transfer to CPU per session
            t_xfer = time.monotonic()
            for i, idx in enumerate(ready_sessions):
                video = frames_list[i][sessions[idx].motion_frames_num:]
                frames_np = video.cpu().numpy().astype(np.uint8)
                sessions[idx].update_motion_frames(updated_mf_list[i])
            xfer_ms = (time.monotonic() - t_xfer) * 1000

            # Per-session latency: cycle start -> frames in hand
            cycle_latency = (time.monotonic() - cycle_t0) * 1000
            for idx in ready_sessions:
                sessions[idx].latencies.append(cycle_latency)
                sessions[idx].slices_processed += 1

            slice_count_b += 1
            if slice_count_b % 10 == 0:
                elapsed = time.monotonic() - mode_b_start
                recent_lats = [l for s in sessions for l in s.latencies[-10:]]
                avg_lat = np.mean(recent_lats) if recent_lats else 0
                avg_batch = np.mean(batch_sizes_seen[-10:])
                print(f"[BENCH-B-{wait_window_ms}] slice={slice_count_b} elapsed={elapsed:.1f}s "
                      f"avg_batch={avg_batch:.1f} avg_latency_10={avg_lat:.1f}ms "
                      f"embed={embed_ms:.1f}ms infer={infer_ms:.1f}ms xfer={xfer_ms:.1f}ms",
                      flush=True)

        mode_b_elapsed = time.monotonic() - mode_b_start

        # Collect Mode B stats
        all_latencies_b = []
        for s in sessions:
            all_latencies_b.extend(s.latencies)

        fill_1 = batch_sizes_seen.count(1) / len(batch_sizes_seen) * 100 if batch_sizes_seen else 0
        fill_2 = batch_sizes_seen.count(2) / len(batch_sizes_seen) * 100 if batch_sizes_seen else 0
        fill_3 = batch_sizes_seen.count(3) / len(batch_sizes_seen) * 100 if batch_sizes_seen else 0

        print(f"\n[BENCH-B-{wait_window_ms}] DONE: {slice_count_b} cycles in {mode_b_elapsed:.1f}s", flush=True)
        print(f"[BENCH-B-{wait_window_ms}] Slices per session: {[s.slices_processed for s in sessions]}", flush=True)
        print(f"[BENCH-B-{wait_window_ms}] Batch sizes: 1={batch_sizes_seen.count(1)} 2={batch_sizes_seen.count(2)} 3={batch_sizes_seen.count(3)}", flush=True)
        print(f"[BENCH-B-{wait_window_ms}] Avg batch size: {np.mean(batch_sizes_seen):.2f}", flush=True)
        print(f"[BENCH-B-{wait_window_ms}] Fill rate: 1/3={fill_1:.0f}% 2/3={fill_2:.0f}% 3/3={fill_3:.0f}%", flush=True)
        print(f"[BENCH-B-{wait_window_ms}] Latency stats (ms): "
              f"avg={np.mean(all_latencies_b):.1f} "
              f"p50={np.percentile(all_latencies_b, 50):.1f} "
              f"p95={np.percentile(all_latencies_b, 95):.1f} "
              f"max={np.max(all_latencies_b):.1f}", flush=True)

        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[BENCH-B-{wait_window_ms}] GPU mem: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)

        all_mode_b_results[wait_window_ms] = {
            "avg_latency": np.mean(all_latencies_b),
            "p50_latency": np.percentile(all_latencies_b, 50),
            "p95_latency": np.percentile(all_latencies_b, 95),
            "max_latency": np.max(all_latencies_b),
            "total_slices": slice_count_b,
            "throughput": slice_count_b / mode_b_elapsed,
            "avg_batch": np.mean(batch_sizes_seen),
            "fill_1": fill_1,
            "fill_2": fill_2,
            "fill_3": fill_3,
            "gpu_reserved": reserved,
        }

        # Clear cache between runs
        torch.cuda.empty_cache()

    # ==========================================
    # MODE C: Adaptive (wait X ms after FIRST session ready)
    # ==========================================
    all_mode_c_results = {}

    for adaptive_window_ms in ADAPTIVE_WINDOWS_MS:
        print("\n" + "=" * 80, flush=True)
        print(f"[BENCH] MODE C: Adaptive (wait {adaptive_window_ms}ms after first session ready)", flush=True)
        print("=" * 80, flush=True)

        # Reset sessions
        for s in sessions:
            from flash_head.inference import get_base_data
            get_base_data(pipeline, avatar_paths[s.id], base_seed=s.seed, use_face_crop=False)
            s.ref_img_latent = pipeline.ref_img_latent.clone()
            s.latent_motion_frames = pipeline.latent_motion_frames.clone()
            s.original_color_reference = pipeline.original_color_reference.clone()
            s.audio_pos = 0
            s.latencies = []
            s.slices_processed = 0

        batch_sizes_seen = []
        mode_c_start = time.monotonic()
        slice_count_c = 0

        while (time.monotonic() - mode_c_start) < DURATION_SECONDS:
            cycle_t0 = time.monotonic()

            # Simulate audio arrival: each session has audio ready at different times
            ready_times = []
            for s in sessions:
                jitter = np.random.uniform(0, AUDIO_JITTER_MS) / 1000.0
                ready_times.append(time.monotonic() + jitter)

            # Sort sessions by ready time
            sorted_indices = sorted(range(NUM_SESSIONS), key=lambda i: ready_times[i])

            # Wait for FIRST session to be ready
            first_idx = sorted_indices[0]
            first_ready = ready_times[first_idx]
            now = time.monotonic()
            if first_ready > now:
                time.sleep(first_ready - now)

            # Now wait up to adaptive_window_ms for more sessions
            adaptive_deadline = time.monotonic() + (adaptive_window_ms / 1000.0)
            ready_sessions = [first_idx]

            for idx in sorted_indices[1:]:
                if ready_times[idx] <= adaptive_deadline:
                    # Wait until this session is ready
                    now = time.monotonic()
                    if ready_times[idx] > now:
                        time.sleep(ready_times[idx] - now)
                    ready_sessions.append(idx)
                else:
                    break

            batch_size = len(ready_sessions)
            batch_sizes_seen.append(batch_size)

            # Generate audio embeddings for ready sessions
            t_embed = time.monotonic()
            audio_embs = []
            for idx in ready_sessions:
                s = sessions[idx]
                audio_embs.append(s.get_audio_embedding())
            embed_ms = (time.monotonic() - t_embed) * 1000

            # Prepare per-session state for batched call
            current_mf = [sessions[idx].latent_motion_frames.clone() for idx in ready_sessions]
            current_gens = [torch.Generator(device=pipeline.device).manual_seed(sessions[idx].seed) for idx in ready_sessions]
            current_refs = [sessions[idx].original_color_reference.clone() for idx in ready_sessions]
            current_ccs = [sessions[idx].color_correction_strength for idx in ready_sessions]
            current_ref_latents = [sessions[idx].ref_img_latent.clone() for idx in ready_sessions]

            # Batched inference
            t_infer = time.monotonic()
            frames_list, updated_mf_list = run_pipeline_batch(
                pipeline,
                audio_embs,
                current_mf,
                current_ref_latents,
                current_gens,
                current_refs,
                current_ccs,
            )
            infer_ms = (time.monotonic() - t_infer) * 1000

            # Transfer to CPU per session
            t_xfer = time.monotonic()
            for i, idx in enumerate(ready_sessions):
                video = frames_list[i][sessions[idx].motion_frames_num:]
                frames_np = video.cpu().numpy().astype(np.uint8)
                sessions[idx].update_motion_frames(updated_mf_list[i])
            xfer_ms = (time.monotonic() - t_xfer) * 1000

            # Per-session latency: cycle start -> frames in hand
            cycle_latency = (time.monotonic() - cycle_t0) * 1000
            for idx in ready_sessions:
                sessions[idx].latencies.append(cycle_latency)
                sessions[idx].slices_processed += 1

            slice_count_c += 1
            if slice_count_c % 10 == 0:
                elapsed = time.monotonic() - mode_c_start
                recent_lats = [l for s in sessions for l in s.latencies[-10:]]
                avg_lat = np.mean(recent_lats) if recent_lats else 0
                avg_batch = np.mean(batch_sizes_seen[-10:])
                print(f"[BENCH-C-{adaptive_window_ms}] slice={slice_count_c} elapsed={elapsed:.1f}s "
                      f"avg_batch={avg_batch:.1f} avg_latency_10={avg_lat:.1f}ms "
                      f"embed={embed_ms:.1f}ms infer={infer_ms:.1f}ms xfer={xfer_ms:.1f}ms",
                      flush=True)

        mode_c_elapsed = time.monotonic() - mode_c_start

        # Collect Mode C stats
        all_latencies_c = []
        for s in sessions:
            all_latencies_c.extend(s.latencies)

        fill_1 = batch_sizes_seen.count(1) / len(batch_sizes_seen) * 100 if batch_sizes_seen else 0
        fill_2 = batch_sizes_seen.count(2) / len(batch_sizes_seen) * 100 if batch_sizes_seen else 0
        fill_3 = batch_sizes_seen.count(3) / len(batch_sizes_seen) * 100 if batch_sizes_seen else 0

        print(f"\n[BENCH-C-{adaptive_window_ms}] DONE: {slice_count_c} cycles in {mode_c_elapsed:.1f}s", flush=True)
        print(f"[BENCH-C-{adaptive_window_ms}] Slices per session: {[s.slices_processed for s in sessions]}", flush=True)
        print(f"[BENCH-C-{adaptive_window_ms}] Batch sizes: 1={batch_sizes_seen.count(1)} 2={batch_sizes_seen.count(2)} 3={batch_sizes_seen.count(3)}", flush=True)
        print(f"[BENCH-C-{adaptive_window_ms}] Avg batch size: {np.mean(batch_sizes_seen):.2f}", flush=True)
        print(f"[BENCH-C-{adaptive_window_ms}] Fill rate: 1/3={fill_1:.0f}% 2/3={fill_2:.0f}% 3/3={fill_3:.0f}%", flush=True)
        print(f"[BENCH-C-{adaptive_window_ms}] Latency stats (ms): "
              f"avg={np.mean(all_latencies_c):.1f} "
              f"p50={np.percentile(all_latencies_c, 50):.1f} "
              f"p95={np.percentile(all_latencies_c, 95):.1f} "
              f"max={np.max(all_latencies_c):.1f}", flush=True)

        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[BENCH-C-{adaptive_window_ms}] GPU mem: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)

        all_mode_c_results[adaptive_window_ms] = {
            "avg_latency": np.mean(all_latencies_c),
            "p50_latency": np.percentile(all_latencies_c, 50),
            "p95_latency": np.percentile(all_latencies_c, 95),
            "max_latency": np.max(all_latencies_c),
            "total_slices": slice_count_c,
            "throughput": slice_count_c / mode_c_elapsed,
            "avg_batch": np.mean(batch_sizes_seen),
            "fill_1": fill_1,
            "fill_2": fill_2,
            "fill_3": fill_3,
            "gpu_reserved": reserved,
        }

        torch.cuda.empty_cache()

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 80, flush=True)
    print("[BENCH] SUMMARY", flush=True)
    print("=" * 80, flush=True)

    avg_a = np.mean(all_latencies_a)

    # Sequential baseline
    print(f"\nBaseline: Sequential (3x run_pipeline, 142 SMs each)")
    print(f"  Avg latency: {avg_a:.1f}ms, P95: {np.percentile(all_latencies_a, 95):.1f}ms, "
          f"Throughput: {slice_count/mode_a_elapsed:.1f} slices/s", flush=True)

    # Combined comparison table: fixed + adaptive
    print(f"\n{'Mode':>15} | {'AvgLat':>10} | {'P50':>10} | {'P95':>10} | {'Max':>10} | {'AvgBatch':>8} | {'Fill1':>6} | {'Fill2':>6} | {'Fill3':>6} | {'Throughput':>10} | {'Speedup':>8}")
    print("-" * 130)

    best_mode = None
    best_speedup = 0

    for ww in WAIT_WINDOWS_MS:
        r = all_mode_b_results[ww]
        speedup = avg_a / r["avg_latency"] if r["avg_latency"] > 0 else 0
        if speedup > best_speedup:
            best_speedup = speedup
            best_mode = f"Fixed {ww}ms"
        print(f"{'Fixed ' + str(ww) + 'ms':>15} | {r['avg_latency']:>8.1f}ms | {r['p50_latency']:>8.1f}ms | {r['p95_latency']:>8.1f}ms | {r['max_latency']:>8.1f}ms | {r['avg_batch']:>8.2f} | {r['fill_1']:>5.0f}% | {r['fill_2']:>5.0f}% | {r['fill_3']:>5.0f}% | {r['throughput']:>8.1f}/s | {speedup:>7.2f}x")

    for aw in ADAPTIVE_WINDOWS_MS:
        r = all_mode_c_results[aw]
        speedup = avg_a / r["avg_latency"] if r["avg_latency"] > 0 else 0
        if speedup > best_speedup:
            best_speedup = speedup
            best_mode = f"Adaptive {aw}ms"
        print(f"{'Adapt ' + str(aw) + 'ms':>15} | {r['avg_latency']:>8.1f}ms | {r['p50_latency']:>8.1f}ms | {r['p95_latency']:>8.1f}ms | {r['max_latency']:>8.1f}ms | {r['avg_batch']:>8.2f} | {r['fill_1']:>5.0f}% | {r['fill_2']:>5.0f}% | {r['fill_3']:>5.0f}% | {r['throughput']:>8.1f}/s | {speedup:>7.2f}x")

    print(f"\n[BENCH] Best mode: {best_mode} ({best_speedup:.2f}x speedup over sequential)", flush=True)
    print(f"[BENCH] Sequential baseline: {avg_a:.1f}ms avg, {np.percentile(all_latencies_a, 95):.1f}ms p95", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("[BENCH] Benchmark complete", flush=True)
    print("=" * 80, flush=True)

    # Close log file and restore stdout
    sys.stdout.log_file.close()
    sys.stdout = sys.stdout.terminal

    # Read log content to return to local machine
    with open(LOG_FILE, "r") as f:
        log_content = f.read()

    return log_content


if __name__ == "__main__":
    log_content = run_streaming_benchmark.remote()
    local_log = os.path.join(os.path.dirname(__file__), "benchmark_results.log")
    with open(local_log, "w") as f:
        f.write(log_content)
    print(f"\n[LOCAL] Log saved to: {local_log}")
