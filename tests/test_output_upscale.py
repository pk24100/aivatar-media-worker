"""Tests for the shared output upscaling module (streaming/output_upscale.py).

All tests run on CPU tensors - no GPU required. The cv2 fallback path is
exercised naturally on CUDA-less hosts.
"""
import numpy as np
import pytest
import torch

from streaming.core.upscale import (
    get_output_size,
    get_unsharp_params,
    upscale_frame_cv2,
    upscale_frames_numpy,
    upscale_slice_torch,
)


def _gradient_image(height: int, width: int) -> np.ndarray:
    """Smooth RGB gradient with a few sharp edges (exercises bicubic + unsharp)."""
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    frame = np.add.outer(y, x) / 2.0
    frame[height // 4: height // 2, width // 4: width // 2] = 255.0
    return np.stack([frame] * 3, axis=2).astype(np.uint8)


def test_get_output_size_defaults(monkeypatch):
    monkeypatch.delenv("AIVATAR_OUTPUT_WIDTH", raising=False)
    monkeypatch.delenv("AIVATAR_OUTPUT_HEIGHT", raising=False)
    assert get_output_size() == (1024, 1024)


def test_get_output_size_env_override(monkeypatch):
    monkeypatch.setenv("AIVATAR_OUTPUT_WIDTH", "512")
    monkeypatch.setenv("AIVATAR_OUTPUT_HEIGHT", "768")
    assert get_output_size() == (512, 768)


def test_get_unsharp_params_defaults_and_env(monkeypatch):
    monkeypatch.delenv("AIVATAR_VIDEO_UNSHARP_AMOUNT", raising=False)
    monkeypatch.delenv("AIVATAR_VIDEO_UNSHARP_SIGMA", raising=False)
    assert get_unsharp_params() == (0.5, 1.0)
    monkeypatch.setenv("AIVATAR_VIDEO_UNSHARP_AMOUNT", "0.0")
    monkeypatch.setenv("AIVATAR_VIDEO_UNSHARP_SIGMA", "2.5")
    assert get_unsharp_params() == (0.0, 2.5)


def test_upscale_slice_torch_upscales_batch():
    video = torch.rand(4, 8, 8, 3, dtype=torch.float32) * 255.0
    out = upscale_slice_torch(video, 16, 16)
    assert out.shape == (4, 16, 16, 3)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 255.0


def test_upscale_slice_torch_noop_at_target():
    video = torch.rand(2, 16, 16, 3, dtype=torch.float32) * 255.0
    out = upscale_slice_torch(video, 16, 16)
    assert out is video


def test_upscale_slice_torch_cpu_tensor():
    video = torch.rand(1, 8, 8, 3, dtype=torch.float32) * 255.0
    out = upscale_slice_torch(video, 32, 32)
    assert out.shape == (1, 32, 32, 3)
    assert out.device == video.device


def test_upscale_slice_torch_amount_zero_differs_from_default():
    video = torch.rand(1, 8, 8, 3, dtype=torch.float32) * 255.0
    bicubic_only = upscale_slice_torch(video, 16, 16, amount=0.0)
    sharpened = upscale_slice_torch(video, 16, 16, amount=0.5)
    assert not torch.allclose(bicubic_only, sharpened)


def test_upscale_slice_torch_clamps_overshoot():
    # Single bright pixel on black: bicubic ringing pushes values outside
    # [0, 255] before the clamp.
    video = torch.zeros(1, 8, 8, 3, dtype=torch.float32)
    video[0, 4, 4, :] = 255.0
    out = upscale_slice_torch(video, 16, 16, amount=1.5)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 255.0


def test_upscale_frames_numpy_batch():
    frames = [_gradient_image(8, 8) for _ in range(3)]
    out = upscale_frames_numpy(frames, 16, 16)
    assert len(out) == 3
    for frame in out:
        assert frame.shape == (16, 16, 3)
        assert frame.dtype == np.uint8


def test_upscale_frames_numpy_noop_at_target():
    frames = [_gradient_image(16, 16) for _ in range(2)]
    out = upscale_frames_numpy(frames, 16, 16)
    assert out == frames


def test_upscale_frame_cv2_upscales():
    frame = _gradient_image(8, 8)
    out = upscale_frame_cv2(frame, 16, 16)
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def test_upscale_frame_cv2_noop_at_target():
    frame = _gradient_image(16, 16)
    out = upscale_frame_cv2(frame, 16, 16)
    assert out is frame


def test_torch_and_cv2_paths_stay_visually_close():
    """Guards against the GPU path and the CPU fallback drifting apart."""
    frame = _gradient_image(32, 32)
    tensor = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0)
    torch_out = upscale_slice_torch(tensor, 64, 64).squeeze(0).numpy()
    cv2_out = upscale_frame_cv2(frame, 64, 64).astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(torch_out - cv2_out)))
    assert mean_abs_diff < 10.0, f"paths diverged: mean abs diff {mean_abs_diff:.2f}"


def test_idle_video_loop_normalize_upscales_stale_frames():
    from streaming.orchestration.idle_video import IdleVideoLoop

    loop = IdleVideoLoop()
    loop.frames = [np.zeros((512, 512, 3), dtype=np.uint8)]
    loop.total_frames = 1
    loop.normalize(1024, 1024)
    assert loop.frames[0].shape == (1024, 1024, 3)


def test_idle_video_loop_normalize_noop_at_target():
    from streaming.orchestration.idle_video import IdleVideoLoop

    loop = IdleVideoLoop()
    frame = _gradient_image(1024, 1024)
    loop.frames = [frame]
    loop.total_frames = 1
    loop.normalize(1024, 1024)
    assert loop.frames[0] is frame
