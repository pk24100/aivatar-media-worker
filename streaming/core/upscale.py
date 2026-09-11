# Shared output upscaling: every published frame passes through the SAME
# GPU function (bicubic + unsharp mask) so live frames, idle assets, stale
# idle assets and the static fallback are pixel-consistent across crossfades.
#
# The FlashHead model generates at its native trained resolution (512x512,
# see SoulX-FlashHead infer_params.yaml). This module upscales to the
# published output size (default 1024x1024) right before CPU transfer.
#
# Rollback: set AIVATAR_OUTPUT_WIDTH=512 and AIVATAR_OUTPUT_HEIGHT=512 to
# disable upscaling entirely (every path becomes a passthrough no-op).
import math
import os
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Bounds transient VRAM in upscale_frames_numpy: 32 frames x 1024x1024x3
# float32 is roughly 400MB peak per chunk.
_UPSCALE_CHUNK_FRAMES = 32

_cv2_fallback_logged = False


def _log_cv2_fallback_once(reason: str):
    global _cv2_fallback_logged
    if _cv2_fallback_logged:
        return
    _cv2_fallback_logged = True
    logger.warning(
        "Output upscaling using CPU cv2 fallback (INTER_CUBIC + unsharp) - "
        "GPU path unavailable: %s",
        reason,
    )


def get_output_size() -> tuple[int, int]:
    """Published output size. Setting both to 512 disables upscaling."""
    return (
        int(os.getenv("AIVATAR_OUTPUT_WIDTH", "1024")),
        int(os.getenv("AIVATAR_OUTPUT_HEIGHT", "1024")),
    )


def get_unsharp_params() -> tuple[float, float]:
    """Unsharp mask params: (amount, sigma). amount=0 disables sharpening."""
    return (
        float(os.getenv("AIVATAR_VIDEO_UNSHARP_AMOUNT", "0.5")),
        float(os.getenv("AIVATAR_VIDEO_UNSHARP_SIGMA", "1.0")),
    )


def _gaussian_kernel_2d(sigma: float, device, dtype) -> torch.Tensor:
    k = int(math.ceil(sigma * 3.0)) * 2 + 1
    x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


def upscale_slice_torch(
    video: torch.Tensor,
    out_w: int,
    out_h: int,
    amount: float | None = None,
    sigma: float | None = None,
) -> torch.Tensor:
    """GPU batch upscale + unsharp. video: (T, H, W, C) float32 in [0, 255].

    Returns (T, out_h, out_w, C) float32 clamped to [0, 255]. No-op when
    already at target. Works on CPU tensors too (tests, CUDA-less hosts).
    """
    if video.shape[1] == out_h and video.shape[2] == out_w:
        return video
    if amount is None or sigma is None:
        default_amount, default_sigma = get_unsharp_params()
        amount = default_amount if amount is None else amount
        sigma = default_sigma if sigma is None else sigma
    t, _, _, c = video.shape
    x = video.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
    x = F.interpolate(x, size=(out_h, out_w), mode="bicubic", align_corners=False)
    if amount > 0:
        k = _gaussian_kernel_2d(sigma, x.device, x.dtype)
        kernel = k.expand(c, 1, k.shape[0], k.shape[1])
        blurred = F.conv2d(x, kernel, padding=k.shape[0] // 2, groups=c)
        x = x + amount * (x - blurred)
    x = x.clamp(0.0, 255.0)
    return x.permute(0, 2, 3, 1).contiguous()


def upscale_frame_cv2(
    frame: np.ndarray,
    out_w: int,
    out_h: int,
    amount: float | None = None,
    sigma: float | None = None,
) -> np.ndarray:
    """DEFENSIVE CPU FALLBACK ONLY (used when CUDA is unavailable - logged).
    frame: (H, W, 3) uint8 RGB. No-op when already at target.
    Uses INTER_CUBIC (closest cv2 match to the GPU bicubic path) + same
    unsharp params, so CPU-fallback output stays close to the GPU path."""
    if frame.shape[0] == out_h and frame.shape[1] == out_w:
        return frame
    _log_cv2_fallback_once("torch/CUDA path unavailable")
    if amount is None or sigma is None:
        default_amount, default_sigma = get_unsharp_params()
        amount = default_amount if amount is None else amount
        sigma = default_sigma if sigma is None else sigma
    up = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    if amount > 0:
        blur = cv2.GaussianBlur(up, (0, 0), sigmaX=sigma)
        up = cv2.addWeighted(up, 1.0 + amount, blur, -amount, 0)  # addWeighted auto-clips
    return up


def upscale_frames_numpy(
    frames: list[np.ndarray], out_w: int, out_h: int
) -> list[np.ndarray]:
    """Batch numpy frames through upscale_slice_torch on GPU (chunked to
    bound VRAM). Used by IdleVideoLoop.normalize for stale 512 assets and by
    the static fallback path. Falls back to upscale_frame_cv2 per frame only
    if CUDA is unavailable or the GPU path fails (logged)."""
    if not frames:
        return []
    if frames[0].shape[0] == out_h and frames[0].shape[1] == out_w:
        return list(frames)
    if not torch.cuda.is_available():
        _log_cv2_fallback_once("CUDA unavailable")
        return [upscale_frame_cv2(f, out_w, out_h) for f in frames]
    try:
        results: list[np.ndarray] = []
        for start in range(0, len(frames), _UPSCALE_CHUNK_FRAMES):
            chunk = frames[start:start + _UPSCALE_CHUNK_FRAMES]
            batch = torch.from_numpy(
                np.stack([f.astype(np.float32) for f in chunk])
            ).cuda(non_blocking=True)
            batch = upscale_slice_torch(batch, out_w, out_h)
            results.extend(batch.cpu().numpy().astype(np.uint8))
        return results
    except Exception as exc:
        logger.warning("GPU frame upscale failed (%s); using cv2 fallback", exc)
        return [upscale_frame_cv2(f, out_w, out_h) for f in frames]
