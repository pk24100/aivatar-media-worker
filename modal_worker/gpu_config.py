"""GPU configuration shared by modal_app.py and modal_app_stress.py.

All GPU-specific settings (Modal gpu string, SageAttention CUDA arch, video
codec, NVENC compatibility) are driven from this dict. Add new GPUs here as
needed.
"""

GPU_CONFIG = {
    "L4": {
        "modal_gpu": "L4",
        "cuda_arch": "8.9",
        "video_codec": "h264_hw",
        "nvenc_ok": True,
        "notes": "Ada sm89. NVENC works with LiveKit.",
    },
    "L40S": {
        "modal_gpu": "L40S",
        "cuda_arch": "8.9",
        "video_codec": "h264_hw",
        "nvenc_ok": True,
        "notes": "Ada sm89. NVENC works with LiveKit.",
    },
    "H100": {
        "modal_gpu": "H100",
        "cuda_arch": "9.0",
        "video_codec": "h264_hw",
        "nvenc_ok": True,
        "notes": "Hopper sm90. NVENC 7th-gen works with LiveKit.",
    },
    "H200": {
        "modal_gpu": "H200",
        "cuda_arch": "9.0",
        "video_codec": "h264_hw",
        "nvenc_ok": True,
        "notes": "Hopper sm90. Same compute as H100, more VRAM (141GB). NVENC 7th-gen works with LiveKit.",
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
    "RTX-PRO-6000": {
        "modal_gpu": "RTX-PRO-6000",
        "cuda_arch": "12.0",
        "video_codec": "vp8",
        "nvenc_ok": False,
        "notes": "Blackwell sm120. NVENC 9th-gen incompatible with LiveKit, use VP8 software.",
    },
    "B200": {
        "modal_gpu": "B200",
        "cuda_arch": "10.0",
        "video_codec": "vp8",
        "nvenc_ok": False,
        "notes": "Blackwell GB100 sm100. Same arch family as RTX PRO 6000 (sm120). Use VP8 software - NVENC compatibility unverified with LiveKit.",
    },
    "B300": {
        "modal_gpu": "B300",
        "cuda_arch": "10.3",
        "video_codec": "vp8",
        "nvenc_ok": False,
        "notes": "Blackwell Ultra GB300 sm103. Build natively with sm103a (compute_100a PTX is NOT forward-compatible to sm103 due to 'a' suffix per NVIDIA Blackwell Compatibility Guide). Most SMs (~240). Use VP8 software.",
    },
}


def resolve_gpu(target_gpu):
    """Return the GPU_CONFIG entry for target_gpu, raising on unknown names."""
    if target_gpu not in GPU_CONFIG:
        raise ValueError(
            f"Unknown TARGET_GPU='{target_gpu}'. Valid options: {list(GPU_CONFIG.keys())}"
        )
    return GPU_CONFIG[target_gpu]
