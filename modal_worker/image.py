"""Modal image builders shared by modal_app.py and modal_app_stress.py.

The SageAttention install command differs between production (prebuilt arch
list) and stress (patched per-GPU source build), so it is injected via
`sageattention_cmd`. All local paths are resolved absolutely from the repo
root so this module works regardless of the deployer's working directory.
"""

from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parent.parent

_BASE_ENV = {
    "UCX_TLS": "self",
    "UCX_NET_DEVICES": "none",
    "UCX_UNIFIED_TLS_MODE": "y",
    "UCX_MEMTYPE_CACHE": "n",
    "NCCL_DEBUG": "INFO",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}


def build_worker_image(sageattention_cmd, extra_pip=(), extra_env=None):
    """Build the GPU worker image.

    sageattention_cmd: shell command that installs SageAttention (differs
        between production and stress builds).
    extra_pip: additional pip packages beyond the shared base set.
    extra_env: additional environment variables merged over the base set.
    """
    env = dict(_BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return (
        modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.05-py3")
        .env(env)
        .apt_install("git", "git-lfs", "ffmpeg", "libsndfile1", "wget", "ca-certificates")
        .pip_install("ninja", "daily-python==0.31.0", *extra_pip)
        .run_commands(sageattention_cmd)
        .pip_install_from_requirements(str(_REPO_ROOT / "requirements.txt"))
        # Download FlashHead model weights from HuggingFace into the image layer.
        # This bakes ~6GB of weights directly into the image, eliminating the
        # ~38s snapshot_download from Modal Volume at every container boot.
        .run_commands(
            "huggingface-cli download pkam24100/aivatar-flashhead-model --local-dir /app/models/SoulX-FlashHead-1_3B",
            secrets=[modal.Secret.from_name("huggingface-secret")],
        )
        .add_local_dir(str(_REPO_ROOT / "streaming"), "/app/streaming", copy=True)
        .add_local_dir(str(_REPO_ROOT / "utils"), "/app/utils", copy=True)
        .add_local_dir(str(_REPO_ROOT / "config"), "/app/config", copy=True)
        .add_local_dir(str(_REPO_ROOT / "modal_worker"), "/app/modal_worker", copy=True)
        .add_local_dir(str(_REPO_ROOT / "SoulX-FlashHead"), "/app/SoulX-FlashHead", copy=True)
        .add_local_file(
            str(_REPO_ROOT / "SoulX-FlashHead/flash_head/src/modules/flash_head_model_snapshot_patch.py"),
            "/app/SoulX-FlashHead/flash_head/src/modules/flash_head_model.py",
            copy=True,
        )
        .add_local_dir(str(_REPO_ROOT / "models/wav2vec2-base-960h"), "/app/models/wav2vec2-base-960h", copy=True)
        .add_local_dir(str(_REPO_ROOT / "worker"), "/app/worker", copy=True)
        .add_local_file(str(_REPO_ROOT / "handler.py"), "/app/handler.py", copy=True)
        .add_local_file(str(_REPO_ROOT / "app_factory.py"), "/app/app_factory.py", copy=True)
    )


def build_autoscaler_image():
    """CPU-only image for the autoscaler HTTP endpoint (no GPU layers).

    Includes modal_worker/ because modal_app.py imports from it at module
    level (build_autoscaler_image, build_worker_image, run_batched_warmup,
    WorkerBase). Modal hydrates the autoscaler function by importing
    modal_app.py, so all top-level imports must resolve in this image too.
    """
    return (
        modal.Image.debian_slim()
        .pip_install("fastapi[standard]")
        .add_local_dir(str(_REPO_ROOT / "utils"), "/app/utils", copy=True)
        .add_local_dir(str(_REPO_ROOT / "modal_worker"), "/app/modal_worker", copy=True)
    )
