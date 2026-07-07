"""
DEPRECATED: FlashHead model weights are now baked into the Modal image at
build time via modal_app.py's .run_commands() step. This script is no
longer needed. Kept for reference only.

Original purpose: one-off Modal script to download FlashHead model weights
into the aivatar-models volume.

Usage (no longer needed):
    modal run download_models.py

After running, verify:
    modal volume ls aivatar-models /huggingface-cache/hub
"""

import os
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub")
)

app = modal.App("aivatar-download-models", image=image)


@app.function(
    volumes={"/models": modal.Volume.from_name("aivatar-models", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def download():
    from huggingface_hub import snapshot_download

    cache_dir = "/models/huggingface-cache/hub"
    os.makedirs(cache_dir, exist_ok=True)

    snapshot_download(
        repo_id="pkam24100/aivatar-flashhead-model",
        cache_dir=cache_dir,
        token=os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
    )

    print("FlashHead models downloaded successfully to", cache_dir)


@app.local_entrypoint()
def main():
    download.remote()
