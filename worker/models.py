"""FlashHead/Wav2Vec model resolution and the shared model pool."""

import logging
import os

from huggingface_hub import snapshot_download

from utils.model_pool import FlashHeadModelPool
from worker.config import (
    FLASHHEAD_HF_CACHE_DIR,
    FLASHHEAD_HF_REPO_ID,
    FLASHHEAD_HF_TOKEN,
    LEGACY_FLASHHEAD_CKPT_DIR,
    WAV2VEC_DIR,
    WORKER_POOL_SIZE,
)

# Keep the logger name "handler" so log output is identical to the monolith.
logger = logging.getLogger("handler")


def _is_valid_flashhead_ckpt_dir(path):
    return (
        bool(path)
        and os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "Model_Lite"))
        and os.path.isdir(os.path.join(path, "VAE_LTX"))
    )


def _resolve_flashhead_ckpt_dir():
    if _is_valid_flashhead_ckpt_dir(LEGACY_FLASHHEAD_CKPT_DIR):
        logger.info("Using baked FlashHead checkpoint directory: %s", LEGACY_FLASHHEAD_CKPT_DIR)
        return LEGACY_FLASHHEAD_CKPT_DIR

    if FLASHHEAD_HF_REPO_ID:
        try:
            resolved_dir = snapshot_download(
                repo_id=FLASHHEAD_HF_REPO_ID,
                cache_dir=FLASHHEAD_HF_CACHE_DIR,
                token=FLASHHEAD_HF_TOKEN,
                local_files_only=False,
            )
        except Exception as error:
            logger.warning(
                "Failed to resolve FlashHead checkpoint directory from Hugging Face "
                "repo %s: error=%s",
                FLASHHEAD_HF_REPO_ID,
                error.__class__.__name__,
            )
        else:
            if _is_valid_flashhead_ckpt_dir(resolved_dir):
                logger.info(
                    "Using Hugging Face FlashHead checkpoint snapshot from %s: %s",
                    FLASHHEAD_HF_REPO_ID,
                    resolved_dir,
                )
                return resolved_dir
            logger.warning(
                "Resolved Hugging Face snapshot is missing Model_Lite or VAE_LTX: %s",
                resolved_dir,
            )

    logger.info("Falling back to legacy FlashHead checkpoint directory: %s", LEGACY_FLASHHEAD_CKPT_DIR)
    return LEGACY_FLASHHEAD_CKPT_DIR


FLASHHEAD_CKPT_DIR = _resolve_flashhead_ckpt_dir()

# Modal's CPU snapshot only needs one immediately usable pipeline. The remaining
# capacity is loaded after restore while that first session can already stream.
INITIAL_PIPELINE_COUNT = (
    1
    if os.getenv("FLASHHEAD_LOAD_DEVICE", "").lower() == "cpu"
    else WORKER_POOL_SIZE
)
model_pool = FlashHeadModelPool(
    max_size=WORKER_POOL_SIZE,
    initial_size=INITIAL_PIPELINE_COUNT,
    ckpt_dir=FLASHHEAD_CKPT_DIR,
    wav2vec_dir=WAV2VEC_DIR,
)


# Verify FlashHead and Wav2Vec model directories exist.
def _verify_models():
    if not _is_valid_flashhead_ckpt_dir(FLASHHEAD_CKPT_DIR):
        print(f"Warning: FlashHead checkpoint dir not found at {FLASHHEAD_CKPT_DIR}")
    if not os.path.exists(WAV2VEC_DIR):
        print(f"Warning: Wav2Vec dir not found at {WAV2VEC_DIR}")

    return FLASHHEAD_CKPT_DIR


DATA_ROOT = _verify_models()
