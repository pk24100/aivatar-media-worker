#!/usr/bin/env bash
# scripts/download_models.sh
# Downloads SoulX-FlashHead and Wav2Vec2 models locally for Docker build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=== Downloading AiVatar Models ==="
echo ""

# Check if hf CLI is installed
if ! command -v hf &> /dev/null; then
    echo "hf CLI not found. Installing..."
    pip3 install -U "huggingface_hub[cli]"
fi

# Create models directory
mkdir -p models

DOWNLOAD_FLASHHEAD="${DOWNLOAD_FLASHHEAD:-0}"
FLASHHEAD_SOURCE_REPO_ID="${FLASHHEAD_SOURCE_REPO_ID:-Soul-AILab/SoulX-FlashHead-1_3B}"

if [ "${DOWNLOAD_FLASHHEAD}" = "1" ] || [ "${DOWNLOAD_FLASHHEAD}" = "true" ] || [ "${DOWNLOAD_FLASHHEAD}" = "yes" ]; then
    echo "1. Downloading SoulX-FlashHead-1_3B (FlashHead Lite)..."
    echo "   This is ~6.11GB, may take several minutes..."
    mkdir -p models/SoulX-FlashHead-1_3B
    hf download "${FLASHHEAD_SOURCE_REPO_ID}" \
        --local-dir ./models/SoulX-FlashHead-1_3B
    echo "   Done!"
else
    echo "1. Skipping SoulX-FlashHead-1_3B download (RunPod cached-model flow)."
    echo "   Set DOWNLOAD_FLASHHEAD=1 to download a local fallback checkpoint copy."
fi

# 2. Download Wav2Vec2 model (~360MB)
echo ""
echo "2. Downloading wav2vec2-base-960h..."
mkdir -p models/wav2vec2-base-960h
hf download facebook/wav2vec2-base-960h \
    --local-dir ./models/wav2vec2-base-960h
echo "   Done!"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Models downloaded to:"
if [ -d ./models/SoulX-FlashHead-1_3B ]; then
    du -sh ./models/SoulX-FlashHead-1_3B
fi
du -sh ./models/wav2vec2-base-960h
echo ""
echo "You can now run: ./scripts/build_on_runpod.sh"
