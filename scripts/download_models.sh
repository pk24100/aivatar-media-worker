#!/usr/bin/env bash
# scripts/download_models.sh
# Downloads SoulX-FlashHead and Wav2Vec2 models locally for Docker build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=== Downloading AiVatar Models ==="
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Installing..."
    pip3 install -U "huggingface_hub[cli]"
fi

# Create models directory
mkdir -p models

# 1. Download FlashHead Lite model (~6.11GB)
echo "1. Downloading SoulX-FlashHead-1_3B (FlashHead Lite)..."
echo "   This is ~6.11GB, may take several minutes..."
mkdir -p models/SoulX-FlashHead-1_3B
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
    --local-dir ./models/SoulX-FlashHead-1_3B
echo "   Done!"

# 2. Download Wav2Vec2 model (~360MB)
echo ""
echo "2. Downloading wav2vec2-base-960h..."
mkdir -p models/wav2vec2-base-960h
huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir ./models/wav2vec2-base-960h
echo "   Done!"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Models downloaded to:"
du -sh ./models/SoulX-FlashHead-1_3B
du -sh ./models/wav2vec2-base-960h
echo ""
echo "You can now run: ./scripts/build_on_runpod.sh"
