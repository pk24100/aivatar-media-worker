#!/usr/bin/env bash
# scripts/build_on_runpod.sh
# Builds the Docker image for AiVatar Media Worker with FlashHead Lite models
# Prerequisites: Run scripts/download_models.sh first to download wav2vec2 locally

set -e

IMAGE_NAME="pk24100/aivatar-worker"
IMAGE_TAG="flashhead-lite-v3"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=== Building ${FULL_IMAGE} ==="
echo ""

# Verify models exist
if [ ! -d "models/wav2vec2-base-960h" ]; then
    echo "ERROR: models/wav2vec2-base-960h not found!"
    echo "Please run: ./scripts/download_models.sh"
    exit 1
fi

echo "Models found:"
du -sh models/wav2vec2-base-960h
echo ""

# Build Docker Image
echo "Building Docker image..."
if command -v docker &> /dev/null; then
    docker build -t ${FULL_IMAGE} .
    echo ""
    echo "Pushing to Docker Hub..."
    docker push ${FULL_IMAGE}
elif command -v buildah &> /dev/null; then
    buildah bud -t ${FULL_IMAGE} .
    echo ""
    echo "Pushing to Docker Hub..."
    buildah push ${FULL_IMAGE} docker://docker.io/${FULL_IMAGE}
else
    echo "ERROR: Neither docker nor buildah found. Cannot build image."
    exit 1
fi

echo ""
echo "=== Build Complete ==="
echo ""
echo "Image pushed to: ${FULL_IMAGE}"
echo ""
echo "Next steps:"
echo "  1. Go to RunPod Console -> Serverless -> Edit Endpoint"
echo "  2. Set Image: ${FULL_IMAGE}"
echo "  3. Set cached model: pkam24100/aivatar-flashhead-model"
echo "  4. Enable FlashBoot, set workersMin=0"
echo "  5. Set env vars:"
echo "       LIVEKIT_URL=wss://your-livekit.cloud"
