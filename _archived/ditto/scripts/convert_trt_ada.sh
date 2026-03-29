#!/bin/bash
set -e

# ── Configuration ──
GPU_TYPE="NVIDIA GeForce RTX 4090"  # Use 4090 for Ada (L4) engines
POD_NAME="ditto-conversion-$(date +%s)"
NETWORK_VOLUME_ID="your-volume-id-here"  # From: runpodctl get volume
IMAGE="nvidia/cuda:12.1.0-devel-ubuntu22.04"

CONVERSION_SCRIPT='
set -e
apt-get update -qq && apt-get install -y -qq git python3-pip

# Clone Ditto repo
git clone https://github.com/antgroup/ditto-talkinghead /workspace/ditto-talkinghead
cd /workspace/ditto-talkinghead

# Install dependencies
pip install -q torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install -q onnxruntime-gpu tensorrt==8.6.1 numpy==2.0.1 cuda-python polygraphy colored tqdm

# Run conversion
python scripts/cvt_onnx_to_trt.py \
    --onnx_dir "/runpod-volume/models/ditto/ditto_onnx" \
    --trt_dir "/runpod-volume/models/ditto/ditto_trt_ada"

echo "Conversion complete! Engines saved."
'

echo "Creating pod: $POD_NAME"

# ── Step 1: Create pod ──
POD_OUTPUT=$(runpodctl create pod \
    --name "$POD_NAME" \
    --gpuType "$GPU_TYPE" \
    --gpuCount 1 \
    --secureCloud \
    --imageName "$IMAGE" \
    --containerDiskSize 20 \
    --networkVolumeId "$NETWORK_VOLUME_ID" \
    --ports "22/tcp")

# Extract pod ID
POD_ID=$(echo "$POD_OUTPUT" | grep -oE '[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}' | head -1)

if [ -z "$POD_ID" ]; then
    echo "Failed to extract pod ID. Raw output:"
    echo "$POD_OUTPUT"
    exit 1
fi

echo "Pod created: $POD_ID"
echo "Waiting for pod to be running..."

# ── Step 2: Wait for pod to be ready ──
for i in {1..30}; do
    STATUS=$(runpodctl get pod "$POD_ID" 2>/dev/null | grep -i "running" || echo "pending")
    if echo "$STATUS" | grep -iq "running"; then
        echo "Pod is running!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 10
done

# Get SSH info
POD_INFO=$(runpodctl get pod "$POD_ID")
SSH_HOST=$(echo "$POD_INFO" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)
SSH_PORT=$(echo "$POD_INFO" | grep -oE '22/tcp:[0-9]+' | cut -d: -f2)

echo "SSH ready: $SSH_HOST:$SSH_PORT"

# ── Step 3: Execute conversion via SSH ──
echo "Running conversion (~15-20 min)..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$SSH_PORT" root@"$SSH_HOST" "$CONVERSION_SCRIPT"

# ── Step 4: Stop and remove pod ──
echo "Conversion complete! Removing pod..."
runpodctl remove pod "$POD_ID"

echo "Done! TensorRT engines saved to /runpod-volume/models/ditto/ditto_trt_ada/"
echo "Total cost: ~\$0.10-0.15 for 20-30 minutes"
