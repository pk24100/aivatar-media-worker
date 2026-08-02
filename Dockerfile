FROM nvcr.io/nvidia/pytorch:26.05-py3

# NGC PyTorch 26.05 ships PyTorch 2.12.0a0 with CUDA 13.2.1 and a matched torch + triton +
# transformer-engine. PyTorch 2.12 is required for GreenContext.Stream() API (green contexts).
# Do NOT override torch with the public wheel index (e.g. cu128) -- doing so installs a
# Triton that has no PTX table for CUDA 13.2 and breaks torch.compile() at runtime with
# "Triton only support CUDA 10.0 or higher, but got CUDA version: 13.2".

RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libsndfile1 wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# Install ninja first (build dep for flash-attn). Then install flash-attn against the
# image's bundled torch (no --index-url, no version pin). flash-attn install is best-
# effort: SoulX-FlashHead falls back to PyTorch SDPA if flash-attn isn't importable.
RUN pip3 install --upgrade pip && \
    pip3 install ninja && \
    (pip3 install 'flash-attn>=2.8.4' --no-build-isolation || \
     echo "flash-attn install failed; falling back to PyTorch SDPA at runtime") && \
    pip3 install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY entrypoint.sh /app/entrypoint.sh
COPY streaming /app/streaming
COPY utils /app/utils
COPY SoulX-FlashHead /app/SoulX-FlashHead

COPY models/wav2vec2-base-960h /app/models/wav2vec2-base-960h

RUN chmod +x /app/entrypoint.sh

CMD ["python3", "-u", "handler.py"]
