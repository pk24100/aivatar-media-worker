FROM nvcr.io/nvidia/pytorch:26.02-py3

RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libsndfile1 wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install ninja && \
    pip3 install flash_attn==2.8.0.post2 --no-build-isolation && \
    pip3 install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY entrypoint.sh /app/entrypoint.sh
COPY streaming /app/streaming
COPY utils /app/utils
COPY SoulX-FlashHead /app/SoulX-FlashHead

COPY models/SoulX-FlashHead-1_3B /app/models/SoulX-FlashHead-1_3B
COPY models/wav2vec2-base-960h /app/models/wav2vec2-base-960h

RUN chmod +x /app/entrypoint.sh

CMD ["python3", "-u", "handler.py"]
