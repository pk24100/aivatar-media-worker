FROM nvcr.io/nvidia/tensorrt:23.08-py3

RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libsndfile1 wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN chmod +x /app/entrypoint.sh

# Bake models into the image (copied during build on RunPod pod)
COPY models/ditto /app/models/ditto
COPY ditto-talkinghead /app/ditto-talkinghead

CMD ["python3", "-u", "handler.py"]
