import os
import queue
import sys
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch


class QueueFrameWriter:
    def __init__(self, _video_path, fps=25, frame_queue=None, max_queue=200, **_kwargs):
        self.fps = fps
        self.frame_queue = frame_queue or queue.Queue(maxsize=max_queue)

    def __call__(self, img, fmt="bgr"):
        frame = img
        if fmt == "bgr":
            frame = frame[..., ::-1]
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def close(self):
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass


class DittoStreamingEngine:
    def __init__(
        self,
        model_root: str,
        source_path: str,
        cfg_path: Optional[str] = None,
        data_root: Optional[str] = None,
        repo_path: Optional[str] = None,
        fps: int = 25,
        queue_size: int = 200,
        chunksize: Tuple[int, int, int] = (3, 5, 2),
        model_instance: Optional[object] = None,
    ):
        self.model_root = model_root
        self.source_path = source_path
        self.cfg_path = cfg_path or self._resolve_cfg_path(model_root)
        self.data_root = data_root or self._resolve_data_root(model_root)
        self.repo_path = repo_path or os.getenv("DITTO_REPO_PATH", "/app/ditto-talkinghead")
        self.fps = fps
        self.chunksize = chunksize
        self.frame_queue = queue.Queue(maxsize=queue_size)

        stream_module = self._load_stream_module(self.repo_path)
        stream_module.VideoWriterByImageIO = lambda video_path, fps=25, **kwargs: QueueFrameWriter(
            video_path,
            fps=fps,
            frame_queue=self.frame_queue,
            max_queue=queue_size,
            **kwargs,
        )
        self._stream_module = stream_module
        
        if model_instance:
            self._sdk = model_instance
        else:
            self._sdk = stream_module.StreamSDK(self.cfg_path, self.data_root)
            
        output_path = tempfile.mktemp(suffix=".mp4")
        self._sdk.setup(
            source_path,
            output_path,
            online_mode=True,
            N_d=-1,
        )
        self._sdk.setup_Nd(N_d=-1)

    def run_chunk(self, audio_chunk: np.ndarray):
        self._sdk.run_chunk(audio_chunk, chunksize=self.chunksize)

    def close(self):
        self._sdk.close()

    @staticmethod
    def _load_stream_module(repo_path: str):
        if not repo_path:
            raise ValueError("DITTO_REPO_PATH is required to load stream_pipeline_online.py")
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            import stream_pipeline_online as stream_module
        except Exception as exc:
            raise ImportError(
                "Unable to import stream_pipeline_online from Ditto repository. "
                "Ensure DITTO_REPO_PATH points to the ditto-talkinghead repo."
            ) from exc
        return stream_module

    @staticmethod
    def _resolve_cfg_path(model_root: str) -> str:
        online_cfg = os.path.join(model_root, "ditto_cfg", "v0.4_hubert_cfg_trt_online.pkl")
        if os.path.isfile(online_cfg):
            return online_cfg
        fallback_cfg = os.path.join(model_root, "ditto_cfg", "v0.4_hubert_cfg_trt.pkl")
        if os.path.isfile(fallback_cfg):
            return fallback_cfg
        raise FileNotFoundError(
            "Ditto config not found. Expected v0.4_hubert_cfg_trt_online.pkl in ditto_cfg."
        )

    @staticmethod
    def _resolve_data_root(model_root: str) -> str:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) >= (8, 9):
                preferred = "ditto_trt_ada"
            else:
                preferred = "ditto_trt_Ampere_Plus"
            preferred_path = os.path.join(model_root, preferred)
            if os.path.isdir(preferred_path) and len(os.listdir(preferred_path)) > 0:
                return preferred_path

        candidates = [
            "ditto_trt_Ampere_Plus",
            "ditto_trt_ada",
            "ditto_trt_3090",
            "ditto_trt_custom",
            "ditto_onnx",
            "ditto_pytorch",
        ]
        for candidate in candidates:
            path = os.path.join(model_root, candidate)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError("Ditto model directory not found under model_root.")
