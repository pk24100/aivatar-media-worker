import asyncio
import queue

class DittoModelPool:
    def __init__(self, pool_size=3, model_root="/app/models/ditto"):
        self.pool_size = pool_size
        self.model_root = model_root
        self.pool = queue.Queue(maxsize=pool_size)
        
        # We don't initialize the stream engines completely here because 
        # StreamSDK requires a specific `source_image` and `output_path` during its `.setup()` call.
        # However, we can preload the heavy neural network weights into VRAM by initializing 
        # the base StreamSDK classes without calling setup yet.
        self.initialize_pool()

    def initialize_pool(self):
        """
        Preload multiple independent instances of the Ditto model into VRAM.
        Note: The actual `setup()` must be called per-session since it binds to a specific source image.
        """
        import os
        import sys
        
        repo_path = os.getenv("DITTO_REPO_PATH", "/app/ditto-talkinghead")
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
            
        try:
            import stream_pipeline_online as stream_module
        except ImportError as e:
            print(f"Warning: Could not import stream_pipeline_online: {e}")
            return

        cfg_path = self._resolve_cfg_path()
        data_root = self._resolve_data_root()

        for i in range(self.pool_size):
            try:
                # Instantiate the base SDK, which loads the TRT engines/weights into VRAM
                sdk_instance = stream_module.StreamSDK(cfg_path, data_root)
                self.pool.put(sdk_instance)
                print(f"Initialized Ditto pool instance {i+1}/{self.pool_size}")
            except Exception as e:
                print(f"Failed to initialize Ditto instance {i}: {e}")

    async def acquire(self):
        """Async method to get a free model instance from the pool."""
        while self.pool.empty():
            await asyncio.sleep(0.1)
        return self.pool.get()

    def release(self, model_instance):
        """Return the model instance to the pool."""
        # Optional: Add any cleanup/reset logic here before returning to pool
        self.pool.put(model_instance)

    def _resolve_cfg_path(self) -> str:
        import os
        online_cfg = os.path.join(self.model_root, "ditto_cfg", "v0.4_hubert_cfg_trt_online.pkl")
        if os.path.isfile(online_cfg):
            return online_cfg
        fallback_cfg = os.path.join(self.model_root, "ditto_cfg", "v0.4_hubert_cfg_trt.pkl")
        if os.path.isfile(fallback_cfg):
            return fallback_cfg
        raise FileNotFoundError("Ditto config not found.")

    def _resolve_data_root(self) -> str:
        import os
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) >= (8, 9):
                preferred = "ditto_trt_ada"
            else:
                preferred = "ditto_trt_Ampere_Plus"
            preferred_path = os.path.join(self.model_root, preferred)
            if os.path.isdir(preferred_path) and len(os.listdir(preferred_path)) > 0:
                return preferred_path

        candidates = ["ditto_trt_Ampere_Plus", "ditto_trt_ada", "ditto_trt_3090", "ditto_trt_custom", "ditto_onnx", "ditto_pytorch"]
        for candidate in candidates:
            path = os.path.join(self.model_root, candidate)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError("Ditto model directory not found.")
