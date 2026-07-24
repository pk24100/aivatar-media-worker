import os
import sys
import asyncio
import logging

# Add SoulX-FlashHead to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "SoulX-FlashHead"))
from flash_head.inference import get_pipeline

logger = logging.getLogger(__name__)

# Manage a pool of FlashHead inference pipelines.
class FlashHeadModelPool:
    def __init__(self, max_size, ckpt_dir=None, wav2vec_dir=None, initial_size=None):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        if initial_size is None:
            initial_size = max_size
        if not 1 <= initial_size <= max_size:
            raise ValueError("initial_size must be between 1 and max_size")

        self.max_size = max_size
        self.current_size = 0
        self.pool = asyncio.Queue(maxsize=max_size)
        self._lock = asyncio.Lock()

        self.ckpt_dir = ckpt_dir or os.getenv("FLASHHEAD_CKPT_DIR", "/app/models/SoulX-FlashHead-1_3B")
        self.wav2vec_dir = wav2vec_dir or os.getenv("WAV2VEC_DIR", "/app/models/wav2vec2-base-960h")
        self.model_type = "lite"

        # Verify models exist
        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"FlashHead checkpoint directory not found: {self.ckpt_dir}")
        if not os.path.exists(self.wav2vec_dir):
            raise FileNotFoundError(f"Wav2Vec directory not found: {self.wav2vec_dir}")

        logger.info(
            "Initializing FlashHead pool: initial_size=%s max_size=%s",
            initial_size,
            max_size,
        )

        # Modal loads one CPU pipeline into its memory snapshot. Other serving
        # environments retain eager pool initialization by using initial_size=max_size.
        for _ in range(initial_size):
            try:
                self._load_one()
            except Exception as e:
                logger.error(f"Failed to load pipeline {self.current_size + 1}: {e}")
                raise

    def _build_pipeline(self):
        """Build a single pipeline without mutating the asyncio-owned pool."""
        world_size = 1
        logger.info(f"Loading pipeline {self.current_size + 1}/{self.max_size}...")
        return get_pipeline(world_size, self.ckpt_dir, self.model_type, self.wav2vec_dir)

    def _add_pipeline(self, pipeline):
        self.pool.put_nowait(pipeline)
        self.current_size += 1
        logger.info(f"Pipeline {self.current_size}/{self.max_size} loaded successfully.")

    def _load_one(self):
        """Load one pipeline during synchronous eager initialization."""
        self._add_pipeline(self._build_pipeline())

    async def acquire(self):
        """Acquire a pipeline, waiting for background capacity when necessary."""
        logger.debug(f"Attempting to acquire pipeline. Available: {self.pool.qsize()}")
        pipeline = await self.pool.get()
        logger.debug(f"Pipeline acquired. Remaining: {self.pool.qsize()}")
        return pipeline

    async def load_remaining(self):
        """Load unsnapshotted capacity without blocking the aiohttp event loop."""
        loaded_count = 0
        while self.current_size < self.max_size:
            try:
                pipeline = await asyncio.to_thread(self._build_pipeline)
            except Exception:
                logger.exception(
                    "Failed to load background pipeline %s/%s",
                    self.current_size + 1,
                    self.max_size,
                )
                break

            async with self._lock:
                if self.current_size >= self.max_size:
                    break
                self._add_pipeline(pipeline)
                loaded_count += 1

        return loaded_count


    def release(self, pipeline):
        """Release a pipeline back to the pool."""
        try:
            self.pool.put_nowait(pipeline)
            logger.debug(f"Pipeline released. Available: {self.pool.qsize()}")
        except asyncio.QueueFull:
            logger.error("Attempted to release pipeline to full pool. This shouldn't happen.")

    def move_to_device(self, device):
        """Move all loaded pipelines to a new device (CPU snapshot restore)."""
        pipelines = []
        while not self.pool.empty():
            pipelines.append(self.pool.get_nowait())
        for p in pipelines:
            p.move_to_device(device)
            self.pool.put_nowait(p)
        logger.info(f"Moved {len(pipelines)} pipelines to {device}")

    # Return the number of currently available pipelines in the pool.
    def get_available_count(self):
        return self.pool.qsize()
