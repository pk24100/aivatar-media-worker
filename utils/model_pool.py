import os
import sys
import asyncio
import logging

# Add SoulX-FlashHead to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "SoulX-FlashHead"))
from flash_head.inference import get_pipeline

logger = logging.getLogger(__name__)

# Manage a pool of FlashHead inference pipelines.
# All max_size pipelines are loaded at construction so they are captured
# in the CPU memory snapshot. On cold boot (snapshot restore), all
# pipelines are instantly available - zero load latency for sessions.
class FlashHeadModelPool:
    def __init__(self, max_size=3, ckpt_dir=None, wav2vec_dir=None):
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

        logger.info(f"Initializing FlashHead pool: max_size={max_size}")

        # Load all pipelines at construction (captured in CPU memory snapshot)
        for i in range(max_size):
            try:
                self._load_one()
            except Exception as e:
                logger.error(f"Failed to load pipeline {i+1}: {e}")
                raise

    def _load_one(self):
        """Load a single pipeline and add to pool (sync)."""
        world_size = 1
        logger.info(f"Loading pipeline {self.current_size + 1}/{self.max_size}...")
        pipeline = get_pipeline(world_size, self.ckpt_dir, self.model_type, self.wav2vec_dir)
        self.pool.put_nowait(pipeline)
        self.current_size += 1
        logger.info(f"Pipeline {self.current_size}/{self.max_size} loaded successfully.")

    async def acquire(self):
        """Acquire a pipeline from the pool.
        Blocks if pool is empty and at max_size (all pipelines in use).
        """
        async with self._lock:
            if self.pool.empty() and self.current_size < self.max_size:
                logger.info(f"Pool empty, loading pipeline {self.current_size + 1}/{self.max_size}")
                self._load_one()
        # Await pool.get() OUTSIDE the lock to avoid deadlock
        # when pool is empty and at max_size (all pipelines in use)
        logger.debug(f"Attempting to acquire pipeline. Available: {self.pool.qsize()}")
        pipeline = await self.pool.get()
        logger.debug(f"Pipeline acquired. Remaining: {self.pool.qsize()}")
        return pipeline

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
