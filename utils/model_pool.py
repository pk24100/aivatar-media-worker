import os
import sys
import asyncio
import logging

# Add SoulX-FlashHead to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "SoulX-FlashHead"))
from flash_head.inference import get_pipeline

logger = logging.getLogger(__name__)

# Manage a pool of pre-loaded FlashHead inference pipelines.
class FlashHeadModelPool:
    # Load and initialize the configured number of pipelines.
    def __init__(self, size=3, ckpt_dir=None, wav2vec_dir=None):
        self.size = size
        self.pool = asyncio.Queue(maxsize=size)
        
        self.ckpt_dir = ckpt_dir or os.getenv("FLASHHEAD_CKPT_DIR", "/app/models/SoulX-FlashHead-1_3B")
        self.wav2vec_dir = wav2vec_dir or os.getenv("WAV2VEC_DIR", "/app/models/wav2vec2-base-960h")
        self.model_type = "lite"
        
        # Verify models exist
        if not os.path.exists(self.ckpt_dir):
            raise FileNotFoundError(f"FlashHead checkpoint directory not found: {self.ckpt_dir}")
        if not os.path.exists(self.wav2vec_dir):
            raise FileNotFoundError(f"Wav2Vec directory not found: {self.wav2vec_dir}")
            
        logger.info(f"Initializing FlashHead pool of size {size}")

        # Initialize pool - each pipeline is standalone (avatar prepared per-session)
        for i in range(size):
            try:
                # Assuming single GPU setup
                world_size = 1
                logger.info(f"Loading pipeline {i+1}/{size}...")
                pipeline = get_pipeline(world_size, self.ckpt_dir, self.model_type, self.wav2vec_dir)
                
                # Store pipeline only (avatar prepared per-session)
                self.pool.put_nowait(pipeline)
                logger.info(f"Pipeline {i+1} loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load pipeline {i+1}: {e}")
                raise

    async def acquire(self):
        """Acquire a pipeline from the pool. Blocks if pool is empty."""
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
        """Move all pipelines to a new device (CPU snapshot restore)."""
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
