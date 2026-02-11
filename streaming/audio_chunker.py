import numpy as np


class AudioChunker:
    def __init__(self, sample_rate=16000, chunksize=(3, 5, 2)):
        self.sample_rate = sample_rate
        self.chunksize = chunksize
        self.chunk_step = int(chunksize[1] * 0.04 * sample_rate)
        self.chunk_total = int(sum(chunksize) * 0.04 * sample_rate) + 80
        self.left_pad = int(chunksize[0] * 0.04 * sample_rate)
        self.buffer = np.zeros((self.left_pad,), dtype=np.float32)
        self.cursor = 0

    def add_samples(self, samples):
        if samples is None or len(samples) == 0:
            return []
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self.buffer = np.concatenate([self.buffer, samples], axis=0)
        chunks = []
        while self.cursor + self.chunk_total <= len(self.buffer):
            chunk = self.buffer[self.cursor : self.cursor + self.chunk_total]
            chunks.append(chunk)
            self.cursor += self.chunk_step
        if self.cursor > self.chunk_total:
            self.buffer = self.buffer[self.cursor :]
            self.cursor = 0
        return chunks

    def flush(self):
        if self.cursor >= len(self.buffer):
            return []
        remaining = len(self.buffer) - self.cursor
        if remaining <= 0:
            return []
        pad = max(0, self.chunk_total - remaining)
        if pad:
            self.buffer = np.concatenate(
                [self.buffer, np.zeros((pad,), dtype=self.buffer.dtype)], axis=0
            )
        chunk = self.buffer[self.cursor : self.cursor + self.chunk_total]
        self.cursor = len(self.buffer)
        return [chunk]
