# RL pool, using deque
from collections import deque
import numpy as np


class Pool:
    def __init__(self, max_size=10_000, batch_size=32):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=self.batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)
