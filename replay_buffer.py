import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """经验回放存储"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.stack(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states),
                np.array(dones, dtype=np.bool_))

    def __len__(self):
        return len(self.buffer)