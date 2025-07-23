import random
import numpy as np
from collections import deque
from typing import Tuple, List, Any


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    Used by DQN-based algorithms and off-policy methods.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of batched (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer with sum tree for efficient sampling.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Prioritization exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add transition with maximum priority."""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[np.ndarray, ...]:
        """
        Sample transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling exponent
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(weights, dtype=np.float32),
            indices
        )
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions."""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)