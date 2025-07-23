"""Common utilities and components for reinforcement learning.

This module contains shared components used across different RL algorithms:
- Neural network architectures
- Replay buffers
- Environment wrappers
- Utility functions
- Logging utilities
"""

from .networks import (
    AtariCNN,
    DQNNetwork,
    DuelingDQNNetwork,
    ActorNetwork,
    CriticNetwork,
    ActorCriticNetwork,
    ContinuousActorNetwork,
    ContinuousCriticNetwork,
)
from .replay_buffer import ReplayBuffer
from .utils import get_device, soft_update, hard_update, EpsilonScheduler, compute_gae, compute_returns
from .env_wrappers import make_atari_env
from .logger import Logger
from .gif_wrapper import GifRecorderWrapper

__all__ = [
    # Networks
    "AtariCNN",
    "DQNNetwork", 
    "DuelingDQNNetwork",
    "ActorNetwork",
    "CriticNetwork",
    "ActorCriticNetwork",
    "ContinuousActorNetwork",
    "ContinuousCriticNetwork",
    # Replay buffer
    "ReplayBuffer",
    # Utils
    "get_device",
    "soft_update", 
    "hard_update",
    "EpsilonScheduler",
    "compute_gae",
    "compute_returns",
    # Env wrappers
    "make_atari_env",
    # Logging
    "Logger",
    "GifRecorderWrapper",
]