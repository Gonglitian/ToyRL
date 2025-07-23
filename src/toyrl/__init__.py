"""ToyRL: A comprehensive implementation of classic deep reinforcement learning algorithms.

This package provides clean, modular implementations of fundamental RL algorithms 
including DQN, SAC, PPO, A2C, and more, with shared components for easy 
experimentation and comparison.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("toyrl")
except importlib.metadata.PackageNotFoundError:
    # For development installations
    __version__ = "0.1.0.dev0"

# Lazy imports to avoid import errors during build
def __getattr__(name):
    """Lazy import for module attributes."""
    if name == "DQN":
        from .algorithms.dqn import DQN
        return DQN
    elif name == "DoubleDQN":
        from .algorithms.dqn import DoubleDQN
        return DoubleDQN
    elif name == "DuelingDQN":
        from .algorithms.dqn import DuelingDQN
        return DuelingDQN
    elif name == "SAC":
        from .algorithms.sac import SAC
        return SAC
    elif name == "REINFORCE":
        from .algorithms.policy_gradient import REINFORCE
        return REINFORCE
    elif name == "ActorCritic":
        from .algorithms.policy_gradient import ActorCritic
        return ActorCritic
    elif name == "A2C":
        from .algorithms.policy_gradient import A2C
        return A2C
    elif name == "A3C":
        from .algorithms.advanced_pg import A3C
        return A3C
    elif name == "PPO":
        from .algorithms.advanced_pg import PPO
        return PPO
    elif name == "TRPO":
        from .algorithms.advanced_pg import TRPO
        return TRPO
    elif name == "AtariCNN":
        from .common.networks import AtariCNN
        return AtariCNN
    elif name == "DQNNetwork":
        from .common.networks import DQNNetwork
        return DQNNetwork
    elif name == "DuelingDQNNetwork":
        from .common.networks import DuelingDQNNetwork
        return DuelingDQNNetwork
    elif name == "ContinuousActorNetwork":
        from .common.networks import ContinuousActorNetwork
        return ContinuousActorNetwork
    elif name == "ContinuousCriticNetwork":
        from .common.networks import ContinuousCriticNetwork
        return ContinuousCriticNetwork
    elif name == "ReplayBuffer":
        from .common.replay_buffer import ReplayBuffer
        return ReplayBuffer
    elif name == "get_device":
        from .common.utils import get_device
        return get_device
    elif name == "soft_update":
        from .common.utils import soft_update
        return soft_update
    elif name == "hard_update":
        from .common.utils import hard_update
        return hard_update
    elif name == "make_atari_env":
        from .common.env_wrappers import make_atari_env
        return make_atari_env
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Version
    "__version__",
    # Algorithms
    "DQN",
    "DoubleDQN", 
    "DuelingDQN",
    "SAC",
    "REINFORCE",
    "ActorCritic",
    "A2C",
    "A3C",
    "PPO",
    "TRPO",
    # Networks
    "AtariCNN",
    "DQNNetwork",
    "DuelingDQNNetwork",
    "ContinuousActorNetwork",
    "ContinuousCriticNetwork",
    # Utilities
    "ReplayBuffer",
    "get_device",
    "soft_update",
    "hard_update",
    "make_atari_env",
]