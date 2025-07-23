import numpy as np
import torch
import torch.nn as nn
import random
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from collections import defaultdict


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target_net: nn.Module, policy_net: nn.Module, tau: float) -> None:
    """
    Soft update of target network parameters.
    θ_target = τ * θ_local + (1 - τ) * θ_target
    
    Args:
        target_net: Target network
        policy_net: Policy network
        tau: Interpolation parameter
    """
    for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def hard_update(target_net: nn.Module, policy_net: nn.Module) -> None:
    """
    Hard update of target network parameters.
    
    Args:
        target_net: Target network
        policy_net: Policy network
    """
    target_net.load_state_dict(policy_net.state_dict())


def save_model(model: nn.Module, filepath: str) -> None:
    """
    Save model state dict.
    
    Args:
        model: Model to save
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model(model: nn.Module, filepath: str, device: torch.device) -> None:
    """
    Load model state dict.
    
    Args:
        model: Model to load into
        filepath: Path to load from
        device: Device to load to
    """
    model.load_state_dict(torch.load(filepath, map_location=device))


class EpsilonScheduler:
    """Epsilon decay scheduler for epsilon-greedy exploration."""
    
    def __init__(self, start_epsilon: float = 1.0, end_epsilon: float = 0.01, 
                 decay_steps: int = 1000000):
        """
        Initialize epsilon scheduler.
        
        Args:
            start_epsilon: Initial epsilon value
            end_epsilon: Final epsilon value
            decay_steps: Number of steps to decay over
        """
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        if self.current_step >= self.decay_steps:
            return self.end_epsilon
        
        decay_ratio = self.current_step / self.decay_steps
        epsilon = self.start_epsilon + (self.end_epsilon - self.start_epsilon) * decay_ratio
        return epsilon
    
    def step(self) -> None:
        """Step the scheduler."""
        self.current_step += 1


class RunningMeanStd:
    """Running mean and standard deviation calculator."""
    
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """
        Initialize running statistics.
        
        Args:
            epsilon: Small value to avoid division by zero
            shape: Shape of the data
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update statistics with new batch."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, 
                           batch_count: int) -> None:
        """Update from batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self):
        """Initialize metrics logger."""
        self.metrics = defaultdict(list)
    
    def log(self, **kwargs) -> None:
        """Log metrics."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get all metrics."""
        return dict(self.metrics)
    
    def plot_metrics(self, metrics: List[str], save_path: Optional[str] = None) -> None:
        """
        Plot specified metrics.
        
        Args:
            metrics: List of metric names to plot
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in self.metrics:
                axes[i].plot(self.metrics[metric])
                axes[i].set_title(f'{metric}')
                axes[i].set_xlabel('Episode')
                axes[i].set_ylabel(metric)
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, dict(self.metrics))
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics from file."""
        loaded_metrics = np.load(filepath, allow_pickle=True).item()
        self.metrics.update(loaded_metrics)


def compute_gae(rewards: List[float], values: List[float], next_values: List[float],
                dones: List[bool], gamma: float = 0.99, lam: float = 0.95) -> List[float]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_values: List of next value estimates
        dones: List of done flags
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        List of advantages
    """
    advantages = []
    gae = 0
    
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = next_values[i] if not dones[i] else 0
        else:
            next_value = values[i + 1] if not dones[i] else 0
        
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * lam * gae * (1 - dones[i])
        advantages.insert(0, gae)
    
    return advantages


def compute_returns(rewards: List[float], dones: List[bool], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns.
    
    Args:
        rewards: List of rewards
        dones: List of done flags
        gamma: Discount factor
        
    Returns:
        List of returns
    """
    returns = []
    R = 0
    
    for i in reversed(range(len(rewards))):
        if dones[i]:
            R = 0
        R = rewards[i] + gamma * R
        returns.insert(0, R)
    
    return returns


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute explained variance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Explained variance
    """
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / var_y if var_y > 0 else 0


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial value
        final_value: Final value
        
    Returns:
        Schedule function
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        
        Args:
            progress_remaining: Remaining progress
            
        Returns:
            Current value
        """
        return progress_remaining * initial_value + (1 - progress_remaining) * final_value
    
    return func