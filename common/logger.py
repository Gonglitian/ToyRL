"""
TensorBoard Logger for ToyRL
Extends torch.utils.tensorboard.SummaryWriter with additional functionality.
"""

import os
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class Logger(SummaryWriter):
    """
    Enhanced TensorBoard logger that extends SummaryWriter.
    
    Provides additional functionality for RL logging including:
    - Scalar logging with automatic step tracking
    - Histogram logging for weights and gradients
    - Model graph logging
    - Episode-based and step-based logging
    """
    
    def __init__(self, log_dir: str, comment: str = "", purge_step: Optional[int] = None,
                 max_queue: int = 10, flush_secs: int = 120, filename_suffix: str = ""):
        """
        Initialize the Logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            comment: Comment to append to the default log directory name
            purge_step: Step to purge logs from
            max_queue: Size of the queue for pending events
            flush_secs: How often to flush pending events
            filename_suffix: Suffix for the log file
        """
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize parent SummaryWriter
        super().__init__(
            log_dir=log_dir,
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix
        )
        
        # Track steps and episodes
        self.global_step = 0
        self.episode_count = 0
        
        # Store metrics for backward compatibility
        self.metrics = defaultdict(list)
        
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value to TensorBoard.
        
        Args:
            name: Name of the scalar
            value: Value to log
            step: Step number (uses global_step if None)
        """
        if step is None:
            step = self.global_step
            
        self.add_scalar(name, value, step)
        
        # Store for backward compatibility
        self.metrics[name].append(value)
        
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray, List], 
                     step: Optional[int] = None):
        """
        Log a histogram of values to TensorBoard.
        
        Args:
            name: Name of the histogram
            values: Values to log (tensor, array, or list)
            step: Step number (uses global_step if None)
        """
        if step is None:
            step = self.global_step
            
        # Convert to tensor if needed
        if isinstance(values, (list, np.ndarray)):
            values = torch.tensor(values, dtype=torch.float32)
        elif not isinstance(values, torch.Tensor):
            values = torch.tensor([values], dtype=torch.float32)
            
        self.add_histogram(name, values, step)
        
    def log_image(self, name: str, image: Union[torch.Tensor, np.ndarray], 
                  step: Optional[int] = None):
        """
        Log an image to TensorBoard.
        
        Args:
            name: Name of the image
            image: Image tensor or array
            step: Step number (uses global_step if None)
        """
        if step is None:
            step = self.global_step
            
        self.add_image(name, image, step)
        
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor):
        """
        Log the model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_sample: Sample input tensor for the model
        """
        try:
            self.add_graph(model, input_sample)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
            
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        Log hyperparameters and metrics to TensorBoard.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of metric values
        """
        self.add_hparams(hparams, metrics)
        
    def log_episode_reward(self, reward: float, episode: Optional[int] = None):
        """
        Log episode reward.
        
        Args:
            reward: Episode reward
            episode: Episode number (uses episode_count if None)
        """
        if episode is None:
            episode = self.episode_count
            
        self.log_scalar("Episode/Reward", reward, episode)
        self.episode_count = max(self.episode_count, episode + 1)
        
    def log_episode_length(self, length: int, episode: Optional[int] = None):
        """
        Log episode length.
        
        Args:
            length: Episode length
            episode: Episode number (uses episode_count if None)
        """
        if episode is None:
            episode = self.episode_count
            
        self.log_scalar("Episode/Length", length, episode)
        
    def log_loss(self, loss: float, step: Optional[int] = None):
        """
        Log training loss.
        
        Args:
            loss: Loss value
            step: Step number (uses global_step if None)
        """
        self.log_scalar("Training/Loss", loss, step)
        
    def log_learning_rate(self, lr: float, step: Optional[int] = None):
        """
        Log learning rate.
        
        Args:
            lr: Learning rate
            step: Step number (uses global_step if None)
        """
        self.log_scalar("Training/LearningRate", lr, step)
        
    def log_exploration_rate(self, epsilon: float, step: Optional[int] = None):
        """
        Log exploration rate (epsilon).
        
        Args:
            epsilon: Exploration rate
            step: Step number (uses global_step if None)
        """
        self.log_scalar("Training/ExplorationRate", epsilon, step)
        
    def log_network_weights(self, model: torch.nn.Module, step: Optional[int] = None):
        """
        Log network weights as histograms.
        
        Args:
            model: PyTorch model
            step: Step number (uses global_step if None)
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f"Weights/{name}", param.data, step)
                if param.grad is not None:
                    self.log_histogram(f"Gradients/{name}", param.grad.data, step)
                    
    def log_evaluation_results(self, mean_reward: float, std_reward: float, 
                              episode: Optional[int] = None):
        """
        Log evaluation results.
        
        Args:
            mean_reward: Mean evaluation reward
            std_reward: Standard deviation of evaluation rewards
            episode: Episode number (uses episode_count if None)
        """
        if episode is None:
            episode = self.episode_count
            
        self.log_scalar("Evaluation/MeanReward", mean_reward, episode)
        self.log_scalar("Evaluation/StdReward", std_reward, episode)
        
    def step(self):
        """Increment global step counter."""
        self.global_step += 1
        
    def episode_step(self):
        """Increment episode counter."""
        self.episode_count += 1
        
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get all metrics (for backward compatibility).
        
        Returns:
            Dictionary of metrics
        """
        return dict(self.metrics)
        
    def close(self):
        """Close the logger and flush all pending events."""
        super().close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()