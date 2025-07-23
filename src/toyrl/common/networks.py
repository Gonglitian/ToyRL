import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class AtariCNN(nn.Module):
    """
    Convolutional Neural Network for Atari games.
    Used as feature extractor for DQN-based algorithms.
    """
    
    def __init__(self, input_channels: int = 4, feature_dim: int = 512):
        """
        Initialize Atari CNN.
        
        Args:
            input_channels: Number of input channels (stacked frames)
            feature_dim: Dimension of output features
        """
        super(AtariCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions
        conv_output_size = self._get_conv_output_size(input_channels)
        
        self.fc = nn.Linear(conv_output_size, feature_dim)
    
    def _get_conv_output_size(self, input_channels: int) -> int:
        """Calculate output size after convolution layers."""
        x = torch.zeros(1, input_channels, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class DQNNetwork(nn.Module):
    """Deep Q-Network for value-based methods."""
    
    def __init__(self, input_channels: int, num_actions: int, feature_dim: int = 512):
        """
        Initialize DQN network.
        
        Args:
            input_channels: Number of input channels
            num_actions: Number of possible actions
            feature_dim: Dimension of hidden features
        """
        super(DQNNetwork, self).__init__()
        
        self.feature_extractor = AtariCNN(input_channels, feature_dim)
        self.value_head = nn.Linear(feature_dim, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-values."""
        features = self.feature_extractor(x)
        q_values = self.value_head(features)
        return q_values


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN with separate value and advantage streams."""
    
    def __init__(self, input_channels: int, num_actions: int, feature_dim: int = 512):
        """
        Initialize Dueling DQN network.
        
        Args:
            input_channels: Number of input channels
            num_actions: Number of possible actions
            feature_dim: Dimension of hidden features
        """
        super(DuelingDQNNetwork, self).__init__()
        
        self.feature_extractor = AtariCNN(input_channels, feature_dim)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining value and advantage."""
        features = self.feature_extractor(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class ActorNetwork(nn.Module):
    """Actor network for policy-based methods."""
    
    def __init__(self, input_channels: int, num_actions: int, feature_dim: int = 512):
        """
        Initialize Actor network.
        
        Args:
            input_channels: Number of input channels
            num_actions: Number of possible actions
            feature_dim: Dimension of hidden features
        """
        super(ActorNetwork, self).__init__()
        
        self.feature_extractor = AtariCNN(input_channels, feature_dim)
        self.policy_head = nn.Linear(feature_dim, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action probabilities."""
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        return F.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, input_channels: int, feature_dim: int = 512):
        """
        Initialize Critic network.
        
        Args:
            input_channels: Number of input channels
            feature_dim: Dimension of hidden features
        """
        super(CriticNetwork, self).__init__()
        
        self.feature_extractor = AtariCNN(input_channels, feature_dim)
        self.value_head = nn.Linear(feature_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get state value."""
        features = self.feature_extractor(x)
        value = self.value_head(features)
        return value


class ActorCriticNetwork(nn.Module):
    """Combined Actor-Critic network sharing feature extractor."""
    
    def __init__(self, input_channels: int, num_actions: int, feature_dim: int = 512):
        """
        Initialize Actor-Critic network.
        
        Args:
            input_channels: Number of input channels
            num_actions: Number of possible actions
            feature_dim: Dimension of hidden features
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.feature_extractor = AtariCNN(input_channels, feature_dim)
        self.policy_head = nn.Linear(feature_dim, num_actions)
        self.value_head = nn.Linear(feature_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get both policy and value."""
        features = self.feature_extractor(x)
        
        policy_logits = self.policy_head(features)
        policy = F.softmax(policy_logits, dim=-1)
        
        value = self.value_head(features)
        
        return policy, value


class ContinuousActorNetwork(nn.Module):
    """Actor network for continuous action spaces (SAC)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize continuous actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(ContinuousActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get action mean and log standard deviation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std


class ContinuousCriticNetwork(nn.Module):
    """Critic network for continuous action spaces (SAC)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize continuous critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(ContinuousCriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-value."""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value