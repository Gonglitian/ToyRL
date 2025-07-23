import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional

from ..common.networks import DQNNetwork, DuelingDQNNetwork
from ..common.replay_buffer import ReplayBuffer
from ..common.utils import soft_update, hard_update, get_device, EpsilonScheduler


class DQN:
    """
    Deep Q-Network (DQN) implementation.
    
    Paper: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1000000,
        memory_size: int = 1000000,
        batch_size: int = 32,
        target_update_freq: int = 10000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_shape: Shape of the state space
            num_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Number of steps for epsilon decay
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to use for computation
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or get_device()
        
        # Networks
        input_channels = state_shape[0] if len(state_shape) == 3 else state_shape[-1]
        self.q_network = DQNNetwork(input_channels, num_actions).to(self.device)
        self.target_network = DQNNetwork(input_channels, num_actions).to(self.device)
        
        # Initialize target network
        hard_update(self.target_network, self.q_network)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Exploration
        self.epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_end, epsilon_decay)
        
        # Training step counter
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon_scheduler.get_epsilon():
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            hard_update(self.target_network, self.q_network)
        
        # Update epsilon
        self.epsilon_scheduler.step()
        
        return loss.item()
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']


class DoubleDQN(DQN):
    """
    Double Deep Q-Network (Double DQN) implementation.
    
    Paper: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2016)
    """
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using Double DQN update rule.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            hard_update(self.target_network, self.q_network)
        
        # Update epsilon
        self.epsilon_scheduler.step()
        
        return loss.item()


class DuelingDQN(DQN):
    """
    Dueling Deep Q-Network (Dueling DQN) implementation.
    
    Paper: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1000000,
        memory_size: int = 1000000,
        batch_size: int = 32,
        target_update_freq: int = 10000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Dueling DQN agent.
        
        Args:
            state_shape: Shape of the state space
            num_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Number of steps for epsilon decay
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to use for computation
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or get_device()
        
        # Networks - use dueling architecture
        input_channels = state_shape[0] if len(state_shape) == 3 else state_shape[-1]
        self.q_network = DuelingDQNNetwork(input_channels, num_actions).to(self.device)
        self.target_network = DuelingDQNNetwork(input_channels, num_actions).to(self.device)
        
        # Initialize target network
        hard_update(self.target_network, self.q_network)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Exploration
        self.epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_end, epsilon_decay)
        
        # Training step counter
        self.training_step = 0


class DuelingDoubleDQN(DuelingDQN):
    """
    Dueling Double Deep Q-Network combining both improvements.
    """
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using Double DQN update rule with Dueling architecture.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN with Dueling architecture
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            hard_update(self.target_network, self.q_network)
        
        # Update epsilon
        self.epsilon_scheduler.step()
        
        return loss.item()