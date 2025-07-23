import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from ..common.networks import ActorNetwork, CriticNetwork, ActorCriticNetwork
from ..common.utils import get_device, compute_returns, compute_gae


class REINFORCE:
    """
    REINFORCE (Monte Carlo Policy Gradient) implementation.
    
    Paper: "Simple statistical gradient-following algorithms for connectionist 
            reinforcement learning" (Williams, 1992)
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[torch.device] = None
    ):
        """
        Initialize REINFORCE agent.
        
        Args:
            state_shape: Shape of the state space
            num_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            device: Device to use for computation
        """
        self.gamma = gamma
        self.device = device or get_device()
        
        # Policy network
        input_channels = state_shape[0] if len(state_shape) == 3 else state_shape[-1]
        self.policy_net = ActorNetwork(input_channels, num_actions).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action according to policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_net(state_tensor)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Store log probability for training
        self.log_probs.append(action_dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward: float) -> None:
        """Store reward."""
        self.rewards.append(reward)
    
    def train_episode(self) -> float:
        """
        Train on collected episode data.
        
        Returns:
            Policy loss
        """
        # Compute returns
        returns = compute_returns(self.rewards, [False] * len(self.rewards), self.gamma)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = []
        for log_prob, return_val in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * return_val)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class ActorCritic:
    """
    Actor-Critic implementation.
    
    Paper: "Actor-Critic Algorithms" (Konda & Tsitsiklis, 2000)
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Actor-Critic agent.
        
        Args:
            state_shape: Shape of the state space
            num_actions: Number of possible actions
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            device: Device to use for computation
        """
        self.gamma = gamma
        self.device = device or get_device()
        
        # Networks
        input_channels = state_shape[0] if len(state_shape) == 3 else state_shape[-1]
        self.actor = ActorNetwork(input_channels, num_actions).to(self.device)
        self.critic = CriticNetwork(input_channels).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Episode storage
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action and estimate value.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities and value
        action_probs = self.actor(state_tensor)
        value = self.critic(state_tensor)
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Store for training
        self.log_probs.append(action_dist.log_prob(action))
        self.values.append(value.squeeze())
        
        return action.item()
    
    def store_transition(self, reward: float, done: bool) -> None:
        """Store transition."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def train_episode(self) -> Tuple[float, float]:
        """
        Train on collected episode data.
        
        Returns:
            Actor loss, Critic loss
        """
        # Compute returns and advantages
        returns = compute_returns(self.rewards, self.dones, self.gamma)
        returns = torch.FloatTensor(returns).to(self.device)
        
        values = torch.stack(self.values)
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss (policy gradient)
        actor_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            actor_loss.append(-log_prob * advantage)
        actor_loss = torch.stack(actor_loss).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values, returns)
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        return actor_loss.item(), critic_loss.item()
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


class A2C:
    """
    Advantage Actor-Critic (A2C) implementation.
    
    Paper: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
    Note: This is the synchronous version.
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize A2C agent.
        
        Args:
            state_shape: Shape of the state space
            num_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use for computation
        """
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or get_device()
        
        # Shared network
        input_channels = state_shape[0] if len(state_shape) == 3 else state_shape[-1]
        self.ac_net = ActorCriticNetwork(input_channels, num_actions).to(self.device)
        
        # Optimizer
        self.optimizer = optim.RMSprop(self.ac_net.parameters(), lr=lr, eps=1e-5, alpha=0.99)
        
        # Episode storage
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.dones = []
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using the actor-critic network.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get policy and value
        action_probs, value = self.ac_net(state_tensor)
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Store for training
        self.log_probs.append(action_dist.log_prob(action))
        self.values.append(value.squeeze())
        self.entropies.append(action_dist.entropy())
        
        return action.item()
    
    def store_transition(self, reward: float, done: bool) -> None:
        """Store transition."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def train_episode(self) -> Tuple[float, float, float]:
        """
        Train on collected episode data.
        
        Returns:
            Policy loss, Value loss, Total loss
        """
        # Compute returns
        returns = compute_returns(self.rewards, self.dones, self.gamma)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Stack tensors
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Compute losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss + 
                     self.entropy_coef * entropy_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.dones = []
        
        return policy_loss.item(), value_loss.item(), total_loss.item()
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        torch.save({
            'ac_net': self.ac_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac_net.load_state_dict(checkpoint['ac_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])