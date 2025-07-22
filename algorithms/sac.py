import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from common.networks import ContinuousActorNetwork, ContinuousCriticNetwork
from common.replay_buffer import ReplayBuffer
from common.utils import soft_update, get_device


class SAC:
    """
    Soft Actor-Critic (SAC) implementation.
    
    Paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning 
            with a Stochastic Actor" (Haarnoja et al., 2018)
    
    Note: This implementation is designed for continuous action spaces.
    For discrete action spaces, modifications would be needed.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_range: float = 1.0,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        memory_size: int = 1000000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        device: Optional[torch.device] = None
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_range: Range of actions (assumes symmetric range [-action_range, action_range])
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            lr_alpha: Alpha (temperature) learning rate
            gamma: Discount factor
            tau: Soft update parameter
            alpha: Initial entropy coefficient
            automatic_entropy_tuning: Whether to automatically tune entropy coefficient
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            hidden_dim: Hidden dimension for networks
            device: Device to use for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or get_device()
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        self.actor = ContinuousActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Two critic networks (for stability)
        self.critic1 = ContinuousCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = ContinuousCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Target critics
        self.target_critic1 = ContinuousCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = ContinuousCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Entropy coefficient
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha).to(self.device)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select action.
        
        Args:
            state: Current state
            evaluate: Whether in evaluation mode (deterministic)
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean) * self.action_range
        else:
            with torch.no_grad():
                mean, log_std = self.actor(state_tensor)
                std = log_std.exp()
                
                # Reparameterization trick
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t) * self.action_range
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def sample_action(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick and compute log probability.
        
        Args:
            state_tensor: State tensor
            
        Returns:
            Action and log probability
        """
        mean, log_std = self.actor(state_tensor)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Apply tanh squashing
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Correct for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Scale action to action range
        action = action * self.action_range
        
        return action, log_prob
    
    def train_step(self) -> Optional[dict]:
        """
        Perform one training step.
        
        Returns:
            Training metrics if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.sample_action(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (~dones) * self.gamma * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Optimize critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.sample_action(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (entropy coefficient)
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        soft_update(self.target_critic1, self.critic1, self.tau)
        soft_update(self.target_critic2, self.critic2, self.tau)
        
        # Return training metrics
        metrics = {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item()
        }
        
        if alpha_loss is not None:
            metrics['alpha_loss'] = alpha_loss.item()
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }
        
        if self.automatic_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()


class DiscreteSAC:
    """
    Discrete SAC implementation for discrete action spaces like Atari.
    
    This is an adaptation of SAC for discrete action spaces.
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        memory_size: int = 1000000,
        batch_size: int = 256,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Discrete SAC agent.
        
        Args:
            state_shape: Shape of the state space
            num_actions: Number of possible actions
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            lr_alpha: Alpha (temperature) learning rate
            gamma: Discount factor
            tau: Soft update parameter
            alpha: Initial entropy coefficient
            automatic_entropy_tuning: Whether to automatically tune entropy coefficient
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            device: Device to use for computation
        """
        from common.networks import ActorNetwork, DQNNetwork
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or get_device()
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        input_channels = state_shape[0] if len(state_shape) == 3 else state_shape[-1]
        self.actor = ActorNetwork(input_channels, num_actions).to(self.device)
        
        # Two critic networks (Q-networks)
        self.critic1 = DQNNetwork(input_channels, num_actions).to(self.device)
        self.critic2 = DQNNetwork(input_channels, num_actions).to(self.device)
        
        # Target critics
        self.target_critic1 = DQNNetwork(input_channels, num_actions).to(self.device)
        self.target_critic2 = DQNNetwork(input_channels, num_actions).to(self.device)
        
        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Entropy coefficient
        if automatic_entropy_tuning:
            self.target_entropy = -0.98 * np.log(1.0 / num_actions)  # Heuristic
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha).to(self.device)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            
            if evaluate:
                action = action_probs.argmax()
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
        
        return action.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[dict]:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_log_probs = torch.log(next_action_probs + 1e-8)
            
            target_q1 = self.target_critic1(next_states)
            target_q2 = self.target_critic2(next_states)
            target_q = torch.min(target_q1, target_q2)
            
            # Soft value function
            next_v = (next_action_probs * (target_q - self.alpha * next_log_probs)).sum(dim=1)
            target_q_values = rewards + (~dones) * self.gamma * next_v
        
        current_q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze()
        current_q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        critic1_loss = F.mse_loss(current_q1, target_q_values)
        critic2_loss = F.mse_loss(current_q2, target_q_values)
        
        # Optimize critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        action_probs = self.actor(states)
        log_probs = torch.log(action_probs + 1e-8)
        
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q = torch.min(q1, q2)
        
        # Actor loss
        actor_loss = (action_probs * (self.alpha * log_probs - q)).sum(dim=1).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = None
        if self.automatic_entropy_tuning:
            entropy = -(action_probs * log_probs).sum(dim=1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        soft_update(self.target_critic1, self.critic1, self.tau)
        soft_update(self.target_critic2, self.critic2, self.tau)
        
        # Return metrics
        metrics = {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item()
        }
        
        if alpha_loss is not None:
            metrics['alpha_loss'] = alpha_loss.item()
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }
        
        if self.automatic_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()