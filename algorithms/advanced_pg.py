import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import threading
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict

from common.networks import ActorCriticNetwork
from common.utils import get_device, compute_gae, compute_returns


class A3C:
    """
    Asynchronous Advantage Actor-Critic (A3C) implementation.
    
    Paper: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 40.0,
        num_workers: int = 4,
        device: Optional[torch.device] = None
    ):
        """
        Initialize A3C agent.
        
        Args:
            state_shape: Shape of the state space
            num_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_workers: Number of worker processes
            device: Device to use for computation
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_workers = num_workers
        self.device = device or get_device()
        
        # Shared global network
        input_channels = state_shape[0] if len(state_shape) == 3 else state_shape[-1]
        self.global_net = ActorCriticNetwork(input_channels, num_actions).to(self.device)
        self.global_net.share_memory()  # Share across processes
        
        # Global optimizer
        self.optimizer = optim.Adam(self.global_net.parameters(), lr=lr)
        
        # Worker lock for parameter updates
        self.lock = mp.Lock()
    
    def get_local_net(self):
        """Create a local copy of the global network."""
        input_channels = self.global_net.feature_extractor.conv1.in_channels
        num_actions = self.global_net.policy_head.out_features
        local_net = ActorCriticNetwork(input_channels, num_actions).to(self.device)
        local_net.load_state_dict(self.global_net.state_dict())
        return local_net
    
    def sync_with_global(self, local_net):
        """Synchronize local network with global network."""
        local_net.load_state_dict(self.global_net.state_dict())
    
    def train_worker(self, worker_id: int, env_fn, max_episodes: int = 1000):
        """
        Training function for a single worker.
        
        Args:
            worker_id: Worker ID
            env_fn: Function to create environment
            max_episodes: Maximum number of episodes to train
        """
        env = env_fn()
        local_net = self.get_local_net()
        local_optimizer = optim.Adam(local_net.parameters(), lr=7e-4)
        
        episode = 0
        while episode < max_episodes:
            # Sync with global network
            self.sync_with_global(local_net)
            
            # Collect trajectory
            states, actions, rewards, values, log_probs, entropies, dones = [], [], [], [], [], [], []
            
            state, _ = env.reset()
            done = False
            
            while not done and len(states) < 20:  # Short rollouts
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, value = local_net(state_tensor)
                
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                values.append(value.squeeze().item())
                log_probs.append(action_dist.log_prob(action))
                entropies.append(action_dist.entropy())
                dones.append(done)
                
                state = next_state
            
            # Compute returns and advantages
            if done:
                next_value = 0
            else:
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    _, next_value = local_net(next_state_tensor)
                    next_value = next_value.squeeze().item()
            
            returns = compute_returns(rewards + [next_value], dones + [True], self.gamma)[:-1]
            advantages = compute_gae(rewards, values, values[1:] + [next_value], dones, self.gamma, self.gae_lambda)
            
            # Convert to tensors
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            log_probs = torch.stack(log_probs)
            entropies = torch.stack(entropies)
            values_tensor = torch.FloatTensor(values).to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute losses
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values_tensor, returns)
            entropy_loss = -entropies.mean()
            
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Local optimization
            local_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_net.parameters(), self.max_grad_norm)
            
            # Update global network
            with self.lock:
                for global_param, local_param in zip(self.global_net.parameters(), local_net.parameters()):
                    global_param.grad = local_param.grad
                self.optimizer.step()
            
            episode += 1
    
    def train(self, env_fn, max_episodes: int = 1000):
        """
        Train A3C with multiple workers.
        
        Args:
            env_fn: Function to create environment
            max_episodes: Maximum episodes per worker
        """
        processes = []
        for worker_id in range(self.num_workers):
            p = mp.Process(target=self.train_worker, args=(worker_id, env_fn, max_episodes))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        torch.save({
            'global_net': self.global_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_net.load_state_dict(checkpoint['global_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class PPO:
    """
    Proximal Policy Optimization (PPO) implementation.
    
    Paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        device: Optional[torch.device] = None
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_shape: Shape of the state space
            num_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            mini_batch_size: Mini-batch size for PPO updates
            device: Device to use for computation
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device or get_device()
        
        # Actor-Critic network
        input_channels = state_shape[0] if len(state_shape) == 3 else state_shape[-1]
        self.ac_net = ActorCriticNetwork(input_channels, num_actions).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr, eps=1e-5)
        
        # Storage for rollouts
        self.reset_storage()
    
    def reset_storage(self):
        """Reset rollout storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action and get value estimate.
        
        Args:
            state: Current state
            
        Returns:
            Action, log probability, value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.ac_net(state_tensor)
            
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, done: bool) -> None:
        """Store transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        values = self.values + [next_value]
        advantages = compute_gae(self.rewards, self.values, values[1:], self.dones, 
                               self.gamma, self.gae_lambda)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return np.array(advantages), np.array(returns)
    
    def update(self, next_value: float = 0) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            next_value: Value estimate for next state
            
        Returns:
            Training metrics
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Generate mini-batches
            indices = torch.randperm(len(self.states))
            
            for start in range(0, len(self.states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_probs, values = self.ac_net(batch_states)
                action_dist = torch.distributions.Categorical(action_probs)
                
                # New log probabilities
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy()
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_loss_coef * value_loss + 
                             self.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Reset storage
        self.reset_storage()
        
        num_updates = self.ppo_epochs * (len(states) // self.mini_batch_size)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates
        }
    
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


class TRPO:
    """
    Trust Region Policy Optimization (TRPO) implementation.
    
    Paper: "Trust Region Policy Optimization" (Schulman et al., 2015)
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_kl: float = 0.01,
        damping: float = 0.1,
        cg_iters: int = 10,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        device: Optional[torch.device] = None
    ):
        """
        Initialize TRPO agent.
        
        Args:
            state_shape: Shape of the state space
            num_actions: Number of possible actions
            lr_critic: Critic learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            max_kl: Maximum KL divergence
            damping: Damping parameter for conjugate gradient
            cg_iters: Conjugate gradient iterations
            backtrack_iters: Line search iterations
            backtrack_coeff: Line search coefficient
            device: Device to use for computation
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.damping = damping
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.device = device or get_device()
        
        # Networks
        input_channels = state_shape[0] if len(state_shape) == 3 else state_shape[-1]
        self.ac_net = ActorCriticNetwork(input_channels, num_actions).to(self.device)
        
        # Critic optimizer (only critic is optimized with gradient descent)
        self.critic_optimizer = optim.Adam(self.ac_net.value_head.parameters(), lr=lr_critic)
        
        # Storage
        self.reset_storage()
    
    def reset_storage(self):
        """Reset storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.ac_net(state_tensor)
            
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, done: bool) -> None:
        """Store transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def conjugate_gradient(self, Ax_func, b, x_0=None, residual_tol=1e-10):
        """Conjugate gradient algorithm."""
        if x_0 is None:
            x = torch.zeros_like(b)
        else:
            x = x_0.clone()
        
        r = b - Ax_func(x)
        p = r.clone()
        rsold = torch.sum(r * r)
        
        for i in range(self.cg_iters):
            Ap = Ax_func(p)
            alpha = rsold / torch.sum(p * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.sum(r * r)
            
            if torch.sqrt(rsnew) < residual_tol:
                break
                
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x
    
    def update(self, next_value: float = 0) -> Dict[str, float]:
        """Update using TRPO."""
        # Compute advantages and returns
        values = self.values + [next_value]
        advantages = compute_gae(self.rewards, self.values, values[1:], self.dones,
                               self.gamma, self.gae_lambda)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old policy and value predictions
        with torch.no_grad():
            old_action_probs, old_values = self.ac_net(states)
            old_action_dist = torch.distributions.Categorical(old_action_probs)
        
        # Policy gradient
        action_probs, values = self.ac_net(states)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        
        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = -(ratio * advantages).mean()
        
        # Policy gradient
        policy_grads = torch.autograd.grad(policy_loss, self.ac_net.policy_head.parameters(), retain_graph=True)
        policy_grad = torch.cat([grad.view(-1) for grad in policy_grads])
        
        # Fisher Information Matrix-vector product
        def fvp(v):
            kl = torch.distributions.kl.kl_divergence(old_action_dist, action_dist).mean()
            grads = torch.autograd.grad(kl, self.ac_net.policy_head.parameters(), create_graph=True)
            flat_grad = torch.cat([grad.view(-1) for grad in grads])
            
            kl_v = (flat_grad * v).sum()
            grads = torch.autograd.grad(kl_v, self.ac_net.policy_head.parameters())
            flat_grad_grad = torch.cat([grad.contiguous().view(-1) for grad in grads])
            
            return flat_grad_grad + self.damping * v
        
        # Solve for step direction using conjugate gradient
        stepdir = self.conjugate_gradient(fvp, -policy_grad)
        
        # Compute step size
        shs = 0.5 * torch.sum(stepdir * fvp(stepdir))
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm
        
        # Line search
        def get_kl():
            action_probs, _ = self.ac_net(states)
            action_dist = torch.distributions.Categorical(action_probs)
            return torch.distributions.kl.kl_divergence(old_action_dist, action_dist).mean()
        
        def get_loss():
            action_probs, _ = self.ac_net(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            return -(ratio * advantages).mean()
        
        # Save current parameters
        old_params = torch.cat([param.view(-1) for param in self.ac_net.policy_head.parameters()])
        old_loss = get_loss()
        
        # Line search
        for i in range(self.backtrack_iters):
            new_params = old_params + (self.backtrack_coeff ** i) * fullstep
            
            # Set new parameters
            idx = 0
            for param in self.ac_net.policy_head.parameters():
                param_length = param.numel()
                param.data = new_params[idx:idx + param_length].view(param.shape)
                idx += param_length
            
            new_loss = get_loss()
            new_kl = get_kl()
            
            if new_kl <= self.max_kl and new_loss < old_loss:
                break
        else:
            # Restore old parameters if no improvement
            idx = 0
            for param in self.ac_net.policy_head.parameters():
                param_length = param.numel()
                param.data = old_params[idx:idx + param_length].view(param.shape)
                idx += param_length
        
        # Update critic
        value_loss = F.mse_loss(values.squeeze(), returns)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        # Reset storage
        self.reset_storage()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_divergence': get_kl().item()
        }
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        torch.save({
            'ac_net': self.ac_net.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac_net.load_state_dict(checkpoint['ac_net'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])