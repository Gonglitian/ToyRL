#!/usr/bin/env python3
"""
Unified training script for all RL algorithms.

This script allows training any of the implemented algorithms on Atari environments.
Supports both Hydra configuration and traditional argparse for backward compatibility.
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import Any, Dict, Union
from tqdm import tqdm

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from common.env_wrappers import make_atari_env
from common.utils import set_seed, MetricsLogger
from common.logger import Logger

# Import all algorithms
from algorithms.dqn import DQN, DoubleDQN, DuelingDQN, DuelingDoubleDQN
from algorithms.policy_gradient import REINFORCE, ActorCritic, A2C
from algorithms.advanced_pg import PPO, TRPO  # A3C requires multiprocessing
from algorithms.sac import DiscreteSAC


def parse_args():
    """Parse command line arguments for backward compatibility."""
    parser = argparse.ArgumentParser(description='Train RL algorithms on Atari')
    
    # Add use_argparse flag to determine which system to use
    parser.add_argument('--use_argparse', action='store_true',
                        help='Use argparse instead of Hydra (for backward compatibility)')
    
    # Environment
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='Atari environment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Algorithm
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'dueling_double_dqn',
                                'reinforce', 'actor_critic', 'a2c', 'ppo', 'trpo', 'sac'],
                        help='Algorithm to train')
    
    # Training
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Maximum steps per episode')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Evaluation frequency (episodes)')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='Model saving frequency (episodes)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs and plots')
    
    return parser.parse_args()


def create_agent(algorithm: str, env, config: Union[Dict, Any]):
    """Create the specified algorithm agent."""
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    # Get algorithm name - handle both argparse and Hydra config
    if hasattr(config, 'algorithm_name'):
        algorithm = config.algorithm_name
    elif hasattr(config, 'algorithm'):
        algorithm = config.algorithm
    
    if algorithm == 'dqn':
        return DQN(state_shape, num_actions)
    elif algorithm == 'double_dqn':
        return DoubleDQN(state_shape, num_actions)
    elif algorithm == 'dueling_dqn':
        return DuelingDQN(state_shape, num_actions)
    elif algorithm == 'dueling_double_dqn':
        return DuelingDoubleDQN(state_shape, num_actions)
    elif algorithm == 'reinforce':
        return REINFORCE(state_shape, num_actions)
    elif algorithm == 'actor_critic':
        return ActorCritic(state_shape, num_actions)
    elif algorithm == 'a2c':
        return A2C(state_shape, num_actions)
    elif algorithm == 'ppo':
        return PPO(state_shape, num_actions)
    elif algorithm == 'trpo':
        return TRPO(state_shape, num_actions)
    elif algorithm == 'sac':
        return DiscreteSAC(state_shape, num_actions)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_logger(config: Union[Dict, Any], algorithm: str) -> Logger:
    """Create TensorBoard logger."""
    # Get log directory
    if hasattr(config, 'log_dir'):
        log_dir = config.log_dir
    else:
        log_dir = getattr(config, 'log_dir', 'runs')
    
    # Create timestamped directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir, f"{algorithm}_{timestamp}")
    
    return Logger(log_dir)


def get_sample_input(env):
    """Get a sample input for model graph logging."""
    state, _ = env.reset()
    return torch.FloatTensor(state).unsqueeze(0)


def evaluate_agent(agent, env, episodes=5, max_steps=10000):
    """Evaluate the agent."""
    total_rewards = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if hasattr(agent, 'select_action'):
                if hasattr(agent.select_action, '__code__') and 'evaluate' in agent.select_action.__code__.co_varnames:
                    action = agent.select_action(state, evaluate=True)
                else:
                    action = agent.select_action(state)
            else:
                # For REINFORCE and similar algorithms
                action = agent.select_action(state)
            
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)


def train_value_based(agent, env, config, logger: Logger, use_tensorboard: bool = True):
    """Train value-based algorithms (DQN variants)."""
    best_reward = -float('inf')
    global_step = 0
    
    # Log model graph if enabled
    if use_tensorboard and hasattr(config, 'tensorboard') and config.tensorboard.get('log_model_graph', True):
        try:
            sample_input = get_sample_input(env)
            if hasattr(agent, 'q_network'):
                logger.log_model_graph(agent.q_network, sample_input)
            elif hasattr(agent, 'policy_network'):
                logger.log_model_graph(agent.policy_network, sample_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    episodes = getattr(config, 'episodes', 1000)
    max_steps = getattr(config, 'max_steps', 10000)
    eval_freq = getattr(config, 'eval_freq', 100)
    eval_episodes = getattr(config, 'eval_episodes', 5)
    save_freq = getattr(config, 'save_freq', 200)
    save_dir = getattr(config, 'save_dir', 'models')
    algorithm = getattr(config, 'algorithm', getattr(config, 'algorithm_name', 'dqn'))
    
    for episode in tqdm(range(episodes), desc=f"Training {algorithm}"):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss += loss
                steps += 1
                global_step += 1
                
                # Log training loss
                if use_tensorboard:
                    logger.log_loss(loss, global_step)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Log episode metrics
        avg_loss = episode_loss / max(steps, 1)
        
        if use_tensorboard:
            logger.log_episode_reward(episode_reward, episode)
            logger.log_episode_length(episode_length, episode)
            if hasattr(agent, 'epsilon'):
                logger.log_exploration_rate(agent.epsilon, global_step)
            
            # Log network weights periodically
            if hasattr(config, 'tensorboard') and config.tensorboard.get('log_histograms', True):
                if episode % 50 == 0:  # Log every 50 episodes
                    if hasattr(agent, 'q_network'):
                        logger.log_network_weights(agent.q_network, global_step)
        else:
            # Backward compatibility logging
            logger.log(episode_reward=episode_reward, episode_loss=avg_loss)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_env = make_atari_env(getattr(config, 'env_name', getattr(config, 'env', 'PongNoFrameskip-v4')))
            eval_reward, eval_std = evaluate_agent(agent, eval_env, eval_episodes, max_steps)
            eval_env.close()
            
            if use_tensorboard:
                logger.log_evaluation_results(eval_reward, eval_std, episode)
            else:
                logger.log(eval_reward=eval_reward, eval_std=eval_std)
                
            print(f"Episode {episode + 1}: Eval Reward = {eval_reward:.2f} ± {eval_std:.2f}")
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                os.makedirs(save_dir, exist_ok=True)
                best_path = os.path.join(save_dir, f'{algorithm}_best.pth')
                agent.save_model(best_path)
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{algorithm}_episode_{episode + 1}.pth')
            agent.save_model(save_path)


def train_policy_based(agent, env, config, logger: Logger, use_tensorboard: bool = True):
    """Train policy-based algorithms (REINFORCE, Actor-Critic, A2C)."""
    best_reward = -float('inf')
    
    # Log model graph if enabled
    if use_tensorboard and hasattr(config, 'tensorboard') and config.tensorboard.get('log_model_graph', True):
        try:
            sample_input = get_sample_input(env)
            if hasattr(agent, 'policy_network'):
                logger.log_model_graph(agent.policy_network, sample_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    episodes = getattr(config, 'episodes', 1000)
    max_steps = getattr(config, 'max_steps', 10000)
    eval_freq = getattr(config, 'eval_freq', 100)
    eval_episodes = getattr(config, 'eval_episodes', 5)
    save_freq = getattr(config, 'save_freq', 200)
    save_dir = getattr(config, 'save_dir', 'models')
    algorithm = getattr(config, 'algorithm', getattr(config, 'algorithm_name', 'reinforce'))
    
    for episode in tqdm(range(episodes), desc=f"Training {algorithm}"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            if hasattr(agent, 'store_reward'):
                agent.store_reward(reward)
            elif hasattr(agent, 'store_transition'):
                agent.store_transition(reward, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Train at end of episode
        if hasattr(agent, 'train_episode'):
            if algorithm == 'actor_critic':
                actor_loss, critic_loss = agent.train_episode()
                if use_tensorboard:
                    logger.log_scalar("Training/ActorLoss", actor_loss, episode)
                    logger.log_scalar("Training/CriticLoss", critic_loss, episode)
                else:
                    logger.log(episode_reward=episode_reward, actor_loss=actor_loss, critic_loss=critic_loss)
            elif algorithm == 'a2c':
                policy_loss, value_loss, total_loss = agent.train_episode()
                if use_tensorboard:
                    logger.log_scalar("Training/PolicyLoss", policy_loss, episode)
                    logger.log_scalar("Training/ValueLoss", value_loss, episode)
                    logger.log_scalar("Training/TotalLoss", total_loss, episode)
                else:
                    logger.log(episode_reward=episode_reward, policy_loss=policy_loss, 
                              value_loss=value_loss, total_loss=total_loss)
            else:  # REINFORCE
                policy_loss = agent.train_episode()
                if use_tensorboard:
                    logger.log_scalar("Training/PolicyLoss", policy_loss, episode)
                else:
                    logger.log(episode_reward=episode_reward, policy_loss=policy_loss)
        
        # Log episode metrics
        if use_tensorboard:
            logger.log_episode_reward(episode_reward, episode)
            logger.log_episode_length(episode_length, episode)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_env = make_atari_env(getattr(config, 'env_name', getattr(config, 'env', 'PongNoFrameskip-v4')))
            eval_reward, eval_std = evaluate_agent(agent, eval_env, eval_episodes, max_steps)
            eval_env.close()
            
            if use_tensorboard:
                logger.log_evaluation_results(eval_reward, eval_std, episode)
            else:
                logger.log(eval_reward=eval_reward, eval_std=eval_std)
                
            print(f"Episode {episode + 1}: Eval Reward = {eval_reward:.2f} ± {eval_std:.2f}")
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                os.makedirs(save_dir, exist_ok=True)
                best_path = os.path.join(save_dir, f'{algorithm}_best.pth')
                agent.save_model(best_path)
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{algorithm}_episode_{episode + 1}.pth')
            agent.save_model(save_path)


def train_advanced_pg(agent, env, config, logger: Logger, use_tensorboard: bool = True):
    """Train advanced policy gradient algorithms (PPO, TRPO)."""
    best_reward = -float('inf')
    
    # Log model graph if enabled
    if use_tensorboard and hasattr(config, 'tensorboard') and config.tensorboard.get('log_model_graph', True):
        try:
            sample_input = get_sample_input(env)
            if hasattr(agent, 'policy_network'):
                logger.log_model_graph(agent.policy_network, sample_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    episodes = getattr(config, 'episodes', 1000)
    max_steps = getattr(config, 'max_steps', 10000)
    eval_freq = getattr(config, 'eval_freq', 100)
    eval_episodes = getattr(config, 'eval_episodes', 5)
    save_freq = getattr(config, 'save_freq', 200)
    save_dir = getattr(config, 'save_dir', 'models')
    algorithm = getattr(config, 'algorithm', getattr(config, 'algorithm_name', 'ppo'))
    
    for episode in tqdm(range(episodes), desc=f"Training {algorithm}"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action and get additional info
            if algorithm in ['ppo', 'trpo']:
                action, log_prob, value = agent.select_action(state)
            else:
                action = agent.select_action(state)
                log_prob = value = 0  # Placeholder
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            if algorithm in ['ppo', 'trpo']:
                agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Train at end of episode
        if hasattr(agent, 'update'):
            # Compute next value for bootstrapping
            next_value = 0
            if not done and algorithm in ['ppo', 'trpo']:
                _, _, next_value = agent.select_action(next_state)
            
            metrics = agent.update(next_value)
            if use_tensorboard:
                for key, value in metrics.items():
                    logger.log_scalar(f"Training/{key}", value, episode)
                logger.log_episode_reward(episode_reward, episode)
                logger.log_episode_length(episode_length, episode)
            else:
                logger.log(episode_reward=episode_reward, **metrics)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_env = make_atari_env(getattr(config, 'env_name', getattr(config, 'env', 'PongNoFrameskip-v4')))
            eval_reward, eval_std = evaluate_agent(agent, eval_env, eval_episodes, max_steps)
            eval_env.close()
            
            if use_tensorboard:
                logger.log_evaluation_results(eval_reward, eval_std, episode)
            else:
                logger.log(eval_reward=eval_reward, eval_std=eval_std)
                
            print(f"Episode {episode + 1}: Eval Reward = {eval_reward:.2f} ± {eval_std:.2f}")
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                os.makedirs(save_dir, exist_ok=True)
                best_path = os.path.join(save_dir, f'{algorithm}_best.pth')
                agent.save_model(best_path)
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{algorithm}_episode_{episode + 1}.pth')
            agent.save_model(save_path)


def train_sac(agent, env, config, logger: Logger, use_tensorboard: bool = True):
    """Train SAC algorithm."""
    best_reward = -float('inf')
    global_step = 0
    
    # Log model graph if enabled
    if use_tensorboard and hasattr(config, 'tensorboard') and config.tensorboard.get('log_model_graph', True):
        try:
            sample_input = get_sample_input(env)
            if hasattr(agent, 'policy_net'):
                logger.log_model_graph(agent.policy_net, sample_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    episodes = getattr(config, 'episodes', 1000)
    max_steps = getattr(config, 'max_steps', 10000)
    eval_freq = getattr(config, 'eval_freq', 100)
    eval_episodes = getattr(config, 'eval_episodes', 5)
    save_freq = getattr(config, 'save_freq', 200)
    save_dir = getattr(config, 'save_dir', 'models')
    algorithm = getattr(config, 'algorithm', getattr(config, 'algorithm_name', 'sac'))
    
    for episode in tqdm(range(episodes), desc=f"Training {algorithm}"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, evaluate=False)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            metrics = agent.train_step()
            if metrics is not None:
                global_step += 1
                if use_tensorboard:
                    for key, value in metrics.items():
                        logger.log_scalar(f"Training/{key}", value, global_step)
                else:
                    logger.log(**metrics)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        if use_tensorboard:
            logger.log_episode_reward(episode_reward, episode)
            logger.log_episode_length(episode_length, episode)
        else:
            logger.log(episode_reward=episode_reward)
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_env = make_atari_env(getattr(config, 'env_name', getattr(config, 'env', 'PongNoFrameskip-v4')))
            eval_reward, eval_std = evaluate_agent(agent, eval_env, eval_episodes, max_steps)
            eval_env.close()
            
            if use_tensorboard:
                logger.log_evaluation_results(eval_reward, eval_std, episode)
            else:
                logger.log(eval_reward=eval_reward, eval_std=eval_std)
                
            print(f"Episode {episode + 1}: Eval Reward = {eval_reward:.2f} ± {eval_std:.2f}")
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                os.makedirs(save_dir, exist_ok=True)
                best_path = os.path.join(save_dir, f'{algorithm}_best.pth')
                agent.save_model(best_path)
        
        # Save model periodically
        if (episode + 1) % save_freq == 0:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{algorithm}_episode_{episode + 1}.pth')
            agent.save_model(save_path)


def train_with_config(config: Union[DictConfig, Any]):
    """Train with either Hydra config or argparse config."""
    # Determine if we're using TensorBoard (Hydra) or old logger (argparse)
    use_tensorboard = isinstance(config, DictConfig)
    
    # Set seed for reproducibility
    seed = getattr(config, 'seed', 42)
    set_seed(seed)
    
    # Create directories
    save_dir = getattr(config, 'save_dir', 'models')
    log_dir = getattr(config, 'log_dir', 'logs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env_name = getattr(config, 'env_name', getattr(config, 'env', 'PongNoFrameskip-v4'))
    env = make_atari_env(env_name)
    print(f"Environment: {env_name}")
    print(f"State shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Create agent
    algorithm = getattr(config, 'algorithm_name', getattr(config, 'algorithm', 'dqn'))
    agent = create_agent(algorithm, env, config)
    print(f"Algorithm: {algorithm}")
    print(f"Device: {agent.device if hasattr(agent, 'device') else 'cpu'}")
    
    # Create logger
    if use_tensorboard:
        logger = create_logger(config, algorithm)
        print(f"TensorBoard logs will be saved to: {logger.log_dir}")
    else:
        logger = MetricsLogger()
    
    # Train based on algorithm type
    episodes = getattr(config, 'episodes', 1000)
    print(f"Starting training for {episodes} episodes...")
    start_time = time.time()
    
    if algorithm in ['dqn', 'double_dqn', 'dueling_dqn', 'dueling_double_dqn']:
        train_value_based(agent, env, config, logger, use_tensorboard)
    elif algorithm in ['reinforce', 'actor_critic', 'a2c']:
        train_policy_based(agent, env, config, logger, use_tensorboard)
    elif algorithm in ['ppo', 'trpo']:
        train_advanced_pg(agent, env, config, logger, use_tensorboard)
    elif algorithm == 'sac':
        train_sac(agent, env, config, logger, use_tensorboard)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, f'{algorithm}_final.pth')
    agent.save_model(final_path)
    
    if use_tensorboard:
        # Close TensorBoard logger
        logger.close()
    else:
        # Backward compatibility: Save metrics and create plots
        metrics_path = os.path.join(log_dir, f'{algorithm}_metrics.npy')
        logger.save_metrics(metrics_path)
        
        # Plot training curves
        if 'episode_reward' in logger.metrics:
            plot_path = os.path.join(log_dir, f'{algorithm}_training_curves.png')
            plot_metrics = ['episode_reward']
            if 'eval_reward' in logger.metrics:
                plot_metrics.append('eval_reward')
            logger.plot_metrics(plot_metrics, save_path=plot_path)
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_episodes = getattr(config, 'eval_episodes', 5)
    max_steps = getattr(config, 'max_steps', 10000)
    eval_reward, eval_std = evaluate_agent(agent, env, eval_episodes, max_steps)
    print(f"Final evaluation reward: {eval_reward:.2f} ± {eval_std:.2f}")
    
    env.close()


@hydra.main(version_base="1.3", config_path="configs", config_name="trainer")
def main_hydra(cfg: DictConfig) -> None:
    """Main function for Hydra configuration."""
    print("Using Hydra configuration")
    print(f"Working directory: {os.getcwd()}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    train_with_config(cfg)


def main_argparse():
    """Main function for argparse (backward compatibility)."""
    print("Using argparse configuration (backward compatibility mode)")
    args = parse_args()
    train_with_config(args)


def main():
    """Main entry point that determines which configuration system to use."""
    # Check if --use_argparse is in the command line arguments
    if '--use_argparse' in sys.argv:
        main_argparse()
    else:
        # Use Hydra by default
        main_hydra()


if __name__ == "__main__":
    main()