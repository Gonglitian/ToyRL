#!/usr/bin/env python3
"""
Unified training script for all RL algorithms.

This script allows training any of the implemented algorithms on Atari environments.
"""

import argparse
import os
import time
import numpy as np
from tqdm import tqdm

import torch
from common.env_wrappers import make_atari_env
from common.utils import set_seed, MetricsLogger

# Import all algorithms
from algorithms.dqn import DQN, DoubleDQN, DuelingDQN, DuelingDoubleDQN
from algorithms.policy_gradient import REINFORCE, ActorCritic, A2C
from algorithms.advanced_pg import PPO, TRPO  # A3C requires multiprocessing
from algorithms.sac import DiscreteSAC


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL algorithms on Atari')
    
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


def create_agent(algorithm, env, args):
    """Create the specified algorithm agent."""
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
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


def train_value_based(agent, env, args, logger):
    """Train value-based algorithms (DQN variants)."""
    best_reward = -float('inf')
    
    for episode in tqdm(range(args.episodes), desc=f"Training {args.algorithm}"):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        for step in range(args.max_steps):
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
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Log metrics
        avg_loss = episode_loss / max(steps, 1)
        logger.log(episode_reward=episode_reward, episode_loss=avg_loss)
        
        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            eval_env = make_atari_env(args.env)
            eval_reward, eval_std = evaluate_agent(agent, eval_env, args.eval_episodes, args.max_steps)
            eval_env.close()
            
            logger.log(eval_reward=eval_reward, eval_std=eval_std)
            print(f"Episode {episode + 1}: Eval Reward = {eval_reward:.2f} ± {eval_std:.2f}")
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_path = os.path.join(args.save_dir, f'{args.algorithm}_best.pth')
                agent.save_model(best_path)
        
        # Save model periodically
        if (episode + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'{args.algorithm}_episode_{episode + 1}.pth')
            agent.save_model(save_path)


def train_policy_based(agent, env, args, logger):
    """Train policy-based algorithms (REINFORCE, Actor-Critic, A2C)."""
    best_reward = -float('inf')
    
    for episode in tqdm(range(args.episodes), desc=f"Training {args.algorithm}"):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(args.max_steps):
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
            state = next_state
            
            if done:
                break
        
        # Train at end of episode
        if hasattr(agent, 'train_episode'):
            if args.algorithm == 'actor_critic':
                actor_loss, critic_loss = agent.train_episode()
                logger.log(episode_reward=episode_reward, actor_loss=actor_loss, critic_loss=critic_loss)
            elif args.algorithm == 'a2c':
                policy_loss, value_loss, total_loss = agent.train_episode()
                logger.log(episode_reward=episode_reward, policy_loss=policy_loss, 
                          value_loss=value_loss, total_loss=total_loss)
            else:  # REINFORCE
                policy_loss = agent.train_episode()
                logger.log(episode_reward=episode_reward, policy_loss=policy_loss)
        
        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            eval_env = make_atari_env(args.env)
            eval_reward, eval_std = evaluate_agent(agent, eval_env, args.eval_episodes, args.max_steps)
            eval_env.close()
            
            logger.log(eval_reward=eval_reward, eval_std=eval_std)
            print(f"Episode {episode + 1}: Eval Reward = {eval_reward:.2f} ± {eval_std:.2f}")
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_path = os.path.join(args.save_dir, f'{args.algorithm}_best.pth')
                agent.save_model(best_path)
        
        # Save model periodically
        if (episode + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'{args.algorithm}_episode_{episode + 1}.pth')
            agent.save_model(save_path)


def train_advanced_pg(agent, env, args, logger):
    """Train advanced policy gradient algorithms (PPO, TRPO)."""
    best_reward = -float('inf')
    
    for episode in tqdm(range(args.episodes), desc=f"Training {args.algorithm}"):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(args.max_steps):
            # Select action and get additional info
            if args.algorithm in ['ppo', 'trpo']:
                action, log_prob, value = agent.select_action(state)
            else:
                action = agent.select_action(state)
                log_prob = value = 0  # Placeholder
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            if args.algorithm in ['ppo', 'trpo']:
                agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Train at end of episode
        if hasattr(agent, 'update'):
            # Compute next value for bootstrapping
            next_value = 0
            if not done and args.algorithm in ['ppo', 'trpo']:
                _, _, next_value = agent.select_action(next_state)
            
            metrics = agent.update(next_value)
            logger.log(episode_reward=episode_reward, **metrics)
        
        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            eval_env = make_atari_env(args.env)
            eval_reward, eval_std = evaluate_agent(agent, eval_env, args.eval_episodes, args.max_steps)
            eval_env.close()
            
            logger.log(eval_reward=eval_reward, eval_std=eval_std)
            print(f"Episode {episode + 1}: Eval Reward = {eval_reward:.2f} ± {eval_std:.2f}")
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_path = os.path.join(args.save_dir, f'{args.algorithm}_best.pth')
                agent.save_model(best_path)
        
        # Save model periodically
        if (episode + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'{args.algorithm}_episode_{episode + 1}.pth')
            agent.save_model(save_path)


def train_sac(agent, env, args, logger):
    """Train SAC algorithm."""
    best_reward = -float('inf')
    
    for episode in tqdm(range(args.episodes), desc=f"Training {args.algorithm}"):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(args.max_steps):
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
                logger.log(**metrics)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        logger.log(episode_reward=episode_reward)
        
        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            eval_env = make_atari_env(args.env)
            eval_reward, eval_std = evaluate_agent(agent, eval_env, args.eval_episodes, args.max_steps)
            eval_env.close()
            
            logger.log(eval_reward=eval_reward, eval_std=eval_std)
            print(f"Episode {episode + 1}: Eval Reward = {eval_reward:.2f} ± {eval_std:.2f}")
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_path = os.path.join(args.save_dir, f'{args.algorithm}_best.pth')
                agent.save_model(best_path)
        
        # Save model periodically
        if (episode + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'{args.algorithm}_episode_{episode + 1}.pth')
            agent.save_model(save_path)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create environment
    env = make_atari_env(args.env)
    print(f"Environment: {args.env}")
    print(f"State shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Create agent
    agent = create_agent(args.algorithm, env, args)
    print(f"Algorithm: {args.algorithm}")
    print(f"Device: {agent.device if hasattr(agent, 'device') else 'cpu'}")
    
    # Create logger
    logger = MetricsLogger()
    
    # Train based on algorithm type
    print(f"Starting training for {args.episodes} episodes...")
    start_time = time.time()
    
    if args.algorithm in ['dqn', 'double_dqn', 'dueling_dqn', 'dueling_double_dqn']:
        train_value_based(agent, env, args, logger)
    elif args.algorithm in ['reinforce', 'actor_critic', 'a2c']:
        train_policy_based(agent, env, args, logger)
    elif args.algorithm in ['ppo', 'trpo']:
        train_advanced_pg(agent, env, args, logger)
    elif args.algorithm == 'sac':
        train_sac(agent, env, args, logger)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_path = os.path.join(args.save_dir, f'{args.algorithm}_final.pth')
    agent.save_model(final_path)
    
    # Save metrics and create plots
    metrics_path = os.path.join(args.log_dir, f'{args.algorithm}_metrics.npy')
    logger.save_metrics(metrics_path)
    
    # Plot training curves
    if 'episode_reward' in logger.metrics:
        plot_path = os.path.join(args.log_dir, f'{args.algorithm}_training_curves.png')
        plot_metrics = ['episode_reward']
        if 'eval_reward' in logger.metrics:
            plot_metrics.append('eval_reward')
        logger.plot_metrics(plot_metrics, save_path=plot_path)
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_reward, eval_std = evaluate_agent(agent, env, 10, args.max_steps)
    print(f"Final evaluation reward: {eval_reward:.2f} ± {eval_std:.2f}")
    
    env.close()


if __name__ == "__main__":
    main()