#!/usr/bin/env python3
"""
Inference script for running trained RL agents.

This script loads trained models and runs them in environments with optional
GIF recording and statistics collection. Supports all algorithms trained
with the training script.

Usage:
    # Basic inference with default settings
    python inference.py model.checkpoint_path=models/dqn_best.pth
    
    # Run with GIF recording
    python inference.py model.checkpoint_path=models/ppo_best.pth recording.enabled=true
    
    # Run specific number of episodes
    python inference.py model.checkpoint_path=models/a2c_best.pth inference.num_episodes=5
    
    # Run with different algorithm
    python inference.py algorithm=ppo model.checkpoint_path=models/ppo_best.pth
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from common.env_wrappers import make_atari_env
from common.gif_wrapper import (
    GifRecorderWrapper, 
    record_every_n_episodes, 
    record_specific_episodes, 
    record_first_and_last
)
from common.utils import set_seed

# Import all algorithms
from algorithms.dqn import DQN, DoubleDQN, DuelingDQN, DuelingDoubleDQN
from algorithms.policy_gradient import REINFORCE, ActorCritic, A2C
from algorithms.advanced_pg import PPO, TRPO, A3C
from algorithms.sac import SAC, DiscreteSAC


def get_algorithm_class(algorithm_name: str):
    """Get algorithm class by name."""
    algorithm_map = {
        'dqn': DQN,
        'double_dqn': DoubleDQN,
        'dueling_dqn': DuelingDQN,
        'dueling_double_dqn': DuelingDoubleDQN,
        'reinforce': REINFORCE,
        'actor_critic': ActorCritic,
        'a2c': A2C,
        'ppo': PPO,
        'trpo': TRPO,
        'a3c': A3C,
        'sac': SAC,
        'discrete_sac': DiscreteSAC
    }
    
    if algorithm_name not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                        f"Available: {list(algorithm_map.keys())}")
    
    return algorithm_map[algorithm_name]


def create_agent(cfg: DictConfig, env, device: torch.device):
    """Create and initialize agent from configuration."""
    algorithm_name = cfg.algorithm_name
    AlgorithmClass = get_algorithm_class(algorithm_name)
    
    # For inference, we only need basic parameters since we'll load the trained model
    # The model weights will override the network parameters anyway
    
    # Create agent with minimal parameters (just what's needed for structure)
    agent = AlgorithmClass(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        device=device
    )
    
    return agent


def setup_gif_recording(env, cfg: DictConfig):
    """Setup GIF recording wrapper if enabled."""
    if not cfg.recording.enabled:
        return env
    
    # Create episode trigger based on configuration
    record_type = cfg.recording.record_episodes
    record_params = cfg.recording.record_params
    
    if record_type == "all":
        episode_trigger = None  # Record all episodes
    elif record_type == "every_n":
        episode_trigger = record_every_n_episodes(record_params.n)
    elif record_type == "first_n":
        episode_trigger = record_first_and_last(
            cfg.inference.num_episodes, 
            first_n=record_params.n, 
            last_n=0
        )
    elif record_type == "last_n":
        episode_trigger = record_first_and_last(
            cfg.inference.num_episodes, 
            first_n=0, 
            last_n=record_params.n
        )
    elif isinstance(record_type, (list, tuple)) or record_type == "specific":
        episode_trigger = record_specific_episodes(record_params.episodes)
    else:
        print(f"Warning: Unknown record_episodes type: {record_type}, recording all")
        episode_trigger = None
    
    # Setup resize parameter
    resize = cfg.recording.resize
    if resize and len(resize) == 2:
        resize = tuple(resize)
    else:
        resize = None
    
    # Create GIF wrapper
    env = GifRecorderWrapper(
        env,
        save_dir=cfg.recording.save_dir,
        episode_trigger=episode_trigger,
        name_prefix=f"{cfg.algorithm_name}_{cfg.env_name}",
        fps=cfg.recording.fps,
        resize=resize,
        quality=cfg.recording.quality
    )
    
    return env


def run_inference_episode(agent, env, cfg: DictConfig) -> Dict[str, Any]:
    """Run a single inference episode."""
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    episode_info = {
        'reward': 0.0,
        'steps': 0,
        'terminated': False,
        'truncated': False
    }
    
    for step in range(cfg.inference.max_steps_per_episode):
        # Get action from agent
        if cfg.inference.deterministic:
            # Use deterministic policy (no exploration)
            if hasattr(agent, 'select_action'):
                if 'training' in agent.select_action.__code__.co_varnames:
                    action = agent.select_action(state, training=False)
                elif 'evaluate' in agent.select_action.__code__.co_varnames:
                    action = agent.select_action(state, evaluate=True)
                else:
                    action = agent.select_action(state)
            else:
                raise AttributeError(f"Agent {type(agent)} doesn't have select_action method")
        else:
            # Use stochastic policy
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state)
            else:
                raise AttributeError(f"Agent {type(agent)} doesn't have select_action method")
        
        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        steps += 1
        state = next_state
        
        # Check if episode ended
        if terminated or truncated:
            break
    
    episode_info['reward'] = total_reward
    episode_info['steps'] = steps
    episode_info['terminated'] = terminated
    episode_info['truncated'] = truncated
    
    return episode_info


def compute_statistics(episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from episode results."""
    rewards = [ep['reward'] for ep in episode_results]
    steps = [ep['steps'] for ep in episode_results]
    
    stats = {
        'num_episodes': len(episode_results),
        'total_reward': sum(rewards),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_steps': np.mean(steps),
        'std_steps': np.std(steps),
        'min_steps': np.min(steps),
        'max_steps': np.max(steps),
        'success_rate': len([ep for ep in episode_results if ep['terminated']]) / len(episode_results)
    }
    
    return stats


def save_results(stats: Dict[str, Any], episode_results: List[Dict[str, Any]], 
                cfg: DictConfig, filepath: str):
    """Save inference results to file."""
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'statistics': convert_numpy_types(stats),
        'episodes': convert_numpy_types(episode_results),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filepath}")


@hydra.main(version_base="1.3", config_path="configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Main inference function."""
    print("Starting inference with Hydra configuration")
    print("Working directory:", os.getcwd())
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    set_seed(cfg.env_params.seed)
    
    # Setup device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"Using device: {device}")
    
    # Check if model file exists
    model_path = cfg.model.checkpoint_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Create environment
    print(f"Creating environment: {cfg.env_name}")
    env = make_atari_env(cfg.env_name, render_mode=cfg.inference.render_mode)
    
    # Setup GIF recording if enabled
    env = setup_gif_recording(env, cfg)
    
    print(f"Environment: {cfg.env_name}")
    print(f"State shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Create and load agent
    print(f"Creating agent: {cfg.algorithm_name}")
    agent = create_agent(cfg, env, device)
    
    # Load model weights
    try:
        agent.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Set agent to evaluation mode
    if hasattr(agent, 'eval'):
        agent.eval()
    elif hasattr(agent, 'set_eval_mode'):
        agent.set_eval_mode()
    
    # Run inference episodes
    print(f"Running {cfg.inference.num_episodes} episodes...")
    episode_results = []
    
    for episode in range(1, cfg.inference.num_episodes + 1):
        print(f"Episode {episode}/{cfg.inference.num_episodes}", end=" ")
        
        episode_info = run_inference_episode(agent, env, cfg)
        episode_results.append(episode_info)
        
        print(f"Reward: {episode_info['reward']:.2f}, Steps: {episode_info['steps']}")
    
    # Compute and display statistics
    if cfg.evaluation.compute_stats:
        print("\n" + "="*50)
        print("INFERENCE STATISTICS")
        print("="*50)
        
        stats = compute_statistics(episode_results)
        
        print(f"Episodes: {stats['num_episodes']}")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"Mean Steps: {stats['mean_steps']:.1f} ± {stats['std_steps']:.1f}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        
        # Save results if requested
        if cfg.evaluation.save_results:
            save_results(stats, episode_results, cfg, cfg.evaluation.results_file)
    
    # Close environment
    env.close()
    print("\nInference completed!")


def run_with_argparse():
    """Fallback function for argparse compatibility."""
    parser = argparse.ArgumentParser(description="Run inference with trained RL models")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--algorithm", default="dqn", help="Algorithm name")
    parser.add_argument("--env", default="PongNoFrameskip-v4", help="Environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--record_gif", action="store_true", help="Record episodes as GIF")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    
    args = parser.parse_args()
    
    print("Warning: Using argparse mode. Consider using Hydra configuration for more features.")
    print(f"Model: {args.model_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    
    # This is a simplified version - full implementation would require
    # recreating the entire Hydra config structure
    print("Please use: python inference.py model.checkpoint_path=your_model.pth")


if __name__ == "__main__":
    # Check if user wants argparse mode
    if "--use_argparse" in sys.argv:
        sys.argv.remove("--use_argparse")
        run_with_argparse()
    else:
        main()