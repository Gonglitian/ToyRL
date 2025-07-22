#!/usr/bin/env python3
"""
Example script demonstrating how to use the RL algorithms library.

This script shows how to:
1. Create an environment
2. Initialize an algorithm
3. Train for a few episodes
4. Evaluate the trained agent
"""

import numpy as np
from common.env_wrappers import make_atari_env
from algorithms.dqn import DQN
from algorithms.policy_gradient import A2C
from common.utils import set_seed


def example_dqn():
    """Example of training DQN on Pong."""
    print("=" * 50)
    print("DQN EXAMPLE")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env = make_atari_env('PongNoFrameskip-v4')
    print(f"Environment: {env}")
    print(f"State shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Create DQN agent
    agent = DQN(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        lr=1e-4,
        memory_size=10000,  # Smaller buffer for quick demo
        batch_size=32
    )
    print(f"Agent created with device: {agent.device}")
    
    # Training loop (just a few episodes for demo)
    print("\nTraining for 5 episodes...")
    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(1000):  # Max 1000 steps per episode
            # Select action
            action = agent.select_action(state, training=True)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train (if enough data)
            loss = agent.train_step()
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Steps = {steps}")
    
    # Quick evaluation
    print("\nEvaluating agent...")
    eval_rewards = []
    for eval_ep in range(3):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        eval_rewards.append(episode_reward)
    
    print(f"Evaluation results: {eval_rewards}")
    print(f"Average reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    
    env.close()


def example_a2c():
    """Example of training A2C on Pong."""
    print("\n" + "=" * 50)
    print("A2C EXAMPLE")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env = make_atari_env('PongNoFrameskip-v4')
    
    # Create A2C agent
    agent = A2C(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        lr=1e-4
    )
    print(f"A2C agent created with device: {agent.device}")
    
    # Training loop
    print("\nTraining for 5 episodes...")
    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(1000):
            # Select action
            action = agent.select_action(state)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(reward, done)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Train at end of episode
        policy_loss, value_loss, total_loss = agent.train_episode()
        print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Steps = {steps}, "
              f"Total Loss = {total_loss:.4f}")
    
    env.close()


def demonstrate_algorithms():
    """Demonstrate multiple algorithms quickly."""
    print("\n" + "=" * 50)
    print("ALGORITHM DEMONSTRATION")
    print("=" * 50)
    
    from algorithms.dqn import DoubleDQN
    from algorithms.policy_gradient import REINFORCE
    from algorithms.advanced_pg import PPO
    
    # Create environment
    env = make_atari_env('PongNoFrameskip-v4')
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    algorithms = [
        ("DQN", DQN(state_shape, num_actions)),
        ("Double DQN", DoubleDQN(state_shape, num_actions)),
        ("REINFORCE", REINFORCE(state_shape, num_actions)),
        ("PPO", PPO(state_shape, num_actions)),
    ]
    
    for name, agent in algorithms:
        print(f"\nTesting {name}...")
        
        # Test action selection
        state, _ = env.reset()
        
        if name == "PPO":
            action, log_prob, value = agent.select_action(state)
            print(f"  Action: {action}, Log Prob: {log_prob:.4f}, Value: {value:.4f}")
        elif hasattr(agent, 'select_action'):
            if 'training' in agent.select_action.__code__.co_varnames:
                action = agent.select_action(state, training=False)
            else:
                action = agent.select_action(state)
            print(f"  Action: {action}")
        
        # Test model saving
        try:
            agent.save_model(f'/tmp/{name.lower().replace(" ", "_")}_test.pth')
            print(f"  ✓ Model saving works")
        except Exception as e:
            print(f"  ✗ Model saving failed: {e}")
    
    env.close()


def main():
    """Run all examples."""
    print("RL ALGORITHMS LIBRARY - QUICK EXAMPLES")
    print("This script demonstrates basic usage of the library.")
    print("For full training, use: python train.py --algorithm <name> --env <env>")
    
    try:
        # Run DQN example
        example_dqn()
        
        # Run A2C example  
        example_a2c()
        
        # Demonstrate multiple algorithms
        demonstrate_algorithms()
        
        print("\n" + "=" * 50)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Run full training: python train.py --algorithm dqn --env PongNoFrameskip-v4")
        print("2. Test all algorithms: python tests/test_algorithms.py")
        print("3. Check the README.md for detailed documentation")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        print("Please check your installation and dependencies.")


if __name__ == "__main__":
    main()