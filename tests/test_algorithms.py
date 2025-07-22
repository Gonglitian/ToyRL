#!/usr/bin/env python3
"""
Test script for all RL algorithms.

This script tests that all algorithms can be instantiated and perform basic operations.
"""

import numpy as np
import torch
import gymnasium as gym
from common.env_wrappers import make_atari_env

# Import all algorithms
from algorithms.dqn import DQN, DoubleDQN, DuelingDQN, DuelingDoubleDQN
from algorithms.policy_gradient import REINFORCE, ActorCritic, A2C
from algorithms.advanced_pg import PPO, TRPO
from algorithms.sac import DiscreteSAC


def test_algorithm(algorithm_class, algorithm_name, state_shape, num_actions, is_discrete=True):
    """Test a single algorithm."""
    print(f"\nTesting {algorithm_name}...")
    
    try:
        # Create agent
        if algorithm_name in ['PPO', 'TRPO']:
            agent = algorithm_class(state_shape, num_actions)
        elif algorithm_name == 'DiscreteSAC':
            agent = DiscreteSAC(state_shape, num_actions)
        else:
            agent = algorithm_class(state_shape, num_actions)
        
        print(f"✓ {algorithm_name} agent created successfully")
        
        # Test action selection
        state = np.random.random(state_shape).astype(np.float32)
        
        if algorithm_name in ['PPO', 'TRPO']:
            action, log_prob, value = agent.select_action(state)
            print(f"✓ Action selection: action={action}, log_prob={log_prob:.4f}, value={value:.4f}")
        elif algorithm_name in ['DiscreteSAC']:
            action = agent.select_action(state, evaluate=False)
            print(f"✓ Action selection: action={action}")
        else:
            if hasattr(agent, 'select_action'):
                if 'training' in agent.select_action.__code__.co_varnames:
                    action = agent.select_action(state, training=False)
                else:
                    action = agent.select_action(state)
            else:
                action = agent.select_action(state)
            print(f"✓ Action selection: action={action}")
        
        # Test saving and loading
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, f'{algorithm_name}_test.pth')
            agent.save_model(save_path)
            print(f"✓ Model saved successfully")
            
            agent.load_model(save_path)
            print(f"✓ Model loaded successfully")
        
        print(f"✓ {algorithm_name} passed all tests!")
        return True
        
    except Exception as e:
        print(f"✗ {algorithm_name} failed: {str(e)}")
        return False


def test_environment():
    """Test environment creation."""
    print("Testing environment creation...")
    
    try:
        env = make_atari_env('ALE/Pong-v5')
        print(f"✓ Environment created: {env}")
        print(f"✓ State shape: {env.observation_space.shape}")
        print(f"✓ Action space: {env.action_space.n}")
        
        # Test environment step
        state, info = env.reset()
        print(f"✓ Environment reset: state shape = {state.shape}")
        
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step: reward = {reward}, done = {terminated or truncated}")
        
        env.close()
        print("✓ Environment test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {str(e)}")
        return False


def test_shared_components():
    """Test shared components."""
    print("\nTesting shared components...")
    
    try:
        from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
        from common.networks import DQNNetwork, ActorCriticNetwork
        from common.utils import get_device, set_seed
        
        # Test replay buffer
        buffer = ReplayBuffer(1000)
        state = np.random.random((4, 84, 84)).astype(np.float32)
        next_state = np.random.random((4, 84, 84)).astype(np.float32)
        buffer.push(state, 0, 1.0, next_state, False)
        print("✓ Replay buffer works")
        
        # Test prioritized replay buffer
        pri_buffer = PrioritizedReplayBuffer(1000)
        pri_buffer.push(state, 0, 1.0, next_state, False)
        print("✓ Prioritized replay buffer works")
        
        # Test networks
        device = get_device()
        dqn_net = DQNNetwork(4, 6).to(device)
        ac_net = ActorCriticNetwork(4, 6).to(device)
        
        test_input = torch.randn(1, 4, 84, 84).to(device)
        dqn_output = dqn_net(test_input)
        ac_output = ac_net(test_input)
        
        print(f"✓ DQN network output shape: {dqn_output.shape}")
        print(f"✓ AC network output shapes: {ac_output[0].shape}, {ac_output[1].shape}")
        
        # Test utils
        set_seed(42)
        print("✓ Seed setting works")
        
        print("✓ Shared components test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Shared components test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("RL ALGORITHMS TEST SUITE")
    print("="*60)
    
    # Test environment first
    env_success = test_environment()
    if not env_success:
        print("Environment test failed. Stopping tests.")
        return
    
    # Test shared components
    components_success = test_shared_components()
    if not components_success:
        print("Shared components test failed. Stopping tests.")
        return
    
    # Create test environment to get state/action dimensions
    env = make_atari_env('ALE/Pong-v5')
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    env.close()
    
    # Test all algorithms
    algorithms_to_test = [
        (DQN, "DQN"),
        (DoubleDQN, "DoubleDQN"),
        (DuelingDQN, "DuelingDQN"),
        (DuelingDoubleDQN, "DuelingDoubleDQN"),
        (REINFORCE, "REINFORCE"),
        (ActorCritic, "ActorCritic"),
        (A2C, "A2C"),
        (PPO, "PPO"),
        (TRPO, "TRPO"),
        (DiscreteSAC, "DiscreteSAC"),
    ]
    
    results = []
    for algorithm_class, algorithm_name in algorithms_to_test:
        success = test_algorithm(algorithm_class, algorithm_name, state_shape, num_actions)
        results.append((algorithm_name, success))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Environment Test: {'✓ PASSED' if env_success else '✗ FAILED'}")
    print(f"Components Test: {'✓ PASSED' if components_success else '✗ FAILED'}")
    print()
    
    for algorithm_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{algorithm_name:20}: {status}")
    
    print(f"\nOverall: {passed}/{total} algorithms passed")
    
    if passed == total and env_success and components_success:
        print("🎉 All tests passed! The RL library is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()