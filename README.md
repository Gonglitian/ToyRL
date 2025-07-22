# Deep Reinforcement Learning Algorithms Library

A comprehensive implementation of classic deep reinforcement learning algorithms tested on Atari environments. This library provides clean, modular implementations of 10 fundamental RL algorithms with shared components for easy experimentation and comparison.

## 🚀 Features

- **10 Classic RL Algorithms**: DQN, Double DQN, Dueling DQN, REINFORCE, Actor-Critic, A2C, A3C, PPO, TRPO, SAC
- **Hydra Configuration System**: Modern configuration management with YAML files and command-line overrides
- **TensorBoard Integration**: Real-time visualization of training metrics, model graphs, and hyperparameters
- **Atari Environment Support**: Pre-configured wrappers for Atari games with standard preprocessing
- **Modular Design**: Shared components like replay buffers, neural networks, and utilities
- **Easy Training**: Unified training script that works with all algorithms
- **Backward Compatibility**: Supports both new Hydra configs and traditional argparse
- **Comprehensive Testing**: Test suite to verify all implementations
- **Documentation**: Detailed explanations of each algorithm and usage examples

## 📁 Project Structure

```
rl-algorithms/
├── algorithms/                 # Algorithm implementations
│   ├── __init__.py
│   ├── dqn.py                 # DQN variants (DQN, Double DQN, Dueling DQN)
│   ├── policy_gradient.py     # Policy gradient methods (REINFORCE, AC, A2C)
│   ├── advanced_pg.py         # Advanced PG methods (A3C, PPO, TRPO)
│   └── sac.py                 # Soft Actor-Critic
├── common/                    # Shared components
│   ├── __init__.py
│   ├── networks.py            # Neural network architectures
│   ├── replay_buffer.py       # Experience replay implementations
│   ├── env_wrappers.py        # Atari environment wrappers
│   ├── utils.py               # Utility functions and classes
│   └── logger.py              # Enhanced TensorBoard logger
├── configs/                   # Hydra configuration files
│   ├── algorithm/             # Algorithm-specific configs
│   │   ├── dqn.yaml          # DQN hyperparameters
│   │   └── ppo.yaml          # PPO hyperparameters
│   ├── env/                   # Environment configs
│   │   └── atari.yaml        # Atari environment settings
│   └── trainer.yaml           # Main training configuration
├── tests/                     # Test scripts
│   ├── __init__.py
│   ├── test_algorithms.py     # Algorithm verification tests
│   └── test_hydra_tensorboard.py  # Hydra & TensorBoard tests
├── train.py                   # Unified training script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd rl-algorithms
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python tests/test_algorithms.py
```

## 🎮 Algorithms Implemented

### Value-Based Methods

#### 1. DQN (Deep Q-Network)
- **Paper**: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- **Key Features**: Experience replay, target networks, ε-greedy exploration
- **Use Case**: Discrete action spaces, stable value learning

#### 2. Double DQN
- **Paper**: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2016)
- **Key Features**: Reduces overestimation bias using two Q-networks
- **Improvement**: More stable Q-value estimates

#### 3. Dueling DQN
- **Paper**: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
- **Key Features**: Separate value and advantage streams
- **Improvement**: Better value function approximation

### Policy-Based Methods

#### 4. REINFORCE
- **Paper**: "Simple statistical gradient-following algorithms for connectionist reinforcement learning" (Williams, 1992)
- **Key Features**: Monte Carlo policy gradient, episode-based learning
- **Use Case**: Simple policy optimization, high variance

#### 5. Actor-Critic
- **Paper**: "Actor-Critic Algorithms" (Konda & Tsitsiklis, 2000)
- **Key Features**: Combined policy and value function learning
- **Improvement**: Reduced variance compared to REINFORCE

#### 6. A2C (Advantage Actor-Critic)
- **Paper**: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
- **Key Features**: Advantage function, entropy regularization
- **Improvement**: More stable learning, shared network

### Advanced Policy Gradient Methods

#### 7. A3C (Asynchronous Advantage Actor-Critic)
- **Paper**: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
- **Key Features**: Asynchronous parallel training, global network
- **Improvement**: Faster training, exploration diversity

#### 8. PPO (Proximal Policy Optimization)
- **Paper**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **Key Features**: Clipped objective, multiple epochs per batch
- **Improvement**: Stable policy updates, easy to tune

#### 9. TRPO (Trust Region Policy Optimization)
- **Paper**: "Trust Region Policy Optimization" (Schulman et al., 2015)
- **Key Features**: Constrained policy updates, conjugate gradient
- **Improvement**: Theoretical guarantees for policy improvement

### Maximum Entropy Methods

#### 10. SAC (Soft Actor-Critic)
- **Paper**: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (Haarnoja et al., 2018)
- **Key Features**: Maximum entropy objective, automatic temperature tuning
- **Improvement**: Sample efficiency, robust exploration

## 🏃‍♂️ Quick Start

### Training with Hydra Configuration (Recommended)

The modern way to train algorithms using Hydra configs with TensorBoard logging:

```bash
# Train DQN on Pong with default settings
python train.py algorithm=dqn env_name=PongNoFrameskip-v4

# Train PPO on Breakout with custom episodes
python train.py algorithm=ppo env_name=BreakoutNoFrameskip-v4 episodes=2000

# Override multiple parameters
python train.py algorithm=dqn env_name=PongNoFrameskip-v4 episodes=1000 seed=123 eval_freq=50
```

### TensorBoard Visualization

Start TensorBoard to monitor training in real-time:

```bash
# Start TensorBoard (after starting training)
tensorboard --logdir=runs

# View in browser at http://localhost:6006
```

The TensorBoard interface will show:
- **Training metrics**: Loss, rewards, episode length
- **Evaluation results**: Mean/std evaluation rewards
- **Model architecture**: Network graph visualization
- **Hyperparameters**: Complete configuration tracking
- **Weight histograms**: Network parameter distributions

### Training an Algorithm (Legacy Mode)

For backward compatibility, you can still use the traditional argparse interface:

```bash
# Train DQN on Pong (legacy mode)
python train.py --use_argparse --algorithm dqn --env PongNoFrameskip-v4 --episodes 1000

# Train PPO on Breakout (legacy mode)
python train.py --use_argparse --algorithm ppo --env BreakoutNoFrameskip-v4 --episodes 2000

# Train with custom parameters (legacy mode)
python train.py --use_argparse --algorithm a2c --env PongNoFrameskip-v4 --episodes 1500 --eval_freq 50 --save_freq 100
```

## ⚙️ Configuration Management

### Hydra Configuration Structure

The configuration is organized hierarchically:

```yaml
# configs/trainer.yaml (main config)
defaults:
  - algorithm: dqn
  - env: atari

seed: 42
episodes: 1000
log_dir: "runs"
tensorboard:
  log_interval: 10
  log_histograms: true
  log_model_graph: true
```

### Algorithm-Specific Configs

Each algorithm has its own configuration file with hyperparameters:

```yaml
# configs/algorithm/dqn.yaml
algorithm_name: "dqn"
dqn:
  learning_rate: 0.0001
  batch_size: 32
  buffer_size: 100000
  target_update_freq: 1000
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.01
```

### Environment Configs

Environment settings are also configurable:

```yaml
# configs/env/atari.yaml
env_name: "PongNoFrameskip-v4"
env:
  frame_skip: 4
  screen_size: 84
  clip_rewards: true
  frame_stack: 4
```

### Advanced Usage

#### Multi-Run Experiments

Run multiple experiments with different hyperparameters:

```bash
# Run with multiple learning rates
python train.py -m dqn.learning_rate=1e-3,1e-4,5e-4

# Run on multiple environments
python train.py -m env_name=PongNoFrameskip-v4,BreakoutNoFrameskip-v4

# Grid search
python train.py -m dqn.learning_rate=1e-3,1e-4 episodes=500,1000
```

### Training Parameters (Legacy Mode)

For backward compatibility with `--use_argparse`:

- `--algorithm`: Choose from {dqn, double_dqn, dueling_dqn, dueling_double_dqn, reinforce, actor_critic, a2c, ppo, trpo, sac}
- `--env`: Atari environment name (e.g., PongNoFrameskip-v4, BreakoutNoFrameskip-v4)
- `--episodes`: Number of training episodes (default: 1000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--eval_freq`: Evaluation frequency in episodes (default: 100)
- `--save_dir`: Directory to save models (default: models)
- `--log_dir`: Directory to save logs and plots (default: logs)

### Example Training Commands

```bash
# Modern Hydra approach
python train.py algorithm=dqn env_name=PongNoFrameskip-v4 episodes=100

# With TensorBoard monitoring
python train.py algorithm=ppo env_name=BreakoutNoFrameskip-v4 episodes=2000 eval_freq=100

# Legacy approach with argparse
python train.py --use_argparse --algorithm dqn --env PongNoFrameskip-v4 --episodes 100

# Compare algorithms (modern approach)
python train.py algorithm=dqn env_name=PongNoFrameskip-v4 episodes=1000 seed=42
python train.py algorithm=ppo env_name=PongNoFrameskip-v4 episodes=1000 seed=42
```

## 🧪 Testing

Run the test suite to verify all algorithms and new features work correctly:

```bash
# Test all algorithms
python tests/test_algorithms.py

# Test Hydra and TensorBoard integration
python tests/test_hydra_tensorboard.py

# Quick integration test (10 steps training)
python train.py algorithm=dqn env_name=PongNoFrameskip-v4 episodes=1 max_steps=10
```

The test suite will:
- ✅ Test environment creation and basic operations
- ✅ Test shared components (networks, replay buffers, utilities)
- ✅ Test each algorithm's instantiation and basic operations
- ✅ Test model saving and loading for each algorithm
- ✅ Test Hydra configuration loading and composition
- ✅ Test TensorBoard logger functionality
- ✅ Test backward compatibility with argparse
- ✅ Verify TensorBoard event file generation

## 📊 Expected Performance

Performance varies by algorithm and environment. Here are typical results after sufficient training:

| Algorithm | Pong | Breakout | Training Time | Sample Efficiency |
|-----------|------|----------|---------------|-------------------|
| DQN | ~18-20 | ~400-500 | Medium | Medium |
| Double DQN | ~19-21 | ~450-550 | Medium | Medium+ |
| Dueling DQN | ~19-21 | ~500-600 | Medium | Medium+ |
| PPO | ~20-21 | ~400-500 | Fast | High |
| A2C | ~18-20 | ~350-450 | Fast | Medium |
| SAC | ~19-21 | ~450-550 | Medium | High |

*Note: Results may vary based on hyperparameters, training duration, and random seeds.*

## 🔧 Customization

### Adding New Algorithms

1. Create a new file in the `algorithms/` directory
2. Implement the algorithm class with required methods:
   - `__init__()`: Initialize networks and hyperparameters
   - `select_action()`: Action selection logic
   - `train_step()` or `update()`: Learning update
   - `save_model()` and `load_model()`: Model persistence

3. Add imports to `train.py` and update the `create_agent()` function

### Modifying Hyperparameters

Each algorithm has configurable hyperparameters in its constructor. You can modify them when creating agents:

```python
from algorithms.dqn import DQN

# Custom DQN with different hyperparameters
agent = DQN(
    state_shape=(4, 84, 84),
    num_actions=6,
    lr=1e-4,  # Lower learning rate
    gamma=0.95,  # Different discount factor
    epsilon_decay=500000,  # Faster epsilon decay
    memory_size=500000,  # Smaller replay buffer
    batch_size=64  # Larger batch size
)
```

### Using Different Environments

The library works with any Atari environment. Popular choices include:

- **PongNoFrameskip-v4**: Simple two-player game, good for testing
- **BreakoutNoFrameskip-v4**: Classic brick-breaking game
- **SpaceInvadersNoFrameskip-v4**: Space shooter game
- **QbertNoFrameskip-v4**: Platform puzzle game
- **SeaquestNoFrameskip-v4**: Underwater adventure game

## 📈 Monitoring Training

Training progress is automatically logged and can be monitored through:

1. **Console Output**: Real-time episode rewards and evaluation results
2. **Saved Metrics**: NumPy files with detailed training metrics
3. **Training Plots**: Automatically generated reward curves
4. **Model Checkpoints**: Periodic model saves for resuming training

### Log Files Structure

```
logs/
├── dqn_metrics.npy           # Training metrics
├── dqn_training_curves.png   # Reward plots
└── ...

models/
├── dqn_best.pth             # Best performing model
├── dqn_final.pth            # Final model
├── dqn_episode_200.pth      # Periodic checkpoints
└── ...
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in algorithm constructors
   - Use smaller networks or replay buffers
   - Add `torch.cuda.empty_cache()` periodically

2. **Slow Training**
   - Ensure CUDA is available and being used
   - Consider using smaller environments for testing
   - Reduce evaluation frequency during training

3. **Poor Performance**
   - Check hyperparameters match literature recommendations
   - Ensure environment preprocessing is correct
   - Try different random seeds
   - Increase training duration

4. **Import Errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path includes the project directory

### Environment Issues

If you encounter issues with Atari environments:

```bash
# Install additional dependencies
pip install gymnasium[atari]
pip install ale-py

# Accept ROM license
ale-import-roms --import-from-pkg atari_py.atari_roms
```

## 📚 References

1. Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.
2. Van Hasselt, H., Guez, A., & Silver, D. "Deep reinforcement learning with double q-learning." AAAI 2016.
3. Wang, Z., et al. "Dueling network architectures for deep reinforcement learning." ICML 2016.
4. Williams, R. J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
5. Mnih, V., et al. "Asynchronous methods for deep reinforcement learning." ICML 2016.
6. Schulman, J., et al. "Trust region policy optimization." ICML 2015.
7. Schulman, J., et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
8. Haarnoja, T., et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." ICML 2018.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new algorithms
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Gym/Gymnasium team for the Atari environments
- PyTorch team for the deep learning framework
- Original algorithm authors for their groundbreaking research
- Open source RL community for implementations and insights

---

**Happy Learning! 🎯**