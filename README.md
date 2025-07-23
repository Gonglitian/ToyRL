# ToyRL: Deep Reinforcement Learning Algorithms Library

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](docs/_build/html/index.html)

A comprehensive implementation of classic deep reinforcement learning algorithms tested on Atari environments. This library provides clean, modular implementations of fundamental RL algorithms with shared components for easy experimentation and comparison.

## 🚀 Features

- **10 Classic RL Algorithms**: DQN variants, Policy Gradient methods, Actor-Critic algorithms
- **Modern Python Packaging**: Installable via `pip install -e .`
- **Modular Design**: Clean, well-documented code with shared utilities
- **Atari Ready**: Pre-configured environment wrappers for Atari 2600 games
- **Comprehensive Documentation**: Auto-generated API docs with Sphinx
- **Easy Installation**: Modern Python packaging with proper dependency management
- **Extensible**: Easy to modify and extend for new algorithms

## 📦 Installation

### From Source (Recommended)

```bash
git clone https://github.com/yourusername/toyrl.git
cd toyrl
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/toyrl.git
cd toyrl
pip install -e ".[dev,docs]"
```

## 🎯 Quick Start

```python
import toyrl
from toyrl.algorithms import SAC, DQN
from toyrl.common import make_atari_env

# Create environment
env = make_atari_env('PongNoFrameskip-v4')

# Initialize algorithm  
agent = DQN(
    state_dim=env.observation_space.shape,
    action_dim=env.action_space.n,
    lr=1e-4
)

# Train
agent.train(env, num_episodes=1000)
```

## 🧠 Algorithms Implemented

### Value-Based Methods
- **DQN** (Deep Q-Network) - `toyrl.algorithms.DQN`
- **Double DQN** - `toyrl.algorithms.DoubleDQN`  
- **Dueling DQN** - `toyrl.algorithms.DuelingDQN`

### Policy Gradient Methods
- **REINFORCE** - `toyrl.algorithms.REINFORCE`
- **Actor-Critic** - `toyrl.algorithms.ActorCritic`
- **A2C** (Advantage Actor-Critic) - `toyrl.algorithms.A2C`

### Advanced Policy Methods
- **A3C** (Asynchronous Advantage Actor-Critic) - `toyrl.algorithms.A3C`
- **PPO** (Proximal Policy Optimization) - `toyrl.algorithms.PPO`
- **TRPO** (Trust Region Policy Optimization) - `toyrl.algorithms.TRPO`

### Actor-Critic Methods
- **SAC** (Soft Actor-Critic) - `toyrl.algorithms.SAC`

## 📁 Project Structure

```
ToyRL/
├── src/toyrl/           # Main package
│   ├── __init__.py      # Package initialization  
│   ├── algorithms/      # RL algorithms
│   │   ├── __init__.py
│   │   ├── dqn.py       # DQN variants
│   │   ├── sac.py       # Soft Actor-Critic
│   │   ├── policy_gradient.py  # REINFORCE, A2C, Actor-Critic
│   │   └── advanced_pg.py      # A3C, PPO, TRPO
│   └── common/          # Shared utilities
│       ├── __init__.py
│       ├── networks.py  # Neural network architectures
│       ├── replay_buffer.py  # Experience replay
│       ├── env_wrappers.py   # Environment preprocessing
│       ├── utils.py     # Helper functions
│       ├── logger.py    # TensorBoard logging
│       └── gif_wrapper.py    # GIF recording
├── docs/                # Documentation
│   ├── conf.py          # Sphinx configuration
│   ├── index.rst        # Main documentation
│   └── _build/html/     # Generated HTML docs
├── examples/            # Usage examples
├── tests/               # Unit tests
├── pyproject.toml       # Project configuration
├── Makefile            # Build commands
├── mkdocs.yml          # Alternative docs (MkDocs)
├── LICENSE             # MIT License
└── README.md           # This file
```

## 📚 Documentation

### Building Documentation

Build the Sphinx documentation locally:

```bash
make docs
```

Then open `docs/_build/html/index.html` in your browser.

### Alternative: MkDocs

This project also supports MkDocs as an alternative documentation system:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve
```

### API Documentation

The API documentation is automatically generated from docstrings and includes:

- Complete algorithm implementations
- Network architectures
- Utility functions
- Environment wrappers
- Examples and usage patterns

## 🛠 Development

### Installing for Development

```bash
git clone https://github.com/yourusername/toyrl.git
cd toyrl
pip install -e ".[dev,docs]"
```

### Running Tests

```bash
python -c "import toyrl; print('ToyRL version:', toyrl.__version__)"
python -c "from toyrl.algorithms import SAC, DQN; print('Algorithms imported successfully')"
```

### Building Documentation

```bash
make docs          # Build Sphinx docs
make docs-serve    # Serve docs locally
make docs-clean    # Clean documentation build
```

### Cleaning Build Artifacts

```bash
make clean
```

## 📋 Requirements

- Python 3.9+
- PyTorch 1.9+
- Gymnasium
- NumPy
- Matplotlib
- TensorBoard

See `pyproject.toml` for complete dependency list.

## 🎮 Usage Examples

### Basic Algorithm Training

```python
import toyrl
from toyrl.algorithms import DQN, PPO, SAC
from toyrl.common import make_atari_env, ReplayBuffer

# Create environment
env = make_atari_env('PongNoFrameskip-v4')

# DQN Example
dqn_agent = DQN(
    state_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
    lr=1e-4,
    epsilon_decay=500000
)

# PPO Example  
ppo_agent = PPO(
    state_dim=env.observation_space.shape,
    action_dim=env.action_space.n,
    lr=3e-4
)

# SAC Example (for continuous control)
sac_agent = SAC(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0], 
    lr=3e-4
)
```

### Using Shared Components

```python
from toyrl.common import (
    ReplayBuffer, 
    DQNNetwork, 
    ContinuousActorNetwork,
    get_device,
    make_atari_env
)

# Create replay buffer
buffer = ReplayBuffer(capacity=100000)

# Create networks
device = get_device()
q_network = DQNNetwork(input_shape=(4, 84, 84), num_actions=6).to(device)
actor_network = ContinuousActorNetwork(state_dim=8, action_dim=2).to(device)

# Environment with preprocessing
env = make_atari_env('BreakoutNoFrameskip-v4', frame_stack=4)
```

### Custom Training Loop

```python
import toyrl
from toyrl.algorithms import DQN
from toyrl.common import make_atari_env

# Setup
env = make_atari_env('PongNoFrameskip-v4')
agent = DQN(state_shape=env.observation_space.shape, num_actions=env.action_space.n)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    episode_reward = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, done)
        agent.update()
        
        state = next_state
        episode_reward += reward
        
        if done or truncated:
            break
    
    print(f"Episode {episode}: Reward = {episode_reward}")
```

## 🤖 GitHub Actions

The project includes a GitHub Actions workflow for automatic documentation deployment:

```yaml
# .github/workflows/docs.yml
name: Build and Deploy Documentation

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -e ".[docs]"
    - name: Build docs
      run: |
        make docs
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all public functions and classes
- Include tests for new features
- Update documentation as needed
- Use Google-style docstrings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **GitHub**: [https://github.com/yourusername/toyrl](https://github.com/yourusername/toyrl)
- **Documentation**: [API Documentation](docs/_build/html/index.html)

## 🙏 Acknowledgments

- OpenAI Gym/Gymnasium for the environment interface
- PyTorch team for the deep learning framework
- The reinforcement learning community for algorithm implementations and insights
- Sphinx team for the documentation framework
- All contributors to this project

## 📚 References

1. Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.
2. Van Hasselt, H., Guez, A., & Silver, D. "Deep reinforcement learning with double q-learning." AAAI 2016.
3. Wang, Z., et al. "Dueling network architectures for deep reinforcement learning." ICML 2016.
4. Williams, R. J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
5. Mnih, V., et al. "Asynchronous methods for deep reinforcement learning." ICML 2016.
6. Schulman, J., et al. "Trust region policy optimization." ICML 2015.
7. Schulman, J., et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
8. Haarnoja, T., et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." ICML 2018.

---

**Happy Learning! 🎯**
