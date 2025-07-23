ToyRL: Deep Reinforcement Learning Algorithms Library
=====================================================

ToyRL is a comprehensive implementation of classic deep reinforcement learning algorithms 
tested on Atari environments. This library provides clean, modular implementations of 
fundamental RL algorithms with shared components for easy experimentation and comparison.

🚀 Features
-----------

- **10 Classic RL Algorithms**: DQN, Double DQN, Dueling DQN, REINFORCE, Actor-Critic, A2C, A3C, PPO, TRPO, SAC
- **Hydra Configuration System**: Modern configuration management with YAML files and command-line overrides
- **TensorBoard Integration**: Real-time visualization of training metrics, model graphs, and hyperparameters
- **Atari Environment Support**: Pre-configured wrappers for Atari games with standard preprocessing
- **Modular Design**: Shared components like replay buffers, neural networks, and utilities
- **Easy Training**: Unified training script that works with all algorithms

🛠 Installation
---------------

Install ToyRL in development mode:

.. code-block:: bash

   git clone https://github.com/yourusername/toyrl.git
   cd toyrl
   pip install -e .

📚 Quick Start
--------------

.. code-block:: python

   import toyrl
   from toyrl.algorithms import SAC
   from toyrl.common import make_atari_env

   # Create environment
   env = make_atari_env("PongNoFrameskip-v4")
   
   # Initialize SAC agent
   agent = SAC(
       state_dim=env.observation_space.shape,
       action_dim=env.action_space.n,
       lr_actor=3e-4,
       lr_critic=3e-4
   )
   
   # Train the agent
   agent.train(num_steps=100000)

📖 API Reference
----------------

.. autosummary::
   :toctree: api/
   :recursive:

   toyrl.algorithms
   toyrl.common

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules

