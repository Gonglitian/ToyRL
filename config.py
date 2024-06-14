class Config:
    def __init__(
        self,
        env_name="CartPole-v1",
        n_episodes=1000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        gamma=0.99,
        lr=0.0005,
        seed=0,
        buffer_size=int(1e5),
        batch_size=64,
        update_every=4,
        tau=1e-3,
        n_hidden=64,
        n_hidden_layers=2,
        device="cpu",
        double_dqn=False,
        dueling_dqn=False,
        prioritized_replay=False,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_decay=0.999,
        model_path="checkpoint.pth",
    ):
        # Environment
        self.env_name = env_name
        # Training
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.lr = lr
        self.seed = seed
        # Memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        # Network
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        # Device
        self.device = device
        # DQN Improvements
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.prioritized_replay = prioritized_replay
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay = beta_decay
        # Model
        self.model_path = model_path


class DQNConfig:

    def __init__(
        self,
        pool_max_size=10_000,
        pool_batch_size=32,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        log_interval=100_000,
    ):
        # env
        self.pool_max_size = pool_max_size
        self.pool_batch_size = pool_batch_size
        # exploration
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        # training
        self.log_interval = 100_000
