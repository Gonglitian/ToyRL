import numpy as np
from pygame_env import *
from pool import *
import torch
from config import *


class Agent:
    def __init__(self, env: PygameEnv, model: torch.nn.Module, config: DQNConfig):
        self.env = env
        self.pool = Pool()
        self.model = model

        self.gamma = 0.95  # discount rate
        # exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # learning
        self.learning_rate = 0.001

    def play(self, random=False, render=False):
        reward_sum = 0
        done = False
        s = self.env.reset()
        while not done:
            if random:
                a = self.env.action_space.sample()
            else:
                if np.random.rand() <= self.epsilon:
                    a = self.env.action_space.sample()
                else:
                    a = (
                        self.model(
                            torch.FloatTensor(s).reshape(1, self.env.action_space.n)
                        )
                        .argmax()
                        .item()
                    )
            ns, r, done, info = self.env.step(a)
            reward_sum += r
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.pool.add((s, a, r, ns, done))
            s = ns
            if render:
                self.env.render()
        return reward_sum

    def get_experience_tensor(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # todo test
        experience_batch = self.pool.sample()
        # 使用zip函数将经验分解为单独的列表
        batch_s, batch_a, batch_r, batch_ns, batch_done = zip(*experience_batch)

        # 直接将列表转换为PyTorch张量
        batch_s = torch.tensor(batch_s, dtype=torch.float32)
        batch_a = torch.tensor(batch_a, dtype=torch.long)
        batch_r = torch.tensor(batch_r, dtype=torch.float32)
        batch_ns = torch.tensor(batch_ns, dtype=torch.float32)
        batch_done = torch.tensor(batch_done, dtype=torch.bool)

        return batch_s, batch_a, batch_r, batch_ns, batch_done

    def train(self):
        return NotImplementedError


class DQNAgent(Agent):
    def __init__(self, env: PygameEnv, model: torch.nn.Module, config: DQNConfig):
        super().__init__(env, model)
        self.env = env
        self.pool = Pool(
            max_size=config.pool_max_size, batch_size=config.pool_batch_size
        )
        self.model = model
        self.config = config
        # training params
        self.n_step = 0
        self.last_log_step = 0

    def train(self):
        batch_s, batch_a, batch_r, batch_ns, batch_done = self.get_experience_tensor()
        # 计算Q(s, a)
        q_s = self.model(batch_s).gather(1, batch_a.unsqueeze(1)).squeeze(1)

        # 计算Q(s', a')
        q_ns = self.model(batch_ns).max(1)[0].detach()

        # 计算Q(s, a)的目标值
        q_target = batch_r + self.gamma * q_ns * (~batch_done)

        # 计算损失
        loss = torch.nn.functional.mse_loss(q_s, q_target)

        # 反向传播
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
