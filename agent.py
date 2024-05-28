import numpy as np
from pygame_env import *
from pool import *


class Agent():
    def __init__(self, env: PygameEnv, model, pool: Pool):
        self.env = env
        self.pool = pool
        self.model = model

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

    def play(self, random=False, render=False):
        done = False
        s = self.env.reset()
        while not done:
            if random:
                a = self.env.action_space.sample()
            else:
                if np.random.rand() <= self.epsilon:
                    a = self.env.action_space.sample()
                else:
                    a = np.argmax(self.model.predict(s))
            ns, r, done, _ = self.env.step(a)
            self.pool.add((s, a, r, ns, done))
            s = ns
            if render:
                self.env.render()
            



    def train(self):
        ...

class DQNAgent(Agent):
    def __init__(self, env: PygameEnv, model, pool: Pool):
        super().__init__(env, model, pool)
        self.env = env
        self.pool = pool
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def train(self):
        experience_batch = self.pool.sample()
        batch_s = []
        batch_a = []
        batch_r = []
        batch_ns = []
        batch_done = []
        # convert to tensor'
        for experience in experience_batch:
            batch_s.append(experience[0])
            batch_a.append(experience[1])
            batch_r.append(experience[2])
            batch_ns.append(experience[3])
            batch_done.append(experience[4])

        batch_s = np.array(batch_s)
        batch_a = np.array(batch_a)
        batch_r = np.array(batch_r)
        batch_ns = np.array(batch_ns)
        batch_done = np.array(batch_done)

        # train in pytorch
        target = batch_r + self.gamma * \
            np.amax(self.model.predict(batch_ns), axis=1)
        target[batch_done] = batch_r[batch_done]
