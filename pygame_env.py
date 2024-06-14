import imageio
from stable_baselines3.common.env_checker import check_env
import pygame
import numpy as np
from numpy import ndarray
import math

# from robot import Robot, Obstacle, Target
import random
import gymnasium
from gymnasium import spaces
import numpy as np
from object import Object

# 颜色定义
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

STEPS_LIMIT = 300


class PygameEnv(gymnasium.Env):
    def __init__(self, screen_size: ndarray) -> None:
        super().__init__()
        # 初始化pygame
        self.screen_size = screen_size
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("PygameEnv")
        self.info_font = pygame.font.SysFont(None, 36)
        self.clock = pygame.time.Clock()  # 创建Clock对象
        self.DELTA_T = 0.5
        # 初始化对象字典
        self.obj_dict: dict[Object] = {}
        self.action_space = ...
        self.observation_space = ...

        # 环境运行变量
        self.steps_count = 0

    def add_objects(self, *objects: Object):
        for obj in objects:
            self.obj_dict[obj.name] = obj
            self.obj_dict[obj.name].env = self
            self.obj_dict[obj.name].screen = self.screen

    def delete_object(self, obj: Object):
        del self.obj_dict[obj.name]

    def reset(self):
        return NotImplementedError

    def step(self, action: int):
        self.handle_event()
        self.handle_action(action)
        self.update_state()
        self.steps_count += 1
        reward = self._get_reward()
        done = self._get_done()
        info = {}
        return self._get_state(), reward, done, info

    def update_state(self):
        for obj in self.obj_dict.values():
            obj.update_state()

    def _get_state(self):
        return NotImplementedError

    def _get_reward(self):
        return NotImplementedError

    def _get_done(self):
        return NotImplementedError

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        self.screen.fill((255, 255, 255))
        for obj in self.obj_dict.values():
            obj.show()
        self.clock.tick(30)
        pygame.display.flip()

    def handle_event(self):
        return NotImplementedError

    def handle_action(self, action):
        return NotImplementedError

    @staticmethod
    def circle_to_circle_collision(circle1, circle2) -> tuple[float, bool]:
        distance = np.linalg.norm(circle1.position - circle2.position)
        return distance, distance <= (circle1.radius + circle2.radius)


# check
# env = RobotEnv(screen_width=400, screen_height=400)
# check_env(env, warn=True)
