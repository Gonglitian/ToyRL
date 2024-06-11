import types
import pygame
from pygame_env import PygameEnv
from object import TwoWheelsRobot, Obstacle, Target
import numpy as np
from gymnasium import spaces
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class RobotEnv(PygameEnv):
    def __init__(self, screen_size) -> None:
        super().__init__(screen_size)
        self.robot = TwoWheelsRobot('robot_0', BLUE, 20, np.array(
            [100, 100]), theta=0, vm=5, vl=5, vr=5)
        self.target = Target('target_0', GREEN, 20, np.array([300, 300]))
        self.obstacle = Obstacle('obstacle_0', RED, 20, np.array([200, 200]))
        self.add_objects(self.robot, self.target, self.obstacle)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.screen.get_width(), self.screen.get_height(), 3), dtype=np.uint8)

    def reset(self):
        self.robot.position = np.array(
            [np.random.randint(self.screen.get_width()), np.random.randint(self.screen.get_height())])
        self.target.position = np.array(
            [np.random.randint(self.screen.get_width()), np.random.randint(self.screen.get_height())])
        self.obstacle.position = np.array(
            [np.random.randint(self.screen.get_width()), np.random.randint(self.screen.get_height())])
        return self.get_state()

    def get_state(self):
        return np.array(
            [
                self.robot.position[0],
                self.robot.position[1],
                self.target.position[0],
                self.target.position[1],
                self.robot.vl,
                self.robot.vr,
                self.robot.theta,
                self.robot.target_distance,
            ]
        ).astype(np.float32)

    def step(self, action):
        self.handle_event()
        self.action_parser(action)
        for obj in self.obj_dict.values():
            obj.update_state()
        reward = ...  # todo
        done = False
        return self.get_state(), reward, done, None

    def action_parser(self, action):
        # 更新速度
        a = self.robot.a
        if action == 0:
            self.robot.vl += a*self.DELTA_T
            self.robot.vr += a*self.DELTA_T
        if action == 1:
            self.robot.vl -= a*self.DELTA_T
            self.robot.vr -= a*self.DELTA_T
        if action == 2:
            self.robot.vr += a*self.DELTA_T
        if action == 3:
            self.robot.vl += a*self.DELTA_T
        if action == 4:
            pass

    def handle_event(self):
        # 处理外部输入
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            # 创建一个新的障碍物
            name = self.infer_name('obstacle')
            obstacle = Obstacle(name, RED, 20, np.array(
                [np.random.randint(self.screen.get_width()), np.random.randint(self.screen.get_height())]))  # 你需要自己定义Obstacle类和它的初始化参数
            self.add_objects(obstacle)

    def infer_name(self, base_name):
        i = 0
        while f'{base_name}_{i}' in self.obj_dict:
            i += 1
        name = f'{base_name}_{i}' if i > 0 else base_name
        return name
