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
            [100, 100]), theta=0, vm=0, vl=0, vr=0)
        self.target = Target('target_0', GREEN, 20, np.array([300, 300]))
        self.obstacle = Obstacle('obstacle_0', RED, 20, np.array([200, 200]))
        self.add_objects(self.robot, self.target, self.obstacle)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(

    def reset(self):
        self.robot.reset()
        for obj in self.obj_dict.values():
            obj.reset()
        return self.get_state()

    def get_state(self):
        return ...  # todo

    def step(self, action):
        self.handle_event()
        self.action_parser(action)
        for obj in self.obj_dict.values():
            obj.update_state()
        reward=...  # todo
        done=False
        return self.get_state(), reward, done, None

    def action_parser(self, action):
        robot: TwoWheelsRobot=self.obj_dict['robot_0']
        # 更新速度
        a=robot.a
        if action == 0:
            robot.vl += a*self.DELTA_T
            robot.vr += a*self.DELTA_T
        if action == 1:
            robot.vl -= a*self.DELTA_T
            robot.vr -= a*self.DELTA_T
        if action == 2:
            robot.vr += a*self.DELTA_T
        if action == 3:
            robot.vl += a*self.DELTA_T
        if action == 4:
            pass

    def handle_event(self):
        # 处理外部输入
        keys=pygame.key.get_pressed()
        if keys[pygame.K_1]:
            # 创建一个新的障碍物
            name=self.infer_name('obstacle')
            obstacle=Obstacle(name, RED, 20, np.array(
                [np.random.randint(self.screen.get_width()), np.random.randint(self.screen.get_height())]))  # 你需要自己定义Obstacle类和它的初始化参数
            self.add_objects(obstacle)

    def infer_name(self, base_name):
        i=0
        while f'{base_name}_{i}' in self.obj_dict:
            i += 1
        name=f'{base_name}_{i}' if i > 0 else base_name
        return name
