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
        # add objs here
        self.robot = TwoWheelsRobot(
            "robot_0", BLUE, 20, np.array([100, 100]), theta=0, vm=5, vl=0, vr=0, a=1
        )
        self.target = Target("target_0", GREEN, 20, np.array([300, 300]))
        # self.obstacle = Obstacle('obstacle_0', RED, 20, np.array([200, 200]))
        self.add_objects(self.robot, self.target)
        # define action and observation space
        self.action_space = spaces.Discrete(5)
        low = np.array(
            [
                -self.screen.get_width(),
                -self.screen.get_height(),
                -self.screen.get_width(),
                -self.screen.get_height(),
                0,
                0,
                -np.pi,
                0,
            ]
        )
        high = np.array(
            [
                self.screen.get_width(),
                self.screen.get_height(),
                self.screen.get_width(),
                self.screen.get_height(),
                self.robot.vm,
                self.robot.vm,
                np.pi,
                (self.screen.get_width() ** 2 +
                 self.screen.get_height() ** 2) ** 0.5,
            ]
        )
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

        self.steps_count = 0
        # reset
        self.reset()

    def reset(self):
        self.robot.position = np.array(
            [
                np.random.randint(self.screen.get_width()),
                np.random.randint(self.screen.get_height()),
            ]
        )
        self.robot.theta = np.random.uniform(-np.pi, np.pi)
        self.robot.vl = 0
        self.robot.vr = 0
        self.target.position = np.array(
            [
                np.random.randint(self.screen.get_width()),
                np.random.randint(self.screen.get_height()),
            ]
        )
        # hint you can add obstacles here
        # self.obstacle.position = np.array(
        # [np.random.randint(self.screen.get_width()), np.random.randint(self.screen.get_height())])
        return self._get_state()

    def _get_state(self):
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

    def _get_reward(self):
        self.distance, collision = self.circle_to_circle_collision(
            self.robot, self.target
        )
        distance_percentage = self.distance / self.screen.get_height() * 100

        if collision:
            # hint: add multiple targets here
            # self.target_count += 1
            # if self.target_count != len(self.target_list):
            #     self.current_target = self.target_list[self.target_count]
            return 100
        elif distance_percentage <= 25:
            return -0.1
        elif distance_percentage <= 50:
            return -0.25
        elif distance_percentage <= 75:
            return -0.5
        else:
            return -1

    def _get_done(self):
        if self.circle_to_circle_collision(self.robot, self.target)[1]:
            return True
        elif self.steps_count >= 300:
            return True
        return False

    def handle_action(self, action):
        # 更新速度
        a = self.robot.a
        if action == 0:
            self.robot.vl += a * self.DELTA_T
            self.robot.vr += a * self.DELTA_T
        if action == 1:
            self.robot.vl -= a * self.DELTA_T
            self.robot.vr -= a * self.DELTA_T
        if action == 2:
            self.robot.vr += a * self.DELTA_T
        if action == 3:
            self.robot.vl += a * self.DELTA_T
        if action == 4:
            pass

    def handle_event(self):
        # 处理外部输入
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            # 创建一个新的障碍物
            name = self.infer_name("obstacle")
            obstacle = Obstacle(
                name,
                RED,
                20,
                np.array(
                    [
                        np.random.randint(self.screen.get_width()),
                        np.random.randint(self.screen.get_height()),
                    ]
                ),
            )
            self.add_objects(obstacle)

    def infer_name(self, base_name):
        i = 0
        while f"{base_name}_{i}" in self.obj_dict:
            i += 1
        name = f"{base_name}_{i}" if i > 0 else base_name
        return name
