import numpy as np
import math
import random
import pygame
from pygame_env import PygameEnv
from pygame import Surface
# 颜色定义
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
# 设置标题
# robot_image = pygame.image.load("robot.png")
# original_width, original_height = robot_image.get_size()
# scaled_width = original_width // 8
# scaled_height = original_height // 8
# robot_image = pygame.transform.scale(
#     robot_image, (scaled_width, scaled_height))
# robot_image = robot_image.convert_alpha()  # 转换图片以提高渲染效率并保留透明度

# robot_image_rect = robot_image.get_rect()

DELTA_T = 0.5
MAX_V = 5


class Object:
    def __init__(self, name, color, position=np.array([0, 0]), theta=0) -> None:
        self.env: PygameEnv = None
        self.screen: Surface = None
        self.name = name
        self.color = color
        self.position = position
        self.theta = theta

        self.a = 0
        self.v = 0
        self.w = 0

    def update_state(self):
        ...

    def show(self):
        ...


class CircleObject(Object):
    def __init__(self, name, color, radius, position=np.array([0, 0]), theta=0) -> None:
        super().__init__(name, color, position, theta)
        self.radius = radius

    def show(self):
        pygame.draw.circle(self.screen, self.color,
                           self.position, self.radius)


class RectObject(Object):
    def __init__(self, name, color, width, height, position, theta) -> None:
        super().__init__(name, color, position, theta)
        self.width = width
        self.height = height

    def show(self):
        pygame.draw.rect(self.env.screen, self.color,
                         (self.position[0], self.position[1], self.width, self.height))


class TwoWheelsRobot(CircleObject):
    def __init__(self, name, color, radius, position=np.array([0, 0]), theta=0, vm=0, vl=0, vr=0) -> None:
        super().__init__(name, color, radius, position, theta)
        self.vm = vm
        self.vl = vl
        self.vr = vr

    def update_state(self):
        # 限制速度不超过最大速度
        if self.vl > self.vm:
            self.vl = self.vm
        if self.vl < 0:
            self.vl = 0
        if self.vr > self.vm:
            self.vr = self.vm
        if self.vr < 0:
            self.vr = 0

        self.v = (self.vl + self.vr) / 2
        self.w = (self.vr - self.vl) / self.radius

        # 更新位置
        self.position[0] += self.v * math.cos(self.theta) * DELTA_T
        self.position[1] -= self.v * math.sin(self.theta) * DELTA_T

        # 限制新位置在屏幕内
        if self.position[0] <= 0:
            self.position[0] = 0
        if self.position[0] >= self.screen.get_width():
            self.position[0] = self.screen.get_width()
        if self.position[1] <= 0:
            self.position[1] = 0
        if self.position[1] >= self.screen.get_height():
            self.position[1] = self.screen.get_height()

        self.theta += self.w * DELTA_T
        self.theta = normalize_angle(self.theta)

    def show(self):
        # 从圆心画一条线到机器人的前方，表示机器人的朝向
        start_head = self.position + self.radius * \
            np.array([math.cos(self.theta), -
                     math.sin(self.theta)])

        pygame.draw.circle(self.screen, self.color, self.position, self.radius)
        pygame.draw.line(self.screen, RED,
                         self.position, start_head, 1)


class Obstacle(CircleObject):
    def __init__(self, name, color, radius, position=np.array([0, 0]), theta=0, v=0, a=0) -> None:
        super().__init__(name, color, radius, position, theta)
        self.w = 1

    def update_state(self):
        # 绕某个圆心做圆周运动
        self.position[0] = 200 + 50 * math.cos(self.theta)
        self.position[1] = 200 + 50 * math.sin(self.theta)
        self.theta += self.w * DELTA_T


class Target(CircleObject):
    def __init__(self, name, color, radius, position=np.array([0, 0]), theta=0, v=0, a=0) -> None:
        super().__init__(name, color, radius, position, theta)


def normalize_angle(angle):
    # 将角度规范化到 -2π 到 +2π 范围内
    return (angle + 4 * math.pi) % (2 * math.pi)
