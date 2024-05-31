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
# 设置标题
robot_image = pygame.image.load("robot.png")
original_width, original_height = robot_image.get_size()
scaled_width = original_width // 8
scaled_height = original_height // 8
robot_image = pygame.transform.scale(
    robot_image, (scaled_width, scaled_height))
robot_image = robot_image.convert_alpha()  # 转换图片以提高渲染效率并保留透明度

robot_image_rect = robot_image.get_rect()

DELTA_T = 0.5
MAX_V = 5


class Object:
    def __init__(self, name, color, position, orientation_angle) -> None:
        self.name = name
        self.env: PygameEnv = None
        self.screen: Surface = None
        self.color = color
        self.positon = position
        self.orientation_angle = orientation_angle

    def show(self):
        ...

    def state_update(self):
        ...


class CircleObject(Object):
    def __init__(self, name, color, pos, radius) -> None:
        super().__init__(name, color, pos)
        self.radius = radius

    def show(self):
        pygame.draw.circle(self.env.screen, self.color,
                           self.positon, self.radius)


class RectObject(Object):
    def __init__(self, name, color, pos, width, height) -> None:
        super().__init__(name, color, pos, 0)
        self.width = width
        self.height = height

    def show(self):
        pygame.draw.rect(self.env.screen, self.color,
                         (self.positon[0], self.positon[1], self.width, self.height))


class Robot(CircleObject):
    def __init__(self, name, color, pos, radius) -> None:
        super().__init__(name, color, pos, radius)
        self.vm = 5
        self.vl = 0
        self.vr = 0
        self.vc = (self.vl + self.vr) / 2
        self.w = 0
        self.a = 1

        self.theta = 0
        self.image_theta = 0

    def update_state(self):
        # 限制速度不超过最大速度
        if self.vl >= self.vm:
            self.vl = self.vm
        if self.vl < 0:
            self.vl = 0
        if self.vr >= self.vm:
            self.vr = self.vm
        if self.vr < 0:
            self.vr = 0

        self.vc = (self.vl + self.vr) / 2
        self.w = (self.vr - self.vl) / self.radius

        # 更新位置
        vc, theta = self.vc, self.theta
        self.positon[0] += vc * math.cos(theta) * DELTA_T
        self.positon[1] -= vc * math.sin(theta) * DELTA_T

        # 限制新位置在屏幕内
        if self.positon[0] <= 0:
            self.positon[0] = 0
        if self.positon[0] >= self.screen_width:
            self.positon[0] = self.screen_width
        if self.positon[1] <= 0:
            self.positon[1] = 0
        if self.positon[1] >= self.screen_hight:
            self.positon[1] = self.screen_hight

        self.theta += self.w * DELTA_T
        self.theta = normalize_angle(self.theta)

    def show(self):
        # 旋转图片并更新其矩形区域的中心点
        # self.image_theta = math.degrees(self.theta) - 90
        # rotated_image = pygame.transform.rotate(robot_image, self.image_theta)
        # robot_image_rect.center = self.pos
        # rotated_image_rect = rotated_image.get_rect(center=robot_image_rect.center)
        # self.screen.blit(rotated_image, rotated_image_rect.topleft)
        # pygame.draw.rect(screen, robot.color,
        #                  (robot.pos[0], robot.pos[1], robot.width, robot.height))
        # pygame.draw.line(screen, RED,
        #                  self.pos, start_head, 1)
        # 从圆心画一条线到机器人的前方，表示机器人的朝向
        start_head = self.pos + self.radius * \
            np.array([math.cos(self.theta), -math.sin(self.theta)])

        pygame.draw.circle(self.screen, self.color, self.pos, self.radius)
        pygame.draw.line(self.screen, RED,
                         self.pos, start_head, 1)
        # 目标点的属性


# def show_data(name, pos, digit=3, *content):
#     content_str = ""
#     for x in content:
#         content_str += str(round(x, digit)) + " "
#     name = info_font.render(f"{name}: ({content_str})", True, RED)
#     screen.blit(name, pos)  # 在机器人坐标下方绘制目标点坐标


class Obstacle(CircleObject):
    def __init__(self) -> None:
        self.theta = 0
        self.pos = np.array([0, 0])
        self.radius = 20
        self.color = RED

        self.v = 1
        self.w = 0.1

    def update_position(self):
        # 绕某个圆心做圆周运动
        self.pos[0] = 200 + 50 * math.cos(self.theta)
        self.pos[1] = 200 + 50 * math.sin(self.theta)
        self.theta += self.w * DELTA_T

    def show(self):
        pygame.draw.circle(self.screen, self.color, self.pos, self.radius)


class Target(CircleObject):
    def __init__(self, name, color, position,orientation_angle) -> None:
        super().__init__(name, color, position)
        self.radius = 20


def normalize_angle(angle):
    # 将角度规范化到 -2π 到 +2π 范围内
    return (angle + 4 * math.pi) % (2 * math.pi)
