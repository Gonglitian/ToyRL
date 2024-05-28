import numpy as np
import math
import random
import pygame

# 颜色定义
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class Object:
    def __init__(self, name, env, color, position, orientation_angle) -> None:
        self.name = name
        self.env = env
        self.color = color
        self.positon = position
        self.orientation_angle = orientation_angle

    def show(self):
        ...

    def state_update(self):
        ...


class CircleObject(Object):
    def __init__(self, name, env, color, pos, radius) -> None:
        super().__init__(name, env, color, pos)
        self.radius = radius


class RectObject(Object):
    def __init__(self, name, env, color, pos, width, height) -> None:
        super().__init__(name, env, color, pos, 0)
        self.width = width
        self.height = height

    def show(self):
        pygame.draw.rect(self.env.screen, self.color,
                         (self.positon[0], self.positon[1], self.width, self.height))

    def update_state(self):
        # 更新速度
        if key == 0:
            self.vl += self.a*DELTA_T
            self.vr += self.a*DELTA_T
        if key == 1:
            self.vl -= self.a*DELTA_T
            self.vr -= self.a*DELTA_T
        if key == 2:
            self.vr += self.a*DELTA_T
        if key == 3:
            self.vl += self.a*DELTA_T
        if key == 4:
            pass

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
        self.pos[0] += vc * math.cos(theta) * DELTA_T
        self.pos[1] -= vc * math.sin(theta) * DELTA_T

        # 限制新位置在屏幕内
        if self.pos[0] <= 0:
            self.pos[0] = 0
        if self.pos[0] >= self.screen_width:
            self.pos[0] = self.screen_width
        if self.pos[1] <= 0:
            self.pos[1] = 0
        if self.pos[1] >= self.screen_hight:
            self.pos[1] = self.screen_hight

        self.theta += self.w * DELTA_T
        self.theta = normalize_angle(self.theta)


class Robot(CircleObject):
    def __init__(self, name, env, color, pos, radius) -> None:
        super().__init__(name, env, color, pos, radius)
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
        self.pos[0] += vc * math.cos(theta) * DELTA_T
        self.pos[1] -= vc * math.sin(theta) * DELTA_T

        # 限制新位置在屏幕内
        if self.pos[0] <= 0:
            self.pos[0] = 0
        if self.pos[0] >= self.screen_width:
            self.pos[0] = self.screen_width
        if self.pos[1] <= 0:
            self.pos[1] = 0
        if self.pos[1] >= self.screen_hight:
            self.pos[1] = self.screen_hight

        self.theta += self.w * DELTA_T
        self.theta = normalize_angle(self.theta)

    # def show(self):
    #     start_head = self.positon + self.radius * \
    #         np.array([math.cos(self.theta), -math.sin(self.theta)])
    #     pygame.draw.circle(self.env.screen, self.color,


# 设置标题
robot_image = pygame.image.load("robot.png")
original_width, original_height = robot_image.get_size()
scaled_width = original_width // 8
scaled_height = original_height // 8
robot_image = pygame.transform.scale(
    robot_image, (scaled_width, scaled_height))
robot_image = robot_image.convert_alpha()  # 转换图片以提高渲染效率并保留透明度

robot_image_rect = robot_image.get_rect()

DELTA_T = 1 / 10 * 5
MAX_V = 5
ACCELERATE = 1


class Robot:
    def __init__(self, screen) -> None:
        self.screen = screen
        self.screen_width, self.screen_hight = screen.get_size()
        self.radius = 20
        self.color = BLUE
        self.vm = MAX_V
        self.vl = 0
        self.vr = 0
        self.vc = (self.vl + self.vr) / 2
        self.w = 0
        self.a = ACCELERATE

        self.pos = np.array([screen_width // 2, screen_height // 2])
        self.theta = 0
        self.image_theta = 0

    def update_velocity(self, key):
        # 更新机器人速度
        # note 人类控制
        if key == 0:
            self.vl += self.a*DELTA_T
            self.vr += self.a*DELTA_T
        if key == 1:
            self.vl -= self.a*DELTA_T
            self.vr -= self.a*DELTA_T
        if key == 2:
            self.vr += self.a*DELTA_T
        if key == 3:
            self.vl += self.a*DELTA_T
        if key == 4:
            pass
        # note agent控制
        # if key == 0:
        #     self.vl += self.a * DELTA_T
        # if key == 1:
        #     self.vl -= self.a * DELTA_T
        # if key == 2:
        #     self.vr += self.a * DELTA_T
        # if key == 3:
        #     self.vr -= self.a * DELTA_T
        # if key == 4:
        #     pass
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

    def update_position(self):
        vc, theta = self.vc, self.theta
        self.pos[0] += vc * math.cos(theta) * DELTA_T
        # 注意因为坐标系方向改变，变化量的符号要仔细考虑
        self.pos[1] -= vc * math.sin(theta) * DELTA_T
        # 限制新位置在屏幕内
        if self.pos[0] <= 0:
            self.pos[0] = 0
        if self.pos[0] >= self.screen_width:
            self.pos[0] = self.screen_width
        if self.pos[1] <= 0:
            self.pos[1] = 0
        if self.pos[1] >= self.screen_hight:
            self.pos[1] = self.screen_hight
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


info_font = pygame.font.SysFont(None, 36)  # 使用36号字体大小


def normalize_angle(angle):
    # 将角度规范化到 -2π 到 +2π 范围内
    return (angle + 4 * math.pi) % (2 * math.pi)


def show_data(name, pos, digit=3, *content):
    content_str = ""
    for x in content:
        content_str += str(round(x, digit)) + " "
    name = info_font.render(f"{name}: ({content_str})", True, RED)
    screen.blit(name, pos)  # 在机器人坐标下方绘制目标点坐标


class Obstacle:
    def __init__(self, screen) -> None:
        self.screen = screen
        self.screen_width, self.screen_hight = screen.get_size()

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


class Target:
    def __init__(self, screen) -> None:
        self.screen = screen
        self.screen_width, self.screen_hight = screen.get_size()
        self.radius = 20
        self.color = GREEN
        self.pos = np.array([0, 0])

    def show(self):
        pygame.draw.circle(self.screen, self.color, self.pos, self.radius)
