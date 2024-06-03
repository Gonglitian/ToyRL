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

        # RL相关变量
        self.state = ...
        self.reward_sum = 0

    def add_objects(self, *objects: Object):
        for obj in objects:
            self.obj_dict[obj.name] = obj
            self.obj_dict[obj.name].env = self
            self.obj_dict[obj.name].screen = self.screen

    def delete_object(self, obj: Object):
        del self.obj_dict[obj.name]

    def reset(self):
        return self.state

    def step(self, action):
        self.action_parser(action)
        for obj in self.obj_dict.values():
            obj.state_update()
        info = {}
        done = self._get_done()
        return self.state, self.reward_sum, done, info

    def _get_done(self):
        pass

    def _get_reward(self):
        pass

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

    def action_parser(self, action):
        pass


class RobotEnv(gymnasium.Env):
    def __init__(self, screen_width, screen_height):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Robot Reinforcement Learning")
        self.robot = Robot(self.screen)
        self.score = 0
        self.info_font = pygame.font.SysFont(None, 36)
        # note 定义动作空间 0:vl+ 1:vl- 2:vr+ 3:vr- 4:do nothing
        self.action_space = spaces.Discrete(5)
        # note 定义状态空间 [robot_pos[0],robot_pos[1],target_pos[0],target_pos[1],vl,vr,theta,distance]
        self.observation_space = spaces.Box(
            np.array(
                [
                    -screen_width,
                    -screen_height,
                    -screen_width,
                    -screen_height,
                    -self.robot.vm,
                    -self.robot.vm,
                    0,
                    0,
                ]
            ),
            np.array(
                [
                    screen_width,
                    screen_height,
                    screen_width,
                    screen_height,
                    self.robot.vm,
                    self.robot.vm,
                    2 * math.pi,
                    (2**0.5) * self.screen_width,
                ]
            ),
        )  # 定义观察空间
        self.target_pos = np.array([screen_width // 3, screen_height // 3])
        self.target_radius = 10

        self.distance = 0
        self.steps_count = 0
        self.left_wheel_speeds = []

        self.right_wheel_speeds = []
        # 储存轨迹
        self.robot_trajectory = []
        self.clock = pygame.time.Clock()  # 创建Clock对象

        self.target_list = []
        self.target_count = 0
        self.frames = []

    def reset(self, seed=None):
        self.robot.pos = np.array(
            [
                random.randint(0, self.screen_width),
                random.randint(0, self.screen_height),
            ]
        )
        # self.target_pos = np.array(
        #     [
        #         random.randint(0, self.screen_width),
        #         random.randint(0, self.screen_height),
        #     ]
        # )
        for _ in range(5):
            self.target_list.append(np.array(
                [
                    random.randint(0, self.screen_width),
                    random.randint(0, self.screen_height),
                ]
            ))
        self.current_target = self.target_list[self.target_count]
        self.info = {}
        self.state = self._get_observation()
        self.robot_trajectory = []
        return (self.state, self.info)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        self.steps_count += 1
        self.robot.update_velocity(action)
        self.robot.update_position()

        _, collision = self.check_circle_to_circle_collision()

        self.state = self._get_observation()
        reward = self._get_reward()

        # 判断是否游戏结束（例如，机器人移出屏幕）
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        self.info = {}
        self.left_wheel_speeds.append(self.robot.vl)
        self.right_wheel_speeds.append(self.robot.vr)
        # 更新轨迹
        self.robot_trajectory.append(np.copy(self.robot.pos))

        return self.state, reward, terminated, truncated, self.info
        # return self._get_observation(), reward, done, info

    def _get_observation(self):
        # 根据需要返回环境的观察值
        return np.array(
            [
                self.robot.pos[0],
                self.robot.pos[1],
                self.current_target[0],
                self.current_target[1],
                self.robot.vl,
                self.robot.vr,
                self.robot.theta,
                self.distance,
            ]
        ).astype(np.float32)

        # def _generate_target_pos(self):
        随机生成目标点的位置
        # return np.array([random.randint(0, self.screen_width), random.randint(0, self.screen_height)])

    def render(self, mode="human"):
        # print(len(self.robot_trajectory))
        # 清除屏幕
        self.screen.fill((255, 255, 255))
        # 绘制机器人
        self.robot.show()
        # 绘制目标点
        pygame.draw.circle(
            self.screen, GREEN, self.current_target, self.target_radius)
        # 绘制轨迹
        # print(self.robot_trajectory[0], len(self.robot_trajectory))
        if len(self.robot_trajectory) > 1:
            pygame.draw.lines(self.screen, RED, False,
                              self.robot_trajectory, 2)
        # 更新屏幕显示
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        self.clock.tick(30)
        pygame.display.flip()
        self.frames.append(pygame.surfarray.array3d(
            pygame.display.get_surface()))

    def _get_terminated(self):
        if self.target_count == len(self.target_list):
            self.target_count = 0
            pygame.quit()
            return True
        return False

    def touch_wall(self):
        return (
            self.robot.pos[0] <= 0
            or self.robot.pos[0] >= self.screen_width
            or self.robot.pos[1] <= 0
            or self.robot.pos[1] >= self.screen_height
        )

    def _get_truncated(self):
        # 限制新位置在屏幕内
        # if self.touch_wall():
        #     # print("touch wall", self.steps_count)
        #     return True
        # if self.steps_count >= STEPS_LIMIT:
        #     self.steps_count = 0
        #     return True
        return False

    def _get_reward(self):
        self.distance, collision = self.check_circle_to_circle_collision()
        distance_percentage = self.distance / self.screen_width * 100

        if collision:
            self.target_count += 1
            if self.target_count != len(self.target_list):
                self.current_target = self.target_list[self.target_count]
            return 100
        elif distance_percentage <= 25:
            return -0.1
        elif distance_percentage <= 50:
            return -0.25
        elif distance_percentage <= 75:
            return -0.5
        else:
            return -1

    def check_circle_to_circle_collision(self):
        a = (self.robot.pos - self.current_target) ** 2
        distance = a.sum() ** 0.5
        return distance, distance <= (self.robot.radius + self.target_radius)

    def close(self):
        # 关闭环境
        pygame.quit()


class RobotDynamicEnv(gymnasium.Env):
    def __init__(self, screen_width, screen_height):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))

        pygame.display.set_caption("Robot Reinforcement Learning")
        self.clock = pygame.time.Clock()  # 创建Clock对象
        self.info_font = pygame.font.SysFont(None, 36)

        self.robot = Robot(self.screen)
        self.obstacle = Obstacle(self.screen)
        self.target = Target(self.screen)

        # note 定义动作空间 0:vl+ 1:vl- 2:vr+ 3:vr- 4:do nothing
        self.action_space = spaces.Discrete(5)
        # note 定义状态空间 [robot.pos[0],robot.pos[1],target.pos[0],target.pos[1],obstacle.pos[0],obstacle.pos[1],vl,vr,theta,distance,dis_obstacle]
        self.observation_space = spaces.Box(
            np.array(
                [
                    -screen_width,
                    -screen_height,
                    -screen_width,
                    -screen_height,
                    -screen_width,
                    -screen_height,
                    -self.robot.vm,
                    -self.robot.vm,
                    0,
                    0,
                    0,
                ]
            ),
            np.array(
                [
                    screen_width,
                    screen_height,
                    screen_width,
                    screen_height,
                    screen_width,
                    screen_height,
                    self.robot.vm,
                    self.robot.vm,
                    2 * math.pi,
                    (2**0.5) * self.screen_width,
                    (2**0.5) * self.screen_width,
                ]
            ),
        )  # 定义观察空间
        self.distance = 0
        self.obstacle_distance = 0

        # 储存robot轨迹坐标，以便画出轨迹
        self.robot_trajectory = []
        self.steps_count = 0
        self.frames = []

    def reset(self, seed=None):
        self.robot.pos = np.array([50, 350])
        self.target.pos = np.array([350, 50])
        self.obstacle.pos = np.array([200, 200])
        self.info = {}
        self.state = self._get_observation()
        return (self.state, self.info)

    def step(self, action):
        self.steps_count += 1
        self.robot.update_velocity(action)

        self.robot.update_position()
        self.obstacle.update_position()

        reward = self._get_reward()
        self.state = self._get_observation()
        # 判断是否游戏结束（例如，机器人移出屏幕）
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = {}
        # 更新机器人坐标轨迹
        # print(len(self.robot_trajectory))
        self.robot_trajectory.append(np.copy(self.robot.pos))
        return self.state, reward, terminated, truncated, info
        # return self._get_observation(), reward, done, info

    def _get_observation(self):
        # 根据需要返回环境的观察值
        return np.array(
            [
                self.robot.pos[0],
                self.robot.pos[1],
                self.target.pos[0],
                self.target.pos[1],
                self.obstacle.pos[0],
                self.obstacle.pos[1],
                self.robot.vl,
                self.robot.vr,
                self.robot.theta,
                self.distance,
                self.obstacle_distance,
            ]
        ).astype(np.float32)

        # def _generate_target_pos(self):
        随机生成目标点的位置
        # return np.array([random.randint(0, self.screen_width), random.randint(0, self.screen_height)])

    def render(self, mode="human"):
        # 检测退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        # 清除屏幕
        self.screen.fill((255, 255, 255))
        # 绘制物体
        self.robot.show()
        self.target.show()
        self.obstacle.show()
        # 绘制轨迹
        if len(self.robot_trajectory) > 1:
            # print("draw lines")
            pygame.draw.lines(self.screen, RED, False,
                              self.robot_trajectory, 2)
        pygame.display.flip()  # 更新屏幕显示
        self.clock.tick(30)
        self.frames.append(pygame.surfarray.array3d(
            pygame.display.get_surface()))

    def _get_terminated(self):
        if self.distance <= (
            self.robot.radius + self.target.radius
        ) or self.obstacle_distance <= (self.obstacle.radius + self.robot.radius):
            pygame.quit()
            return True
        return False

    def touch_wall(self):
        return (
            self.robot.pos[0] <= 0
            or self.robot.pos[0] >= self.screen_width
            or self.robot.pos[1] <= 0
            or self.robot.pos[1] >= self.screen_height
        )

    def _get_truncated(self):
        # # 限制新位置在屏幕内
        # if self.touch_wall():
        #     # print("touch wall", self.steps_count)
        #     return True
        if self.steps_count >= STEPS_LIMIT:
            self.steps_count = 0
            return True
        return False

    def _get_reward(self):
        r = 0
        # 如果机器人与目标点碰撞，返回奖励100
        self.distance, collision_with_target = self.check_circle_to_circle_collision(
            self.robot, self.target
        )
        self.obstacle_distance, collision_with_obsacle = (
            self.check_circle_to_circle_collision(self.robot, self.obstacle)
        )
        dp = self.distance / self.screen_width * 100
        dpo = self.obstacle_distance / self.screen_width * 100
        r += self.get_dp_reward(dp)
        r += self.get_dpo_reward(dpo)

        if collision_with_target:
            r += 100
        if collision_with_obsacle:
            r += -100
        return r

    def get_dp_reward(self, dp):
        if dp <= 25:
            return -0.1
        elif dp <= 50:
            return -0.25
        elif dp <= 75:
            return -0.5
        else:
            return -1

    def get_dpo_reward(self, dpo):
        if dpo <= 25:
            return -1
        elif dpo <= 50:
            return -0.5
        elif dpo <= 75:
            return -0.25
        else:
            return -0.1

    def check_circle_to_circle_collision(self, circle1, circle2):
        distance = np.linalg.norm(circle1.pos - circle2.pos)
        return distance, distance <= (circle1.radius + circle2.radius)

    def close(self):
        # 关闭环境
        pygame.quit()


def circle_to_circle_collision(circle1, circle2) -> tuple[float, bool]:
    distance = np.linalg.norm(circle1.pos - circle2.pos)
    return distance, distance <= (circle1.radius + circle2.radius)
# check
# env = RobotEnv(screen_width=400, screen_height=400)
# check_env(env, warn=True)
