from pygame_env import PygameEnv


class RobotEnv(PygameEnv):
    def __init__(self, screen_size) -> None:
        super().__init__(screen_size)
        self.robot = robot
        self.objects = objects
        self.action_space = action_space
        self.reward_func = reward_func

    def reset(self):
        self.robot.reset()
        for obj in self.objects:
            obj.reset()
        return self.get_state()

    def get_state(self):
        return self.robot.get_state()

    def step(self, action):
        self.action_parser(action)
        for obj in self.objects:
            obj.update_state()
        reward = self.reward_func(self.robot, self.objects)
        done = False
        return self.get_state(), reward, done, None

    def action_parser(self, action):
        # 更新速度
        a = self.obj_dict["robot"].a
        if action == 0:
            self.obj_dict["robot"].vl += a*self.DELTA_T
            self.obj_dict["robot"].vr += a*self.DELTA_T
        if action == 1:
            self.obj_dict["robot"].vl -= a*self.DELTA_T
            self.obj_dict["robot"].vr -= a*self.DELTA_T
        if action == 2:
            self.obj_dict["robot"].vr += a*self.DELTA_T
        if action == 3:
            self.obj_dict["robot"].vl += a*self.DELTA_T
        if action == 4:
            pass
