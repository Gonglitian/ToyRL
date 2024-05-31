class AgentConfig:
    # def __init__(self):
    #     self.lr = 0.0001
    #     self.gamma = 0.99
    #     self.epsilon = 1
    #     self.epsilon_min = 0.01
    #     self.epsilon_decay = 0.995
    #     self.memory_size = 10000
    #     self.batch_size = 20
    #     self.train_start = 1000
    #     self.target_update = 10
    #     self.state_size = 4
    #     self.action_size = 2
    #     self.model = self.build_model()
    #     self.target_model = self.build_model()
    #     self.update_target_model()

    # def build_model(self):
    #     model = Sequential()
    #     model.add(Dense(24, input_dim=self.state_size, activation='relu',
    #                     kernel_initializer='he_uniform'))
    #     model.add(Dense(24, activation='relu',
    #                     kernel_initializer='he_uniform'))
    #     model.add(Dense(self.action_size, activation='linear',
    #                     kernel_initializer='he_uniform'))
    #     model.compile(loss='mse', optimizer=Adam(lr=self.lr))
    #     return model

    # def update_target_model(self):
    #     self.target_model.set_weights(self.model.get_weights())
    ...


class DQNConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        ...
