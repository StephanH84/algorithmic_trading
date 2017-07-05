# TODO: Implement DQN-algorithm using the network interface
import numpy as np


class DQN():
    def __init__(self, network):
        self.network = network
        self.replay_memory = []
        self.step_count = 0
        self.history = []
        self.step_size = 4

    def get_argmax_action(self, state):
        action = None
        success = True
        return action, success

    def store(self, action, reward, new_state):
        phi = self.history[-self.step_size:]
        new_phi = (self.history + [new_state])[-self.step_size:]

        if len(phi) == self.step_size and len(new_phi) == self.step_size:
            self.replay_memory.append([phi, action, reward, new_phi])
        self.history.append(new_state)

    def learn(self, C=10):

        if self.count % C == 0:
            self.reset()
        self.count += 1

    def reset(self):
        pass