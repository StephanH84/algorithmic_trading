# TODO: Implement part of the DQN-algorithm using the network interface (the other up-to-now in network.py)
from network import Network
import random


class DQN():
    def __init__(self, state_is_terminal):
        self.network = Network(state_is_terminal)
        self.replay_memory = []
        self.history = []
        self.step_size = 4
        self.N = 10

    def get_action(self, state):
        success = True

        new_phi = self.history[-self.step_size:]
        if len(new_phi) == self.step_size:
            # Query Q-network
            action = self.network.evaluate(new_phi)
        else:
            success = False
            action = None

        return action, success

    def store(self, action, reward, new_state):
        phi = self.history[-self.step_size:]
        new_phi = (self.history + [new_state])[-self.step_size:]

        if len(phi) == self.step_size and len(new_phi) == self.step_size:
            self.replay_memory.append([phi, action, reward, new_phi])
        self.history.append(new_state.copy())

    def learn(self):
        # provide minibatch
        replay_size = len(self.replay_memory)
        if replay_size >= self.step_size:
            index_list = []
            for n in range(self.N):
                index_list.append(random.randint(0, replay_size-1))
            index_list = list(set(index_list))

            minibatch = [self.replay_memory[p] for p in index_list]

            self.network.learn(minibatch)

