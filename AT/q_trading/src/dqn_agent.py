import random
import time

import numpy as np

from AT.q_trading.src.network import Network


class DQN_Agent():
    def __init__(self, env, alpha, gamma, theta, C, seq_size, N, beta, T_max, activation, buffer_size, noDDQN):
        self.replay_memory = []
        self.history = []
        self.seq_size = seq_size
        self.network = Network(env.state_is_terminal, self.seq_size, alpha, gamma, theta, C, beta, activation, noDDQN)
        self.N = N # mini-batch size
        self.learn_time_random = []
        self.buffer_size = buffer_size

        self.T_max = T_max
        self.frame_counter = 0

    def get_annealling_eps(self):
        u = random.uniform(0, 1)

        eps = [0.1, 0.01, 0.4]

        if 0 < u < 0.4:
            i = 0
        elif 0.4 < u < 0.75:
            i = 1
        else:
            i = 2

        k = - np.log(eps[i]) / self.T_max
        T = self.frame_counter
        eps = np.exp(-k*T)

        self.frame_counter += 1

        if self.frame_counter % 3 == 0:
            print(eps)
        return eps

    def turn(self, state, eps=0.05, dontExplore=False):
        def explore(self):
            return random.randint(-1, 1)

        if dontExplore:
            a = self.exploit(explore, state)
        else:
            rnd = random.uniform(0, 1)
            eps = self.get_annealling_eps()
            if rnd < eps:
                # explore
                a = explore(state)
            else:
                a = self.exploit(explore, state)

        self.action = a
        return a

    def exploit(self, explore, state):
        # exploit
        action_value, success = self.get_action(state)
        if not success:
            a = explore(state)
        else:
            a = np.argmax(np.asarray(action_value)) - 1
        return a

    def update_special(self, actions, rewards, new_state):
        self.store_special(actions, rewards, new_state)
        self.learn_special()


    def update(self, reward, new_state, no_subsample):
        self.store(self.action, reward, new_state)
        self.learn(no_subsample)

    # Now the technical part interfacing to the learning part (network)

    def get_action(self, state):
        success = True

        new_phi = self.history[-self.seq_size:]
        if len(new_phi) == self.seq_size:
            # Query Q-network
            output_value = self.network.evaluate(new_phi)
        else:
            success = False
            output_value = None

        return output_value, success

    def store(self, action, reward, new_state):
        phi = self.history[-self.seq_size:]
        new_phi = (self.history + [new_state])[-self.seq_size:]

        if len(phi) == self.seq_size and len(new_phi) == self.seq_size:
            self.replay_memory.append([phi, action, reward, new_phi])
            self.replay_memory = self.replay_memory[-self.buffer_size:]

        new_state_ = new_state.copy()
        previous_state = self.history[-1] if len(self.history) > 0 else [None, new_state_[1]]
        value = new_state_[1] - previous_state[1] # Take delta price
        date = new_state_[0]
        self.history.append([date, value])


    def store_special(self, actions, rewards, new_state):
        phi = self.history[-self.seq_size:]
        new_phi = (self.history + [new_state])[-self.seq_size:]

        if len(phi) == self.seq_size and len(new_phi) == self.seq_size:
            self.replay_memory.append([phi, actions, rewards, new_phi])
            self.replay_memory = self.replay_memory[-self.buffer_size:]

        new_state_ = new_state.copy()
        previous_state = self.history[-1] if len(self.history) > 0 else [None, new_state_[1]]
        value = new_state_[1] - previous_state[1] # Take delta price
        date = new_state_[0]
        self.history.append([date, value])

    def learn(self, no_subsample):
        # provide minibatch
        replay_size = len(self.replay_memory)
        if replay_size >= self.N * 1.5:
            index_list = []
            t0 = time.time()
            while len(index_list) < self.N: # the loop terminates because of the above condition, also in a sufficient small time due to the factor 2
                index = random.randint(0, replay_size - 1)
                if index not in index_list:
                    index_list.append(index)
            t1 = time.time()
            self.learn_time_random.append(t1 - t0)
            index_list = list(set(index_list))

            minibatch = [self.replay_memory[p] for p in index_list]

            self.network.learn(minibatch, no_subsample)


    def learn_special(self):
        # provide minibatch
        replay_size = len(self.replay_memory)
        if replay_size >= self.N * 1.5:
            index_list = []
            t0 = time.time()

            while len(index_list) < self.N: # the loop terminates because of the above condition, also in a sufficient small time due to the factor 2
                index = random.randint(0, replay_size - 1)
                if index not in index_list:
                    index_list.append(index)

            t1 = time.time()

            self.learn_time_random.append(t1 - t0)
            # index_list = list(set(index_list))

            minibatch = [self.replay_memory[p] for p in index_list]

            self.network.learn(minibatch)

    def reset_history(self):
        self.history = []
        self.replay_memory = []