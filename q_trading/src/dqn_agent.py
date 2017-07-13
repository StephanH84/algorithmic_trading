from network import Network
import random
import time


class DQN_Agent():
    def __init__(self, env, alpha, gamma):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma

        self.replay_memory = []
        self.history = []
        self.step_size = 200
        self.network = Network(self.env.state_is_terminal, self.step_size, alpha, gamma)
        self.N = 64 # mini-batch size before reduction
        self.learn_time_random = []


    def turn(self, state, eps=0.1):
        rnd = random.uniform(0, 1)

        def explore(self):
            return random.randint(-1, 1)

        if rnd < eps:
            # explore
            a = self.explore(state)
        else:
            # exploit
            action_value, success = self.get_action(state)

            if not success:
                a = self.explore(state)
            else:
                a = action_value

        self.action = a
        return a


    def update(self, reward, new_state):
        self.store(self.action, reward, new_state)
        self.learn()

    # Now the technical part interfacing to the learning part (network)

    def get_action(self, state):
        success = True

        new_phi = self.history[-self.step_size:]
        if len(new_phi) == self.step_size:
            # Query Q-network
            output_value = self.network.evaluate(new_phi)
        else:
            success = False
            output_value = None

        return output_value, success

    def store(self, action, reward, new_state):
        phi = self.history[-self.step_size:]
        new_phi = (self.history + [new_state])[-self.step_size:]

        if len(phi) == self.step_size and len(new_phi) == self.step_size:
            self.replay_memory.append([phi, action, reward, new_phi])

        new_state_ = new_state.copy()
        previous_state = self.history[-1]
        value = new_state_[1] - previous_state[1] # Take delta price
        date = new_state_[0]
        self.history.append([date, value])

    def learn(self):
        # provide minibatch
        replay_size = len(self.replay_memory)
        if replay_size >= self.step_size:
            index_list = []
            t0 = time.time()
            for n in range(self.N):
                index_list.append(random.randint(0, replay_size-1))
            t1 = time.time()
            self.learn_time_random.append(t1 - t0)
            index_list = list(set(index_list))

            minibatch = [self.replay_memory[p] for p in index_list]

            self.network.learn(minibatch)

    def reset_history(self):
        self.history = []