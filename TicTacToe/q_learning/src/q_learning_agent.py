import json
import random

from common import encode_np_array


class q_function():
    #
    # the outer side of the interface is built on python's list and numpy's arrays
    # inside is based on json-encoded keys and values
    # IMPORTANT: non-set keys are interpreted as having value 0
    # this models a partial function: states x actions -> real values
    #
    STATE_SIZE = 3
    def __init__(self):
        self.q = {}

    def has_state(self, state):
        state_encoded = encode_np_array(state)
        return state_encoded in self.q

    def get_list(self, state):
        state_encoded = encode_np_array(state)
        try:
            action_value_dict = self.q[state_encoded]
        except KeyError:
            return []

        result = []
        for action_encoded, value in action_value_dict.items():
            action = json.loads(action_encoded)
            result.append([action, value])

        return result

    def add_value(self, state, action, incr):
        if incr == 0.0:
            return

        state_encoded = encode_np_array(state)
        action_encoded = json.dumps(action)

        if state_encoded not in self.q:
            self.q[state_encoded] = {}

        if action_encoded not in self.q[state_encoded]:
            self.q[state_encoded][action_encoded] = 0.0

        self.q[state_encoded][action_encoded] += incr


    def get_value(self, state, action):
        state_encoded = encode_np_array(state)
        action_encoded = json.dumps(action)

        if state_encoded not in self.q:
            return 0.0

        if action_encoded not in self.q[state_encoded]:
            return 0.0

        return self.q[state_encoded][action_encoded]


class Agent():
    def __init__(self, game, alpha=0.001, gamma=0.9):
        self.game = game
        self.q = q_function() # Q-function, implicitly no key means value 0
        self.previous_state = None
        self.alpha = alpha
        self.action_space = game.action_space
        self.gamma = gamma

    def turn(self, state):
        self.previous_state = state
        action = self.select_action(state)
        self.action = action

        return action


    def select_action(self, state, eps=0.1):
        rnd = random.uniform(0, 1)

        if rnd < eps or not self.q.has_state(state):
            # explore
            a = self.explore(state)
            return a
        else:
            # exploit

            action_value_list = self.q.get_list(state)
            action_value_list = sorted(action_value_list, key=lambda item: -item[1])

            for action, _ in action_value_list:
                if self.game.__class__.is_valid_action(state, action):
                    return action

            a = self.explore(state)
            return a


    def explore(self, state):
        action_space = self.game.get_available_action_space()
        if len(action_space) == 0:
            return None
        pos = random.randint(0, len(action_space) - 1)
        a = action_space[pos]
        return a

    def update(self, reward, new_state):
        self.apply_q_rule(self.previous_state, self.action, reward, new_state)

    def apply_q_rule(self, s, a, r, s_):
        # calculate max_{a_} Q(s_, a_)
        if not self.q.has_state(s_):
            maxQ = 0.0
        else:
            maxQ = None
            for action, value in self.q.get_list(s_):
                if maxQ == None:
                    maxQ = value

                if maxQ < value:
                    maxQ = value

            if maxQ == None:
                maxQ = 0.0

        incr = self.alpha * (r + self.gamma * maxQ - self.q.get_value(s, a))
        self.q.add_value(s, a, incr)

    def reset_history(self):
        pass