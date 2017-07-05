# TODO: Implement similar agent as for q_learning
import random

class Agent():
    def __init__(self, game, dqn, alpha=0.001, gamma=0.9):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.dqn = dqn

    def turn(self, state):
        self.previous_state = state
        action = self.select_action(state)
        self.action = action

        return action

    def select_action(self, state, eps=0.1):
        rnd = random.uniform(0, 1)

        if rnd < eps:
            # explore
            a = self.explore(state)
        else:
            # exploit
            max_action, success = self.dqn.get_argmax_action(state)

            if success == None:
                a = self.explore(state)

            a = max_action

        return a

    def explore(self, state):
        action_space = self.game.get_available_action_space()
        if len(action_space) == 0:
            return None
        pos = random.randint(0, len(action_space) - 1)
        a = action_space[pos]
        return a

    def update(self, reward, new_state):
        self.dqn.store(self.previous_state, self.action, reward, new_state)
        self.dqn.learn()

