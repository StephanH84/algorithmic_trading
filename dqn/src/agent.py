# TODO: Implement similar agent as for q_learning
import random
from dqn import DQN

class Agent():
    def __init__(self, game, dqn, alpha=0.001, gamma=0.9):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.dqn = DQN(self.game.state_is_terminal, self.alpha, self.gamma)

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
            action_value, success = self.dqn.get_action(state)

            if not success:
                a = self.explore(state)
            else:
                action_list = []
                for row in range(3):
                    for column in range(3):
                        action_list.append([row, column, action_value[row, column]])

                action_list = sorted(action_list, key=lambda item: -item[2])
                for action in action_list:
                    a = [action[0], action[1]]
                    if self.game.__class__.is_valid_action(state, a):
                        return a

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
        self.dqn.store(self.action, reward, new_state)
        self.dqn.learn()

    def reset_history(self):
        self.dqn.reset_history()
