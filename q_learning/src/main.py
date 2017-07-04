import numpy as np
import json
import random

class IllegalMoveException(Exception):
    pass

class Player():
    A = 1
    B = -1
    EMPTY = 0

class ProductSet():
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def produce(self):
        s = []

        for a in self.A:
            for b in self.B:
                s.append([a, b])

        return s

class Map():
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination


class PartialMap(dict):
    pass


def encode_np_array(array):
    try:
        result = json.dumps([i for i in array.flatten()])
    except:
        assert True
        result = None
    return result


def decode_np_array(str_, size=3):
    try:
        result = np.asarray(json.loads(str_)).reshape([size, size])
    except:
        assert True
        result = None
    return result


class TicTacToe():
    SIZE = 3
    class StateResult():
        UNDEFINED = 0
        DRAW = 1 # FULL
        WIN = 2
        LOSS = 3

    def __init__(self):
        self.state = self.initial_state()
        self.action_space = self.generate_action_space()

    def get_available_action_space(self):
        # returns part of the action space whose corresponding fields on the board are empty
        actions = []
        for action in self.action_space:
            if self.is_valid_action(self.state, action):
                actions.append(action)
        return actions

    def output(self):
        print(self.state)

    def reset(self):
        self.state = self.initial_state()

    def initial_state(self):
        return np.asarray([[Player.EMPTY] * self.SIZE] * self.SIZE)


    # def generate_state_space(self):
    #     return Map(self.generate_action_space(), [Player.A, Player.B, Player.EMPTY])


    def generate_action_space(self):
        return ProductSet(range(self.SIZE), range(self.SIZE)).produce()


    def move(self, player, action):
        # check if position and player are valid
        if player not in [Player.A, Player.B]:
            raise IllegalMoveException()

        x, y = action

        if x not in range(self.SIZE) or y not in range(self.SIZE):
            raise IllegalMoveException()

        # test if already placed:
        if not self.state[x, y] == Player.EMPTY:
            raise IllegalMoveException()

        self.state[x, y] = player
        state = self.check_state()
        reward = float(self.get_reward(state))

        return self.state, reward, state

    def get_reward(self, status):
        if status == self.StateResult.WIN:
            return 1
        elif status == self.StateResult.LOSS:
            return -20
        elif status == self.StateResult.UNDEFINED:
            return 0
        elif status == self.StateResult.DRAW:
            return 0

    def check_state(self):
        # check if draw (full), win or loss or undefined at all for A

        for value in [Player.A, Player.B]:
            if value == Player.A:
                result = self.StateResult.WIN
            elif value == Player.B:
                result = self.StateResult.LOSS

            # verticals
            for x in range(self.SIZE):
                if self.state[x, 0] == value and self.state[x, 1] == value and self.state[x, 2] == value:
                    return result

            # horizontals
            for y in range(self.SIZE):
                if self.state[0, y] == value and self.state[1, y] == value and self.state[2, y] == value:
                    return result

            # diagonals
            if self.state[0, 0] == value and self.state[1, 1] == value and self.state[2, 2] == value:
                return result

            if self.state[0, 2] == value and self.state[1, 1] == value and self.state[2, 0] == value:
                return result

        # otherwise check if the board is full (draw)
        counter_not_empty = 0
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if self.state[x, y] != Player.EMPTY:
                    counter_not_empty += 1

        if counter_not_empty == self.SIZE**2:
            result = self.StateResult.DRAW
            return result

        result = self.StateResult.UNDEFINED
        return result

    @staticmethod
    def is_valid_action(state, action):
        x, y = action
        return state[x, y] == Player.EMPTY

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
        for action_encoded, value in action_value_dict.iteritems():
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
        else:
            # exploit
            max_action, max_value = None, None
            for action, value in self.q.get_list(state):
                if max_value == None:
                    max_action, max_value = action, value

                if max_value < value:
                    max_action, max_value = action, value

            if max_action == None:
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




class RunRL():
    def __init__(self, game, agentA, agentB):
        self.game = game
        self.agentA = agentA
        self.agentB = agentB

    def run(self, N=10000000):

        game_counter = 0
        for n in range(N):
            if n % 2 == 0:
                player = Player.A
                action = self.agentA.turn(self.game.state)
            else:
                player = Player.B
                action = self.agentB.turn(self.game.state)

            if action != None:
                new_state, reward, state = self.game.move(player, action)

                if game_counter > 30000:
                    self.game.output()
                    print(str(state) + "\n")

                if n % 2 == 0:
                    self.agentA.update(reward, new_state)
                else:
                    self.agentB.update(-reward, new_state)  # factor -1, since reward is seen from A's point of view
            else:
                assert True
                state = self.game.StateResult.DRAW

            if state in [self.game.StateResult.WIN, self.game.StateResult.LOSS, self.game.StateResult.DRAW]:
                game_counter += 1
                self.game.reset()


def main():
    game = TicTacToe()
    alpha = 0.0001
    gamma = 0.7
    agentA = Agent(game, alpha, gamma)
    agentB = Agent(game, alpha, gamma)
    runGame = RunRL(game, agentA, agentA)

    runGame.run()


main()