import numpy as np

from common import ProductSet


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
            return -3 # large negative reward to make to make it dislike losing
        elif status == self.StateResult.UNDEFINED:
            return 0
        elif status == self.StateResult.DRAW:
            return 0

    def check_state(self):
        return self.check_state_g(self.state)

    def check_state_g(self, state):
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

    def state_is_terminal(self, state):
        return self.check_state_g(state) in [self.StateResult.WIN, self.StateResult.LOSS, self.StateResult.DRAW]


class IllegalMoveException(Exception):
    pass


class Player():
    A = 1
    B = -1
    EMPTY = 0


class RunRL():
    def __init__(self, game, agentA, agentB):
        self.game = game
        self.agentA = agentA
        self.agentB = agentB

    def run(self, N=10000000, episodes=30000):

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

                if game_counter > episodes:
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
                print("game_counter: %s" % game_counter)
                self.game.reset()
                self.agentA.reset_history()
                self.agentB.reset_history()