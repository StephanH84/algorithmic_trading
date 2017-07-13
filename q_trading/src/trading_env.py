# Trading environment
import numpy as np

class TradingStream():
    def __init__(self, filename):
        self.filename = filename

    def get_next(self): # generator
        file = open(self.filename, "r")

        def next_gen():
            first = True
            for line in file:
                if first:
                    first = False
                    continue

                splitt = line.rstrip("\n").split(";")

                date = splitt[0]
                value = float(splitt[-1].replace(",", "."))

                yield [date, value]

            file.close()

        return next_gen()

class TradingEnv():
    class StateEnv():
        END_OF_DATA = 1
        NEUTRAL = 0

    def __init__(self, trading_stream, window_size):
        self.action_space = [-1, 0, 1] # long, neutral, short
        self.trading_stream = trading_stream
        self.trading_stream_gen = self.trading_stream.get_next()

        self.trading_history = [] # pairs date and value
        self.action_history = []
        self.window_size = window_size
        self.accumulated_reward = []

        self.EOG = False

    def pull_next_state(self):
        next_state = self.trading_stream_gen()
        self.trading_history.append(next_state)

    def analyze(self):
        pass

    def act(self, action):
        self.action_history.append(action)

        EOG = False
        new_state = None
        try:
            new_state = self.pull_next_state()
        except self.__class__.StateEnv.END_OF_DATA:
            EOG = True
            self.EOG = EOG
            return new_state, None, EOG

        reward = self.get_reward(self.trading_history, new_state, action)

        return new_state, reward, EOG

    def get_reward(self, history, new_state, action):
        # Need to assign reward also for variables with undefined values and different time scales
        # Reward is considered multiplicative?
        # For now keep it simple

        if len(history) < self.window_size:
            return 1 # TODO: Clarify if this value is correct

        try:
            long_term_ratio = history[-1][1] / history[0 - self.window_size][1]
            short_term_ratio = (new_state[1] - history[-1][1]) / history[-1][1]
        except:
            return 1 # TODO: Clarify if this value is correct


        reward = (1 + action * short_term_ratio) * long_term_ratio

        return reward

    def state_is_terminal(self):
        return self.EOG



class RunEnv():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self, N=10000):

        self.env.prepare()
        for n in range(N):
            action = self.agent.turn(self.env.trading_history[-1])

            if action != None:
                new_state, reward, EOG = self.env.act(action) # no need to

                if EOG:
                    break

                self.agent.update(reward, new_state)


        # analyze result afterwards
        self.env.analyze()


class IllegalMoveException(Exception):
    pass
