# Trading environment
from q_trading.src.common import plot_data

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
        self.action_space = [-1, 0, 1] # short, neutral, long
        self.trading_stream = trading_stream
        self.trading_stream_gen = self.trading_stream.get_next()

        self.trading_history = [] # pairs date and value
        self.action_history = []
        self.window_size = window_size
        self.profits = []

        self.test_phase = False

        self.EOG = False

    def prepare(self):
        self.pull_next_state()

    def pull_next_state(self):
        next_state = self.trading_stream_gen()
        self.trading_history.append(next_state)

    def plot_wealth(self):
        plot_data(self.profits)

    def decide(self, action):
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

        self.process_profit()

        return new_state, reward, EOG

    def process_profit(self, transaction_cost=0.0):
        previous_action = self.action_history[-2]
        current_action = self.action_history[-1]

        previous_state = self.action_history[-2]
        current_state = self.action_history[-1]

        z = current_state[1] - previous_state[1]

        profit = previous_action * z - transaction_cost * abs(current_action - previous_action)

        date = current_state[0]

        self.profits.append([date, profit])

    def get_reward(self, history, new_state, action):
        # Need to assign reward also for variables with undefined values and different time scales
        # Reward is considered multiplicative?
        # For now keep it simple

        if len(history) < self.window_size:
            print("Problem!")
            return 1 # TODO: Clarify if this value is correct

        try:
            long_term_ratio = history[-1][1] / history[0 - self.window_size][1]
            short_term_ratio = (new_state[1] - history[-1][1]) / history[-1][1]
        except:
            print("Problem!")
            return 1 # TODO: Clarify if this value is correct


        reward = (1 + action * short_term_ratio) * long_term_ratio

        return reward

    def enable_test_phase(self):
        self.test_phase = True


    def state_is_terminal(self):
        return self.EOG



class RunEnv():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self, testing_phase, training_phase=1460):

        self.env.prepare()

        # training phase
        for n in range(training_phase):
            action = self.agent.turn(self.env.trading_history[-1])

            if action != None:
                new_state, reward, EOG = self.env.decide(action)

                if EOG:
                    break

                self.agent.update(reward, new_state)

        # testing phase, i.e. no updates
        self.env.enable_test_phase()

        testing_phase = 2750
        for n in range(testing_phase):
            action = self.agent.turn(self.env.trading_history[-1])

            if action != None:
                new_state, reward, EOG = self.env.decide(action)

                if EOG:
                    break

        self.env.plot_wealth()


class IllegalMoveException(Exception):
    pass
