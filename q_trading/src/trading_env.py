# Trading environment
from q_trading.src.common import plot_data

class TradingStream():
    def __init__(self, filename):
        self.filename = filename
        self.file_handles = []

    def __del__(self):
        for handle in self.file_handles:
            try:
                handle.close()
            except:
                pass

    def get_next(self): # generator
        file = open(self.filename, "r")
        self.file_handles.append(file)

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

        self.trading_history = [] # pairs date and value
        self.action_history = []
        self.window_size = window_size
        self.profits = []

        self.test_phase = False

        self.EOG = False

    def intialize(self):
        self.trading_stream_gen = self.trading_stream.get_next()
        while len(self.trading_history) < self.window_size:
            self.pull_next_state()

    def pull_next_state(self):
        next_state = next(self.trading_stream_gen)
        self.trading_history.append(next_state)
        return next_state

    def plot_wealth(self):
        total_profit = 0.0
        accumulated_profit = []

        for date, profit in self.profits:
            total_profit += profit
            accumulated_profit.append([date, total_profit])

        print(total_profit)

        plot_data(self.profits, name="profits")

        plot_data(accumulated_profit, name="acc_profits")

        plot_data(self.action_history, name="actions")

    def act(self, action):

        EOG = False
        new_state = None
        try:
            new_state = self.pull_next_state()
            self.action_history.append([new_state[0], action])
        except (self.__class__.StateEnv.END_OF_DATA, StopIteration):
            EOG = True
            self.EOG = EOG
            self.action_history.append([None, action])
            return new_state, None, EOG

        reward = self.get_reward(self.trading_history, new_state, action)

        self.process_profit()

        return new_state, reward, EOG

    def process_profit(self, transaction_cost=0.0):
        if self.test_phase and len(self.action_history) > 1 and len(self.trading_history) > 2:
            previous_action = self.action_history[-2][1]
            current_action = self.action_history[-1][1]

            previous_state = self.trading_history[-2]
            current_state = self.trading_history[-1]

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

    def get_rewards(self):
        actions = [-1, 0, 1]

        # pull new state
        EOG = False
        new_state = None
        try:
            new_state = self.pull_next_state()
        except (self.__class__.StateEnv.END_OF_DATA, StopIteration):
            EOG = True
            self.EOG = EOG
            return actions, None, new_state, EOG

        rewards = [self.get_reward(self.trading_history, new_state, action) for action in actions]

        return actions, rewards, new_state, EOG

    def enable_test_phase(self):
        self.test_phase = True


    def state_is_terminal(self, state):
        return self.EOG



class RunEnv():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self, episodes, testing_phase, training_phase):


        for e in range(episodes):
            self.env.intialize()
            if e > 0:
                pass # TODO something like self.env.reset_history()

            # training phase
            for n in range(training_phase):

                if n % 2 == 0:
                    print("Epsiode: %s, Day: %s" % (e, n))

                actions, rewards, new_state, EOG = self.env.get_rewards()

                if EOG:
                    break

                self.agent.update_special(actions, rewards, new_state)

        # testing phase, i.e. no updates
        self.env.enable_test_phase()

        for n in range(testing_phase):
            action = self.agent.turn(self.env.trading_history[-1], dontExplore=True)

            new_state, reward, EOG = self.env.act(action)

            if EOG:
                break

            self.agent.store(action, reward, new_state)

        self.env.plot_wealth()
