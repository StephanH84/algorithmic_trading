from trading_env import TradingEnv, TradingStream, RunEnv
from dqn_agent import DQN_Agent

def main():
    tradingStream = TradingStream("../../data/S&P_500.csv")
    tradingEnv = TradingEnv(tradingStream, window_size=100)
    alpha = 1e-4
    gamma = 0.5

    agent = DQN_Agent(tradingEnv, alpha, gamma)

    runEnv = RunEnv(tradingEnv, agent)

    runEnv.run()

main()