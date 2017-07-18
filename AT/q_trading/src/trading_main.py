from trading_env import TradingEnv, TradingStream, RunEnv
from dqn_agent import DQN_Agent

def main():
    tradingStream = TradingStream("../../../data/S&P_500.csv")
    tradingEnv = TradingEnv(tradingStream, window_size=100)
    alpha = 1e-4
    gamma = 0.85
    theta = 3*1e-4
    C = None
    seq_size = 100
    N = 55
    beta = 80

    agent = DQN_Agent(tradingEnv, alpha, gamma, theta, C, seq_size, N, beta)

    runEnv = RunEnv(tradingEnv, agent)

    # officially
    # runEnv.run(episodes=4, testing_phase=2750, training_phase=1460)

    # runEnv.run(episodes=1, testing_phase=1500, training_phase=2500)

    runEnv.run(episodes=1, testing_phase=1500, training_phase=1500)

main()