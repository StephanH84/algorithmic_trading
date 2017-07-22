from AT.q_trading.src.trading_env import TradingEnv, TradingStream, RunEnv
from dqn_agent import DQN_Agent

def main():
    tradingStream = TradingStream("../../../data/S&P_500.csv")
    tradingEnv = TradingEnv(tradingStream, window_size=100)
    alpha = 1e-4
    gamma = 0.85
    theta = 3*1e-4
    C = None
    seq_size = 329 #200
    N = 64
    buffer_size = int(1.6 * N)
    beta = None # needs to anneal or is dysfunctional
    T_max = 1000
    activation = "RELU"

    agent = DQN_Agent(tradingEnv, alpha, gamma, theta, C, seq_size, N, beta, T_max, activation, buffer_size, noDDQN=True)

    runEnv = RunEnv(tradingEnv, agent)

    # officially
    # runEnv.run(episodes=5, testing_phase=2750, training_phase=1564)

    runEnv.run(episodes=1, testing_phase=2750, training_phase=1564)

    # runEnv.run(episodes=1, testing_phase=1500, training_phase=2500)

    # runEnv.run(episodes=1, testing_phase=1500, training_phase=500)

main()
