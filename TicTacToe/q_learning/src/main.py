from game import TicTacToe, RunRL

from TicTacToe.q_learning import Agent


def main():
    game = TicTacToe()
    alpha = 0.0001
    gamma = 0.7
    agentA = Agent(game, alpha, gamma)
    agentB = Agent(game, alpha, gamma)
    runGame = RunRL(game, agentA, agentA)

    runGame.run()


main()