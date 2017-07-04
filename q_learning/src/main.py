from game import TicTacToe, Player
from q_learning import Agent


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