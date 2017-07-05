# TODO: Implement deep neural network (and resp. interface) for Q-function approximation

class Network():
    def __init__(self, state_is_terminal):
        self.state_is_terminal = state_is_terminal
        self.theta = None
        self.theta_ = None

    def evaluate(self, phi, theta_mode=0):
        # returns the likelihood for the recommended actions
        pass

    def perform_sgd(self, minibatch):
        # calculate output vector:
        y = []
        for batch in minibatch:
            if self.state_is_terminal(batch[3]):
                value = batch[2]
            else:
                argmax_q = 0 # TODO
                value = batch[2] + self.gamma * argmax_q
            y.append(value)