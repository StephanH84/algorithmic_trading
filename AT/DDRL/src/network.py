import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class Network():
    def __init__(self, seq_size=128, T=4):
        self.seq_size = seq_size
        self.T = T

        self.initialize()

    def initialize(self):
        self.prices = tf.placeholder(tf.float32, shape=[self.T + self.seq_size])

        self.output = self.define_network(self.prices)

    def define_network(self, input):

        F_weights = None

        delta_w = weight_variable([self.calc_F_size()])
        delta_b = bias_variable([1])
        delta_u = bias_variable([1])

        F = {}
        F_layers = {}
        delta = {}
        delta[-1] = 0
        for i in range(self.T + 1):
            F[i], F_layers[i] = self.get_F_features(input, F_weights, i) # i shifted featues, in the paper this would correspond to F_{m+i}
            # layers are copies of the symbolic variables for the NN for pattern recognition

            delta[i] = tf.tanh(tf.reduce_sum(delta_w * F[i]) + delta_b + delta_u * delta[i - 1])

        self.F, self.F_layers, self.delta = F, F_layers, delta

        R = {}
        U = {}
        U[-1] = 0
        for t in range(1, self.T + 1):
            R[t] = delta[t-1] * input[t + self.seq_size] - self.c * tf.abs(delta[t] - delta[t - 1])
            U[t] = U[t-1] + R[t]

        self.R = R
        self.U = U


    def calc_F_size(self):
        pass

    def get_F_features(self, input, F_weights, i):
        fuzzy_m, fuzzy_sigma, deep_W1, deep_b1, deep_W2, deep_b2, deep_W3, deep_b3, deep_W4, deep_b4 = F_weights

        input_ = input[i, i+self.seq_size]

        # fuzzy features
        # fuzzy_m, fuzzy_sigma have dimensions [input_dim=self.seq_size, action_dim=3],
        # and come from pretraining
        fuzzy_featues = tf.exp(-(input_ - fuzzy_m)**2 / fuzzy_sigma)
        # fuzzy_featues has dimensions [input_dim, action_dim]

        # plugged into FCN network
        hidden1 = self.make_layer_simple(fuzzy_featues, 128, deep_W1, deep_b1, tf.nn.relu)

        hidden2 = self.make_layer_simple(hidden1, 128, deep_W2, deep_b2, tf.nn.relu)

        hidden3 = self.make_layer_simple(hidden2, 128, deep_W3, deep_b3, tf.nn.relu)

        F = self.make_layer_simple(hidden3, 20, deep_W4, deep_b4, tf.nn.relu)

        return F, [hidden1, hidden2, hidden3]

    def make_layer_simple(self, input, size, W, b, activation):
        assert int(W.shape[0]) == int(input.shape[1])
        assert int(W.shape[1]) == int(b.shape[0])

        o = tf.matmul(input, W) + b

        output = activation(o)

        return output