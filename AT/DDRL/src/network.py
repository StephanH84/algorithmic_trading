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

        self.deep_0b = tf.placeholder(tf.float32, shape=[50])
        self.deep_0 = tf.placeholder(tf.float32, shape=[150])
        self.deep_1 = tf.placeholder(tf.float32, shape=[128])

        fuzzy_m = bias_variable([50, 3])
        fuzzy_sigma = bias_variable([50, 3])
        deep_W1 = weight_variable([150, 128])
        deep_b1 = bias_variable([128])
        deep_W2 = weight_variable([128, 128])
        deep_b2 = bias_variable([128])
        deep_W3 = weight_variable([128, 128])
        deep_b3 = bias_variable([128])
        deep_W4 = weight_variable([128, 20])
        deep_b4 = bias_variable([20])


        F_weights = [fuzzy_m, fuzzy_sigma, deep_W1, deep_b1, deep_W2, deep_b2, deep_W3, deep_b3, deep_W4, deep_b4]
        self.F_weights = F_weights

        # netowork output and symbolic expressions for input data to pretraining losses
        self.define_network(self.prices, F_weights)

        _, self.fuzzy_featues_, self.hidden1, self.hidden2, self.hidden3 = self.get_F_features(self.deep_0b, F_weights)

        # pretraining parameters for deep learning NN
        loss_ae_1 = self.define_ae_pretraining_network(self.deep_0, deep_W1, deep_b1, tf.nn.relu)
        #when running in feedDict: deep_0 need to be assigned to output of self.fuzzy_featues_

        loss_ae_2 = self.define_ae_pretraining_network(self.deep_1, deep_W2, deep_b2, tf.nn.relu)
        #when running in feedDict: deep_1 need to be assigned to output of self.hidden1

        loss_ae_3 = self.define_ae_pretraining_network(self.deep_1, deep_W3, deep_b3, tf.nn.relu)
        #when running in feedDict: deep_0 need to be assigned to output of self.hidden2

        loss_ae_4 = self.define_ae_pretraining_network(self.deep_1, deep_W4, deep_b4, tf.nn.relu)
        #when running in feedDict: deep_0 need to be assigned to output of self.hidden3


    def define_network(self, input, F_weights):

        delta_w = weight_variable([self.calc_F_size()])
        delta_b = bias_variable([1])
        delta_u = bias_variable([1])

        F = {}
        F_layers = {}
        delta = {}
        delta[-1] = 0
        for i in range(self.T + 1):
            input_ = input[i, i + self.seq_size]
            F[i], F_layers[i] = self.get_F_features(input_, F_weights, i) # i shifted featues, in the paper this would correspond to F_{m+i}
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

    def define_ae_pretraining_network(self, input, W, b, activation, beta=0.01):
        #input is a placeholder

        W_decode = weight_variable([int(W.shape[1]), int(W.shape[0])])
        b_decode = bias_variable(int(W.shape[0]))

        o_h = tf.matmul(input, W) + b

        hidden = activation(o_h)

        o_id = tf.matmul(hidden, W_decode) + b_decode

        hidden_id = activation(o_id)

        norm = tf.norm(self.W, 2) + tf.norm(self.b, 2) + tf.norm(self.W_decode, 2) + tf.norm(self.b_decode, 2)
        loss = tf.squared_difference(input - hidden_id) + beta * norm

        return loss


    def calc_F_size(self):
        pass

    def get_F_features(self, input, F_weights):
        fuzzy_m, fuzzy_sigma, deep_W1, deep_b1, deep_W2, deep_b2, deep_W3, deep_b3, deep_W4, deep_b4 = F_weights
        # these are pretrained weights

        # fuzzy features
        # fuzzy_m, fuzzy_sigma have dimensions [input_dim=self.seq_size, action_dim=3],
        # and come from pretraining
        fuzzy_featues_ = tf.exp(-(input - fuzzy_m)**2 / fuzzy_sigma)
        fuzzy_featues = tf.reshape(fuzzy_featues_, shape=[-1])
        # fuzzy_featues has dimensions [input_dim, action_dim]

        # plugged into FCN network
        hidden1 = self.make_layer_simple(fuzzy_featues, 128, deep_W1, deep_b1, tf.nn.relu)

        hidden2 = self.make_layer_simple(hidden1, 128, deep_W2, deep_b2, tf.nn.relu)

        hidden3 = self.make_layer_simple(hidden2, 128, deep_W3, deep_b3, tf.nn.relu)

        F = self.make_layer_simple(hidden3, 20, deep_W4, deep_b4, tf.nn.relu)

        return F, [fuzzy_featues_, hidden1, hidden2, hidden3]

    def make_layer_simple(self, input, size, W, b, activation):
        assert int(W.shape[0]) == int(input.shape[1])
        assert int(W.shape[1]) == int(b.shape[0])

        o = tf.matmul(input, W) + b

        output = activation(o)

        return output