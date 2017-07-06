# TODO: Implement deep neural network (and resp. interface) for Q-function approximation
import tensorflow as tf
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def actions_to_matrix(action):
    matrix = np.zeros([3, 3])
    matrix[action[0], action[1]] = 1
    return matrix

class Network():
    def __init__(self, state_is_terminal):
        self.state_is_terminal = state_is_terminal
        self.theta = None
        self.theta_ = None
        self.initialize()

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def initialize(self):
        self.round_counter = 0
        self.C = 20

        self.y = tf.placeholder(tf.float32, shape=[None])
        self.actions = tf.placeholder(tf.float32, shape=[None, 3, 3])

        # state_seq
        self.phi = tf.placeholder(tf.float32, shape=[None, 3, 3, 4])
        self.output = self.define_network(self.phi)

        output_evaluated = tf.reduce_sum(self.output * self.actions, axis=[1, 2])
        self.loss = tf.reduce_sum(tf.squared_difference(self.y, output_evaluated))

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.output_action = tf.argmax(tf.reshape(self.output, [-1, 9])[0])
        action_ = tf.argmax(tf.reshape(self.actions, [-1, 9])[0])
        correct_prediction = tf.equal(self.output_action, action_)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # save value of weights.
        self.save_weights()


    def define_network(self, phi):
        self.W_conv1 = weight_variable([3, 3, 4, 16])
        self.b_conv1 = bias_variable([3, 3, 16])
        h_conv1 = tf.nn.relu(conv2d(phi, self.W_conv1) + self.b_conv1)

        self.W_conv2 = weight_variable([3, 3, 16, 32])
        self.b_conv2 = bias_variable([3, 3, 32])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2) + self.b_conv2)

        h_conv2_reshaped = tf.reshape(h_conv2, [-1, 3 * 3 * 32])
        self.W_fcn = tf.Variable(tf.zeros([3 * 3 * 32, 9]))
        self.b = tf.Variable(tf.zeros([9]))

        y = tf.matmul(h_conv2_reshaped, self.W_fcn) + self.b

        output_ = tf.nn.softmax(y)

        output = tf.reshape(output_, [-1, 3, 3])

        return output

    def save_weights(self):
        self.W_conv1_saved = self.sess.run(self.W_conv1)
        self.b_conv1_saved = self.sess.run(self.b_conv1)
        self.W_conv2_saved = self.sess.run(self.W_conv2)
        self.b_conv2_saved = self.sess.run(self.b_conv2)
        self.W_fcn_saved = self.sess.run(self.W_fcn)
        self.b_saved = self.sess.run(self.b)

    def evaluate(self, phi):
        # returns the argmax action for given phi
        phi_ = self.sess.run(tf.transpose([np.asarray(phi).tolist()], [0, 2, 3, 1]))
        action_value = self.sess.run(self.output_action, feed_dict={self.phi: phi_})
        return action_value

    def perform_sgd(self, y_, phi_, actions_):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for episode in range(3):
                batch_dict = {self.y: y_, self.phi: phi_, self.actions: actions_}
                self.train_step.run(feed_dict=batch_dict)
                train_accuracy = self.accuracy.eval(feed_dict=batch_dict)
                print('train accuracy %g' % train_accuracy)

    def learn(self, minibatch):
        self.round_counter += 1

        # calculate output vector for stored weights (see save_weights):
        y = []
        actions = []
        phi = []
        for batch in minibatch:
            if self.state_is_terminal(batch[3]):
                value = batch[2]
            else:

                feed_dict = {self.phi: batch[3],
                             self.W_conv1: self.W_conv1_saved,
                             self.b_conv1: self.b_conv1_saved,
                             self.W_conv2: self.W_conv2_saved,
                             self.b_conv2: self.b_conv2_saved,
                             self.W_fcn: self.W_fcn_saved,
                             self.b: self.b_saved}

                output_value = self.sess.run(self.output, feed_dict=feed_dict)
                max_value = tf.reduce_max(output_value)
                value = batch[2] + self.gamma * max_value
            y.append(value)
            actions.append(actions_to_matrix(batch[1]))
            phi.append(batch[0])

        self.perform_sgd(y, phi, actions)

        if self.round_counter % self.C == 0:
            self.save_weights()