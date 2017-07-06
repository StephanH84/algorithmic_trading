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


class Network():
    def __init__(self, state_is_terminal):
        self.state_is_terminal = state_is_terminal
        self.theta = None
        self.theta_ = None
        self.initialize()

    def initialize(self):
        self.y = tf.placeholder(tf.float32, shape=[None])
        self.actions = tf.placeholder(tf.float32, shape=[None, 3, 3])

        # state_seq
        self.phi = tf.placeholder(tf.float32, shape=[None, 3, 3, 4])
        self.output = self.define_network(self.phi)

        output_evaluated = tf.reduce_sum(self.output * self.actions, axis=[1, 2])
        self.loss = tf.reduce_sum(tf.squared_difference(self.y, output_evaluated))

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.output_action = tf.argmax(tf.reshape(self.output, [-1, 9]))
        action_ = tf.argmax(tf.reshape(self.actions, [-1, 9]))
        correct_prediction = tf.equal(self.output_action, action_)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def define_network(self, phi):
        W_conv1 = weight_variable([3, 3, 4, 16])
        b_conv1 = bias_variable([3, 3, 16])
        h_conv1 = tf.nn.relu(conv2d(phi, W_conv1) + b_conv1)

        W_conv2 = weight_variable([3, 3, 16, 32])
        b_conv2 = bias_variable([3, 3, 32])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        h_conv2_reshaped = tf.reshape(h_conv2, [-1, 3 * 3 * 32])
        W_fcn = tf.Variable(tf.zeros([3 * 3 * 32, 9]))
        b = tf.Variable(tf.zeros([9]))

        y = tf.matmul(h_conv2_reshaped, W_fcn) + b

        output_ = tf.nn.softmax(y)

        output = tf.reshape(output_, [-1, 3, 3])

        return output


    def evaluate(self, phi, theta_mode=0):
        # returns the likelihood for the recommended actions
        result = self.sess.run(self.output, feed_dict={self.phi: phi})

    def learn(self, batch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for episode in range(3):
                batch_dict = {self.y: batch[0], self.phi: batch[1], self.actions: batch[2]}
                self.train_step.run(feed_dict=batch_dict)
                train_accuracy = self.accuracy.eval(feed_dict=batch_dict)
                print('train accuracy %g' % train_accuracy)

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