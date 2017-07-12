# TODO: Implement deep neural network (and resp. interface) for Q-function approximation
# This version is according to the official literature, i.e. Mnih's version (reason: to be prepared for the extensions DDQN)

import tensorflow as tf
import numpy as np
import time

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
    def __init__(self, state_is_terminal, step_size, alpha, gamma):
        self.state_is_terminal = state_is_terminal
        self.alpha = alpha
        self.gamma = gamma
        self.step_size = step_size

        self.time_save_weights = []
        self.time_evaluate_t1 = []
        self.time_evaluate_run = []
        self.time_perform_sgd_t1 = []
        self.time_perform_sgd_run = []
        self.time_learn_t1 = []
        self.time_learn_run = []

        self.initialize()


    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def initialize(self):
        self.round_counter = 0
        self.C = 10

        self.y = tf.placeholder(tf.float32, shape=[None])
        self.actions = tf.placeholder(tf.float32, shape=[None, 3, 3])

        # state_seq
        self.phi = tf.placeholder(tf.float32, shape=[None, 3, 3, self.step_size])
        self.output = self.define_network(self.phi)

        output_evaluated = tf.reduce_sum(self.output * self.actions, axis=[1, 2])
        self.loss = tf.reduce_sum(tf.squared_difference(self.y, output_evaluated))

        self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

        self.output_action = tf.argmax(tf.reshape(self.output, [-1, 9])[0])
        action_ = tf.argmax(tf.reshape(self.actions, [-1, 9])[0])
        correct_prediction = tf.equal(self.output_action, action_)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # save value of weights.
        self.save_weights()


    def define_network(self, phi):
        self.W_conv1 = weight_variable([3, 3, self.step_size, self.step_size * 4])
        self.b_conv1 = bias_variable([3, 3, self.step_size * 4])
        h_conv1 = tf.nn.relu(conv2d(phi, self.W_conv1) + self.b_conv1)

        self.W_conv2 = weight_variable([3, 3, self.step_size * 4, self.step_size * 8])
        self.b_conv2 = bias_variable([3, 3, self.step_size * 8])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2) + self.b_conv2)

        h_conv2_reshaped = tf.reshape(h_conv2, [-1, 3 * 3 * self.step_size * 8])
        self.W_fcn = weight_variable([3 * 3 * self.step_size * 8, 9])
        self.b = bias_variable([9])

        y = tf.matmul(h_conv2_reshaped, self.W_fcn) + self.b

        output_ = tf.nn.softmax(y)

        output = tf.reshape(output_, [-1, 3, 3])

        return output

    def save_weights(self):
        t0 = time.time()
        self.W_conv1_target = self.sess.run(self.W_conv1)
        self.b_conv1_target = self.sess.run(self.b_conv1)
        self.W_conv2_target = self.sess.run(self.W_conv2)
        self.b_conv2_target = self.sess.run(self.b_conv2)
        self.W_fcn_target = self.sess.run(self.W_fcn)
        self.b_target = self.sess.run(self.b)
        t1 = time.time()
        self.time_save_weights.append(t1 - t0)

    def evaluate(self, phi, target_network=False):
        # returns the argmax action for given phi
        t0 = time.time()
        phi_ = self.transformation1(phi)
        t1 = time.time()
        if not target_network:
            feed_dict = {self.phi: phi_}
        else:
            feed_dict = {self.phi: phi_,
                         self.W_conv1: self.W_conv1_target,
                         self.b_conv1: self.b_conv1_target,
                         self.W_conv2: self.W_conv2_target,
                         self.b_conv2: self.b_conv2_target,
                         self.W_fcn: self.W_fcn_target,
                         self.b: self.b_target
                         }
        output_value = self.sess.run(self.output, feed_dict=feed_dict)[0]
        t2 = time.time()

        self.time_evaluate_t1.append(t1 - t0) # Result: takes too long
        self.time_evaluate_run.append(t2 - t1)
        return output_value

    def perform_sgd(self, y_, phi_, actions_):
        t0 = time.time()
        phi_2 = self.transformation2(phi_)
        t1 = time.time()
        batch_dict = {self.y: y_, self.phi: phi_2, self.actions: actions_}
        self.sess.run(self.train_step, feed_dict=batch_dict)
        t2 = time.time()

        self.time_perform_sgd_t1.append(t1 - t0) # Result: Takes too long
        self.time_perform_sgd_run.append(t2 - t1)
        # train_accuracy = self.sess.run(self.accuracy, feed_dict=batch_dict)
        # train_loss = self.sess.run(self.loss, feed_dict=batch_dict)
        # print('train accuracy %g, train loss %g' % (train_accuracy, train_loss))

    def learn(self, minibatch):
        self.round_counter += 1

        # calculate output vector for stored weights (see save_weights):
        y = []
        actions = []
        phi = []
        for batch in minibatch:
            value = self.calculate_y(batch)
            y.append(value)
            actions.append(actions_to_matrix(batch[1]))
            phi.append(batch[0])

        self.perform_sgd(y, phi, actions)

        if self.round_counter % self.C == 0:
            self.save_weights()

    def calculate_y(self, batch, DDQN=True):
        if DDQN:
            return self.calculate_y_DDQN(batch)

        # Now additionally with Double DQN
        if self.state_is_terminal(batch[3][-1]):
            value = batch[2]
        else:
            t0 = time.time()

            # switch to target network parameters
            feed_dict = {self.phi: self.transformation1(batch[3]),
                         self.W_conv1: self.W_conv1_target,
                         self.b_conv1: self.b_conv1_target,
                         self.W_conv2: self.W_conv2_target,
                         self.b_conv2: self.b_conv2_target,
                         self.W_fcn: self.W_fcn_target,
                         self.b: self.b_target}

            t1 = time.time()
            output_value = self.sess.run(self.output, feed_dict=feed_dict)
            t2 = time.time()
            self.time_learn_t1.append(t1 - t0)
            self.time_learn_run.append(t2 - t1)
            max_value = np.max(output_value)

            value = batch[2] + self.gamma * max_value
        return value

    @staticmethod
    def map_action_value_to_action_vector(action_value):
        return [int(action_value / 3), action_value % 3]

    def calculate_y_DDQN(self, batch):
        # Now additionally with Double DQN
        if self.state_is_terminal(batch[3][-1]):
            value = batch[2]
        else:
            # get max_action on online network
            feed_dict = {self.phi: self.transformation1(batch[3])}

            action_value = self.sess.run(self.output_action, feed_dict=feed_dict)
            max_action = self.map_action_value_to_action_vector(action_value)

            feed_dict = {self.phi: self.transformation1(batch[3]),
                         self.W_conv1: self.W_conv1_target,
                         self.b_conv1: self.b_conv1_target,
                         self.W_conv2: self.W_conv2_target,
                         self.b_conv2: self.b_conv2_target,
                         self.W_fcn: self.W_fcn_target,
                         self.b: self.b_target}

            output_value = self.sess.run(self.output, feed_dict=feed_dict)

            target_value = output_value[action_value[0], action_value[1]]

            value = batch[2] + self.gamma * target_value
        return value

    def transformation1(self, phi):
        phi_np = np.asarray([phi])
        return np.swapaxes(np.swapaxes(phi_np, 1, 2), 2, 3)

    def transformation2(self, phi):
        phi_np = np.asarray(phi)
        return np.swapaxes(np.swapaxes(phi_np, 1, 2), 2, 3)