# Implementation of deep neural network (and resp. interface) for Q-function approximation
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

def conv1d(x, W):
  return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def actions_to_matrix(action):
    matrix = np.zeros([3, 3])
    matrix[action[0], action[1]] = 1
    return matrix

class Network():
    def __init__(self, alpha, seq_size):
        self.alpha = alpha
        self.seq_size = seq_size # the state must still be one part of the market history

        self.initialize()

    def initialize(self):
        # state_seq batch
        self.state = tf.placeholder(tf.float32, shape=[None, self.seq_size])
        # action_seq batch
        self.action = tf.placeholder(tf.float32, shape=[None, 3])
        # reward_seq batch
        self.R = tf.placeholder(tf.float32, shape=[None, 1])


        # Drop-out keep probability
        self.keep_prob = tf.placeholder(tf.float32)

        # Define network
        self.policy_function, self.value_function, self.weights_policy, self.weights_value, self.weights_common\
            = self.define_network(self.state)

        # policy gradients
        policy_at_action = tf.reduce_sum(self.policy_function * self.actions, axis=[1])

        policy_log_gradients = tf.gradients(tf.log(policy_at_action), self.weights_policy)

        policy_log_gradients_common = tf.gradients(tf.log(policy_at_action), self.weights_common)

        policy_log_gradients_unreduced = policy_log_gradients * (self.R - self.value_function)

        # add entropy term
        if self.beta is None:
            self.loss_entropy = tf.Variable(tf.constant(0.0, shape=[1]))
        else:
            self.loss_entropy = self.beta * tf.reduce_sum(-self.policy_function * tf.log(self.policy_function), axis=[1])
            policy_log_gradients_common += self.loss_entropy
            policy_log_gradients_unreduced += self.loss_entropy

        policy_log_gradients_unreduced_common = policy_log_gradients_common * (self.R - self.value_function)

        self.policy_gradients = tf.reduce_sum(policy_log_gradients_unreduced, axis=[0])

        self.policy_gradients_common = tf.reduce_sum(policy_log_gradients_unreduced_common, axis=[0])

        # value gradients
        self.value_gradients = tf.reduce_sum(tf.gradients(tf.square(self.R - self.value_function), self.weights_value + self.weights_common), axis=[0])

        self.value_gradients_common = tf.reduce_sum(tf.gradients(tf.square(self.R - self.value_function), self.weights_common), axis=[0])


        self.policy_argmax = tf.argmax(self.policy_function)

        # common gradients
        self.common_gradients = self.policy_gradients_common + self.value_gradients_common


    def define_network(self, state):
        # Try a CNN
        input = tf.reshape(state, [-1, self.seq_size, 1])

        self.output0 = input

        self.output1, self.W_conv1, self.b_conv1 = self.make_layer(self.output0, 12, 1, 16)

        self.output2, self.W_conv2, self.b_conv2 = self.make_layer(self.output1, 8, 16, 20)

        self.output3, self.W_conv3, self.b_conv3 = self.make_layer(self.output2, 4, 20, 24)

        output3_size = self.seq_size * 24
        hidden3 = tf.reshape(self.output3, [-1, output3_size])

        # construct policy_function
        self.Wfcn_policy = weight_variable([output3_size, 3])
        self.bfcn_policy = bias_variable([3])
        o_policy = tf.matmul(hidden3, self.Wfcn_policy) + self.bfcn_policy

        policy_function = tf.nn.softmax(o_policy)

        self.Wfcn_value = weight_variable([output3_size, 1])
        self.bfcn_value = bias_variable([1])
        value_function = tf.matmul(hidden3, self.Wfcn_value) + self.bfcn_value

        weights_common = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3]
        weights_policy = [self.Wfcn_policy, self.bfcn_policy]
        weights_value = [self.Wfcn_value, self.bfcn_value]

        return policy_function, value_function, weights_policy, weights_value, weights_common

    def make_layer(self, input, conv_size, input_size, output_size):
        W_conv = weight_variable([conv_size, input_size, output_size])
        b_conv = bias_variable([input.shape[1], output_size])

        conv_ = conv1d(input, W_conv) + b_conv

        h_conv = tf.nn.relu(conv_)

        #dropout = tf.nn.dropout(h_conv, self.keep_prob) # self missing

        output = h_conv

        return output, W_conv, b_conv#, dropout

    def update_weights(self, weights):
        self.weights_policy_value, self.weights_value_value, self.weights_common_value = weights

    def evaluate_policy(self, state):
        # return max arg action
        # evaluate value function at given state
        state_ = self.transformation1(state)
        feed_dict = {self.state: state_}

        self.feed_weights(feed_dict)

        policy_value_ = self.sess.run(self.policy_argmax, feed_dict=feed_dict)
        policy_value = policy_value_[0]
        print(policy_value)

        return policy_value

    def evaluate_value(self, state):
        # evaluate value function at given state
        state_ = self.transformation1(state)
        feed_dict = {self.state: state_}

        self.feed_weights(feed_dict)

        output_value_ = self.sess.run(self.value_function, feed_dict=feed_dict)
        output_value = output_value_[0]
        print(output_value)

        return output_value

    def feed_weights(self, feed_dict):
        for weight, weight_value in zip(self.weights_common, self.weights_common_value):
            feed_dict[weight] = weight_value
        for weight, weight_value in zip(self.weights_value, self.weights_value_value):
            feed_dict[weight] = weight_value
        for weight, weight_value in zip(self.weights_policy, self.weights_policy_value):
            feed_dict[weight] = weight_value

    def calc_gradients(self, state, action, R):
        # evaluate gradients tensors at given state, action, R(eward accumluated)

        state_ = self.transformation1(state)
        feed_dict = {self.state: state_,
                     self.action: action,
                     self.R: R}

        self.feed_weights(feed_dict)

        policy_gradients, value_gradients, common_gradients \
            = self.sess.run([self.policy_gradients, self.value_gradients, self.common_gradients], feed_dict=feed_dict)


        return policy_gradients, value_gradients, common_gradients

    def transformation1(self, phi):
        phi_np = np.asarray([phi])
        result = phi_np[:,:, 1]
        return result

    def transformation2(self, phi):
        phi_np = np.asarray(phi)
        result = phi_np[:,:, 1]
        return result