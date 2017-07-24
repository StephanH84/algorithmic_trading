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
    def __init__(self, state_is_terminal, step_size, alpha, gamma, theta, C, beta, activation, noDDQN):
        self.state_is_terminal = state_is_terminal
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.C = C
        self.seq_size = step_size
        self.keep_prob_value = 0.95
        self.noDDQN = noDDQN


        self.time_save_weights = []
        self.time_evaluate_t1 = []
        self.time_evaluate_run = []
        self.time_perform_sgd_t1 = []
        self.time_perform_sgd_run = []
        self.time_learn_t1 = []
        self.time_learn_run = []
        self.time_learn_make = []

        if activation == "tanh":
            activation_fn = tf.nn.tanh
        else:
            activation_fn = tf.nn.relu
        self.initialize(activation_fn)


    def __del__(self):
        try:
            self.sess.close()
        except:
            pass

    def initialize(self, activation):
        self.round_counter = 0

        # Define placeholders
        self.y = tf.placeholder(tf.float32, shape=[None])
        self.actions = tf.placeholder(tf.float32, shape=[None, 3])

        # state_seq
        self.phi = tf.placeholder(tf.float32, shape=[None, self.seq_size])

        self.keep_prob = tf.placeholder(tf.float32)

        self.phase = tf.placeholder(tf.bool)

        # Define network
        self.output = self.define_fft_network(self.phi, self.phase, activation)

        output_evaluated = tf.reduce_sum(self.output * self.actions, axis=[1])
        self.loss = tf.reduce_sum(tf.squared_difference(self.y, output_evaluated))
        # add entropy term
        if self.beta is None:
            self.loss_entropy = bias_variable([1])
        else:
            self.loss_entropy = self.beta * tf.reduce_sum(-self.output * tf.log(self.output))
            self.loss += self.loss_entropy

        self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

        self.output_action = tf.argmax(self.output[0])
        action_ = tf.argmax(self.actions[0])
        correct_prediction = tf.equal(self.output_action, action_)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # save value of weights.
        self.save_weights()


    def define_network_cnn(self, phi, phase):
        # Try a CNN
        input = tf.reshape(phi, [-1, self.seq_size, 1])

        self.W_conv1 = weight_variable([8, 1, 10])
        self.b_conv1 = bias_variable([self.seq_size, 10])
        self.h_conv1 = tf.nn.relu(conv1d(input, self.W_conv1) + self.b_conv1)

        self.drop1 = tf.nn.dropout(self.h_conv1, self.keep_prob)

        self.W_conv2 = weight_variable([4, 10, 15])
        self.b_conv2 = bias_variable([self.seq_size, 15])
        self.h_conv2 = tf.nn.relu(conv1d(self.h_conv1, self.W_conv2) + self.b_conv2)

        self.drop2 = tf.nn.dropout(self.h_conv2, self.keep_prob)

        hidden2 = tf.reshape(self.h_conv2, [-1, self.seq_size * 15])

        self.Wfcn = weight_variable([self.seq_size * 15, 3])
        self.bfcn = bias_variable([3])
        self.o3 = tf.matmul(hidden2, self.Wfcn) + self.bfcn


        hidden3 = self.o3 # tf.nn.elu(self.o3)

        self.output = tf.nn.softmax(hidden3)

        self.params = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2]
        self.params.extend([self.Wfcn, self.bfcn])

        return self.output


    def define_network_cnn_bn(self, phi, phase, activation):
        # Try a CNN
        input = tf.reshape(phi, [-1, self.seq_size, 1])

        self.bn0 = tf.contrib.layers.batch_norm(input, center=True, scale=True, is_training=phase)

        self.output0 = self.bn0

        self.output1, self.W_conv1, self.b_conv1, self.dropout1, self.bn1 = self.make_layer_1(self.output0, phase, 12, 1, 16, activation)

        self.output2, self.W_conv2, self.b_conv2, self.dropout2, self.bn2 = self.make_layer_1(self.output1, phase, 8, 16, 20, activation)

        self.output3, self.W_conv3, self.b_conv3, self.dropout3, self.bn3 = self.make_layer_1(self.output2, phase, 4, 20, 24, activation)

        output3_size = self.seq_size * 24
        hidden2 = tf.reshape(self.output3, [-1, output3_size])

        self.Wfcn = weight_variable([output3_size, 3])
        self.bfcn = bias_variable([3])
        o3 = tf.matmul(hidden2, self.Wfcn) + self.bfcn

        output = tf.nn.softmax(o3)

        self.params = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3]
        self.params.extend([self.Wfcn, self.bfcn])

        return output


    def make_layer_simple(self, input, size, activation):

        W = weight_variable([int(input.shape[1]), size])
        b = bias_variable([size])
        o3 = tf.matmul(input, W) + b

        h_conv = activation(o3)

        output = h_conv

        return output, W, b

    def define_fft_network(self, input, phase, activation, n_=4):
        assert input.shape[1] == 64 + (n_-1) * (64-5)

        # divide in 5 time-intervals:
        input_sliced = [input[:,n*59:n*59 + 64] for n in range(n_)]

        #input_sliced_fft = [self.get_fft_features(i_slice) for i_slice in input_sliced] # (5x2)xT[Nx64]
        input_sliced_fft = [[i_slice] for i_slice in input_sliced]
        self.input_sliced_fft = input_sliced_fft

        W_conv1 = weight_variable([4, 1, 32])
        b_conv1 = bias_variable([64, int(W_conv1.shape[2])])

        W_conv2 = weight_variable([4, 32, 32])
        b_conv2 = bias_variable([64, int(W_conv2.shape[2])])

        '''W_conv1_last = weight_variable([4, 1, 32])
        b_conv1_last = bias_variable([64, int(W_conv1_last.shape[2])])

        W_conv2_last = weight_variable([4, 32, 32])
        b_conv2_last = bias_variable([64, int(W_conv2_last.shape[2])])'''

        # each item is a tensor of shape Nx64
        cnn_fft = []

        self.layers = []
        for slice in input_sliced_fft:
            slice_conv = []
            for item in slice:
                hidden1, hidden2 = self.do_conv_2(item, W_conv1, b_conv1, W_conv2, b_conv2, tf.nn.relu)
                self.layers.extend([hidden1, hidden2])
                slice_conv.append(hidden2)

            # TODO: Same weights for amplitudes and phases?
            slice_conv_concat = tf.concat(slice_conv, axis=2)
            size1 = int(slice_conv_concat.shape[1]) * int(slice_conv_concat.shape[2])
            slice_conv_concat = tf.reshape(slice_conv_concat, shape=[-1, size1])
            cnn_fft.append(slice_conv_concat)

        abs_features_sliced = []
        for i_slice in input_sliced:
            abs_feature = self.get_abs_features(i_slice)
            abs_features_sliced.append(abs_feature)

        rnn_input_ = []
        for n in range(n_):
            feature_ = tf.concat([cnn_fft[n], abs_features_sliced[n]], axis=1)
            feature_ = tf.reshape(feature_, shape=[-1, 1, int(feature_.shape[1])])
            rnn_input_.append(feature_)

        rnn_input = tf.concat(rnn_input_, axis=1) # needs to have the shape: [None, 5, size of input_sliced_fft[n] + size of abs_features_sliced[n]]

        num_hidden = 16 # TODO: experiment
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        rnn_weights = rnn_cell.trainable_weights

        rnn_output, state = tf.nn.dynamic_rnn(rnn_cell, rnn_input, dtype=tf.float32)
        self.rnn_output = rnn_output

        rnn_output_reshaped = tf.reshape(rnn_output, shape=[-1, int(rnn_output.shape[1]) * int(rnn_output.shape[2])])

        fcn_output, Wfcn, bfcn = self.make_layer_simple(rnn_output_reshaped, 3, tf.nn.relu)
        self.fcn_output = fcn_output

        self.params = [W_conv1, b_conv1, W_conv2, b_conv2, Wfcn, bfcn] + rnn_weights

        output = tf.nn.softmax(fcn_output)

        return output

    def get_abs_features(self, input):
        # input: Nx64
        # max
        input_max = tf.reshape(tf.reduce_max(input, axis=[1]), shape=[-1, 1])
        # min
        input_min = tf.reshape(tf.reduce_min(input, axis=[1]), shape=[-1, 1])
        # mean
        input_mean = tf.reshape(tf.reduce_mean(input, axis=[1]), shape=[-1, 1])
        # regression slope
        slope = tf.reshape(self.calc_slope(input), shape=[-1, 1])


        return tf.concat([input_max, input_min, input_mean, slope], axis=1)

    def calc_slope(self, y):
        time_ = np.asarray([float(n) for n in range(int(y.shape[1]))])
        time_mean = np.mean(time_)
        y_mean = tf.reshape(tf.reduce_mean(y, axis=[1]), shape=[-1, 1])

        cov = tf.reduce_mean((time_ - time_mean) * (y - y_mean), axis=[1])
        var = tf.cast(tf.reduce_mean((time_ - time_mean) ** 2), tf.float32)

        return cov / var

    def do_conv_2(self, input, W_conv1, b_conv1, W_conv2, b_conv2, activation):
        # input has shape Nx64 and must be reshaped to Nx1x64
        input_ = tf.reshape(input, [-1, int(input.shape[1]), 1])

        conv1 = conv1d(input_, W_conv1) + b_conv1
        hidden1 = activation(conv1)

        conv2 = conv1d(hidden1, W_conv2) + b_conv2
        hidden2 = activation(conv2)
        return hidden1, hidden2

    def get_fft_features(self, input):
        # TODO: Work on the details

        input_ = tf.cast(input, tf.complex64)

        input_fft = tf.fft(input_)

        fft_ampl = tf.abs(input_fft)
        fft_phase = tf.atan2(tf.imag(input_fft), tf.real(input_fft))

        return [fft_ampl] #, fft_phase]

    def define_network_simple(self, phi, phase, activation):
        # Try a CNN
        input = phi # tf.reshape(phi, [-1, self.seq_size, 1])

        self.output0 = input

        self.output1, self.W_1, self.b_1 = self.make_layer_simple(self.output0, 100, activation)

        self.output2, self.W_2, self.b_2 = self.make_layer_simple(self.output1, 50, activation)

        self.output3, self.W_3, self.b_3 = self.make_layer_simple(self.output1, 3, activation)

        output = tf.nn.softmax(self.output3)

        self.params = [self.W_1, self.b_1, self.W_2, self.b_2, self.W_3, self.b_3]

        return output


    def define_network_cnn_bn_3(self, phi, phase):
        # Try a CNN
        input = tf.expand_dims(phi, 2)

        self.bn0 = tf.contrib.layers.batch_norm(input, center=True, scale=True, is_training=phase)

        self.output0 = self.bn0

        self.output1_, self.W_conv1, self.b_conv1, self.dropout1, self.bn1\
            = self.make_layer_1(self.output0, phase, 16, 1, 16) # 9, 1, 16)
        self.output1 = self.pooling(self.output1_)

        self.output2_, self.W_conv2, self.b_conv2, self.dropout2, self.bn2 = self.make_layer_1(self.output1, phase, 8, 16, 32)
        # 4, 16, 32) #
        self.output2 = self.pooling(self.output2_)

        self.output3_, self.W_conv3, self.b_conv3, self.dropout3, self.bn3 = self.make_layer_1(self.output2, phase, 4, 32, 64)
        self.output3 = self.pooling(self.output3_)

        output3_shape1 = int(self.output3.shape[1])
        output3_shape2 = int(self.output3.shape[2])
        hidden2 = tf.reshape(self.output3, [-1, output3_shape1 * output3_shape2])

        self.Wfcn = weight_variable([output3_shape1 * output3_shape2, 3])
        self.bfcn = bias_variable([3])
        o3 = tf.matmul(hidden2, self.Wfcn) + self.bfcn

        hidden3 = tf.nn.relu(o3)

        output = tf.nn.softmax(hidden3)

        self.params = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2] #, self.W_conv3, self.b_conv3]
        self.params.extend([self.Wfcn, self.bfcn])

        return output

    def make_layer_1(self, input, phase, conv_size, input_size, output_size, activation):
        W_conv = weight_variable([conv_size, input_size, output_size])
        b_conv = bias_variable([input.shape[1], output_size])

        conv_ = conv1d(input, W_conv) + b_conv
        bn = tf.contrib.layers.batch_norm(conv_, center=True, scale=True, is_training=phase)

        h_conv = activation(bn)

        dropout = tf.nn.dropout(h_conv, self.keep_prob) # self missing

        output = h_conv

        return output, W_conv, b_conv, dropout, bn

    def pooling(self, input):
        input4 = tf.expand_dims(input, 2)
        output4 = tf.nn.avg_pool(input4, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
        output = tf.squeeze(output4, [2])
        return output

    def define_network(self, phi, phase):
        # Try a fully contected NN

        input_size = self.seq_size
        input = phi

        self.W0 = weight_variable([input_size, 100])
        self.b0 = bias_variable([100])
        o0 = tf.matmul(input, self.W0) + self.b0
        hidden0 = tf.nn.relu(o0)

        self.drop0 = tf.nn.dropout(hidden0, self.keep_prob)

        self.W1 = weight_variable([100, 50])
        self.b1 = bias_variable([50])
        o1 = tf.matmul(hidden0, self.W1) + self.b1
        hidden1 = tf.nn.relu(o1)

        self.drop1 = tf.nn.dropout(hidden1, self.keep_prob)


        '''self.W2 = weight_variable([128, 256])
        self.b2 = bias_variable([256])
        o2 = tf.matmul(hidden1, self.W2) + self.b2
        hidden2 = tf.nn.relu(o2)

        self.drop2 = tf.nn.dropout(hidden2, self.keep_prob)'''


        self.W3 = weight_variable([50, 3])
        self.b3 = bias_variable([3])
        o3 = tf.matmul(hidden1, self.W3) + self.b3
        hidden3 = tf.nn.relu(o3)

        output = tf.nn.softmax(hidden3)

        self.params = [self.W0, self.b0, self.W1, self.b1] #, self.W2, self.b2,
        self.params.extend([self.W3, self.b3])

        return output

    def save_weights(self):
        t0 = time.time()

        self.target_params = []

        for param in self.params:
            value = self.sess.run(param)
            self.target_params.append(value)

        t1 = time.time()
        self.time_save_weights.append(t1 - t0)


    def slide_weights(self):
        t0 = time.time()

        self.new_target_params = []

        for param, target_param in zip(self.params, self.target_params):
            value = self.sess.run(param)
            new_value = self.theta * value + (1 - self.theta) * target_param
            self.new_target_params.append(value)

        self.target_params = self.new_target_params

        t1 = time.time()
        self.time_save_weights.append(t1 - t0)

    def evaluate(self, phi, target_network=False):
        # returns the argmax action for given phi
        t0 = time.time()
        phi_ = self.transformation1(phi)
        t1 = time.time()
        if not target_network:
            feed_dict = {self.phi: phi_,
                         self.keep_prob: 1.0}
        else:
            feed_dict = {self.phi: phi_,
                         self.keep_prob: 1.0}

            for param, target_param in zip(self.params, self.target_params):
                feed_dict[param] = target_param

        feed_dict[self.phase] = 0

        output_value_ = self.sess.run(self.output, feed_dict=feed_dict)
        output_value = output_value_[0]
        print(output_value)

        if self.round_counter > 1 and self.round_counter % 20 == 0:
            debug_value = self.sess.run([self.rnn_output, self.fcn_output] + self.layers + self.params, feed_dict=feed_dict)
            assert True

        t2 = time.time()

        self.time_evaluate_t1.append(t1 - t0) # Result: takes too long
        self.time_evaluate_run.append(t2 - t1)
        return output_value

    def perform_sgd(self, y_, phi_, actions_):
        t0 = time.time()
        phi_2 = self.transformation2(phi_)
        t1 = time.time()
        batch_dict = {self.y: y_, self.phi: phi_2, self.actions: actions_, self.keep_prob: self.keep_prob_value}
        batch_dict[self.phase] = 1
        self.sess.run(self.train_step, feed_dict=batch_dict)
        t2 = time.time()

        self.time_perform_sgd_t1.append(t1 - t0) # Result: Takes too long
        self.time_perform_sgd_run.append(t2 - t1)
        # train_accuracy = self.sess.run(self.accuracy, feed_dict=batch_dict)
        train_loss = self.sess.run(self.loss, feed_dict=batch_dict)
        loss_entropy = self.sess.run(self.loss_entropy, feed_dict=batch_dict)

        print('train loss %g, loss entropy %g' % (train_loss, loss_entropy))
        # print('train accuracy %g, train loss %g' % (train_accuracy, train_loss))

    def to_action_vector(self, action_value):
        vect = np.zeros((3))
        vect[action_value] = 1
        return vect

    def learn(self, minibatch, no_subsample=False):
        self.round_counter += 1

        # calculate output vector for stored weights (see save_weights):
        y = []
        actions = []
        phi = []

        t0 = time.time()
        for sample in minibatch:
            # split sample in three subsamples
            samples = []
            if not no_subsample:
                for i in range(3):
                    samples.append([sample[0], sample[1][i], sample[2][i], sample[3]])
            else:
                samples = [sample]

            for subsample in samples:
                action = subsample[1]
                phi_t = subsample[0]

                value = self.calculate_y(subsample)
                y.append(value)
                actions.append(self.to_action_vector(action))
                phi.append(phi_t)

        t1 = time.time()

        self.time_learn_make.append(t1 - t0)
        self.perform_sgd(y, phi, actions)

        if self.C is None or (self.round_counter % self.C == 0):
            self.slide_weights()

    def calculate_y(self, sample, DDQN=True):
        if self.noDDQN is not None:
            DDQN = not self.noDDQN

        if DDQN:
            return self.calculate_y_DDQN(sample)

        # Now additionally with Double DQN
        if self.state_is_terminal(sample[3][-1]):
            value = sample[2]
        else:
            t0 = time.time()

            # switch to target network parameters
            feed_dict = {self.phi: self.transformation1(sample[3]),
                         self.keep_prob: 1.0}

            for param, target_param in zip(self.params, self.target_params):
                feed_dict[param] = target_param

            feed_dict[self.phase] = 0

            t1 = time.time()
            output_value = self.sess.run(self.output, feed_dict=feed_dict)
            t2 = time.time()
            self.time_learn_t1.append(t1 - t0)
            self.time_learn_run.append(t2 - t1)
            max_value = np.max(output_value)

            value = sample[2] + self.gamma * max_value
        return value


    def calculate_y_DDQN(self, sample):
        # Now additionally with Double DQN

        phi_t1 = sample[3]
        reward = sample[2]

        if self.state_is_terminal(phi_t1[-1]):
            value = reward
        else:
            # get max_action on online network
            feed_dict = {self.phi: self.transformation1(phi_t1), self.keep_prob: 1.0}
            feed_dict[self.phase] = 0

            max_action = self.sess.run(self.output_action, feed_dict=feed_dict)

            feed_dict = {self.phi: self.transformation1(phi_t1),
                         self.keep_prob: 1.0}

            for param, target_param in zip(self.params, self.target_params):
                feed_dict[param] = target_param

            feed_dict[self.phase] = 0

            output_value = self.sess.run(self.output, feed_dict=feed_dict)[0]
            # print(output_value)

            target_value = output_value[max_action]

            value = reward + self.gamma * target_value
        return value

    def transformation1(self, phi):
        phi_np = np.asarray([phi])
        result = phi_np[:,:, 1]
        return result

    def transformation2(self, phi):
        phi_np = np.asarray(phi)
        result = phi_np[:,:, 1]
        return result