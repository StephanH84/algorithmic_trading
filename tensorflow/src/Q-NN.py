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

# TODO: write minibatch interface
N = 10 # minibatch size

y = tf.placeholder(tf.float32, shape=[None])
actions = tf.placeholder(tf.float32, shape=[None, 3, 3])

# state_seq
phi = tf.placeholder(tf.float32, shape=[None, 3, 3, 4])

keep_prob = tf.placeholder(tf.float32)

def define_network():
    W_conv1 = weight_variable([3, 3, 4, 16])
    b_conv1 = bias_variable([3, 3, 16])
    h_conv1 = tf.nn.relu(conv2d(phi, W_conv1) + b_conv1)

    h_drop1 = tf.nn.dropout(h_conv1, keep_prob)

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



output = define_network() #model of phi with parameters theta

def evaluate(phi_):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(output, feed_dict={phi: phi_})
        return result

def test_evaluate():
    board = [[-1, 0, 1], [1, 0, 0], [1, 1, 0]]
    board_ = tf.constant([4 * [board]])
    phi_ = tf.transpose(board_, [0, 2, 3, 1])

    with tf.Session() as sess:
        phi_ = sess.run(phi_, feed_dict={keep_prob: 1.0})
    res = evaluate(phi_)
    return phi_


test_evaluate()
exit()

output_evaluated = tf.reduce_sum(output * actions, axis=[1, 2])
loss = tf.reduce_sum(tf.squared_difference(y, output_evaluated))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

output_action = tf.argmax(tf.reshape(output, [-1, 9])[0])
action_ = tf.argmax(tf.reshape(actions, [-1, 9])[0])
correct_prediction = tf.equal(output_action, action_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def learn(batch):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for episode in range(3):
            batch_dict = {y: batch[0], phi: batch[1], actions: batch[2], keep_prob: 0.7}
            train_step.run(feed_dict=batch_dict)
            train_accuracy = accuracy.eval(feed_dict=batch_dict)
            print('train accuracy %g' % train_accuracy)

def test_learn(phi_):
    y_batch = [1, 2]
    phi__ = [tf.constant(phi_)] * 2
    print(tf.size(phi__))
    phi_batch_ = tf.squeeze(phi__)
    with tf.Session() as sess:
        phi_batch = sess.run(phi_batch_)

    def actions_to_matrix(action):
        matrix = np.zeros([3, 3])
        matrix[action[0], action[1]] = 1
        return matrix


    actions_batch = [actions_to_matrix([0, 2]), actions_to_matrix([1, 0])]



    batch = [y_batch, phi_batch, actions_batch]
    learn(batch)
