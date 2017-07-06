import tensorflow as tf

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

def define_network():
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

    output = tf.reshape(output_, [3, 3])

    return output



output = define_network() #model of phi with parameters theta


output_evaluated = tf.reduce_sum(output * actions, axis=[1, 2])
loss = tf.reduce_sum(tf.squared_difference(y, output_evaluated))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

output_action = tf.argmax(tf.reshape(output, [9]))
action_ = tf.argmax(tf.reshape(actions, [-1, 9]))
correct_prediction = tf.equal(output_action, action_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: function to evaluate Q-network
def learn():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch = None # TODO

        batch_dict = {y: batch[0], phi: batch[1], actions: batch[2]}
        train_step.run(feed_dict=batch_dict)
        train_accuracy = accuracy.eval(feed_dict=batch_dict)
        print('train accuracy %g' % train_accuracy)

def evaluate(phi):
    with tf.Session() as sess:
        result = sess.run(output, feed_dict={phi: phi})
        return result

