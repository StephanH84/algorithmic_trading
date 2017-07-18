import tensorflow as tf

data = tf.placeholder(tf.float32)
var = tf.Variable(tf.constant(0.1, shape=[5]))
loss = data * var**2 + data / 2 * var + 3

var_grad = tf.gradients(loss, [var])[0]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

var_grad_val = sess.run(var_grad, feed_dict={data: 4})

print(var_grad_val)