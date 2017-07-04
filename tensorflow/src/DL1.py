import tensorflow as tf

matrix1 = tf.constant([[3., 4., 6.], [3., 4., 8.]])

matrix2 = tf.constant([[2., 5.],[2., 5.],[2., 5.]])

product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)