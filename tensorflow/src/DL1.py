import tensorflow as tf

matrix1 = tf.constant([[3., 4., 6.], [3., 4., 8.]])

matrix2 = tf.constant([[2., 5.],[2., 5.],[2., 5.]])

product = tf.matmul(matrix1, matrix2)

matrix33 = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

matrix_reshaped = tf.reshape(matrix33, [-1, 9])

sess = tf.Session()
result = sess.run(product)
matrix_reshaped_eval = sess.run(matrix_reshaped)

print(result)
print(matrix_reshaped_eval)