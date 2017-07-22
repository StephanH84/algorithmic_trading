import tensorflow as tf


time = tf.constant([[0.01 * n for n in range(0, 200)], [0.01 * n for n in range(0, 200)]])

signal = 3 * tf.sin(2 * time) + 6 * tf.sin(4 * time) + 5 * tf.sin(8 * time)

signal_complex = tf.cast(signal, tf.complex64)

fft = tf.fft(signal_complex)

fft_ampl = tf.abs(fft)
fft_phase = tf.atan2(tf.imag(fft),tf.real(fft))

sess = tf.Session()
result = sess.run([fft_ampl, fft_phase, fft])

print(result)