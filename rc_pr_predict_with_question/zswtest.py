import tensorflow as tf
import numpy as np

X = tf.random_normal(shape=[3, 5, 6], dtype=tf.float32)
X = tf.reshape(X, [-1, 5, 6])
print (X)
cell = tf.nn.rnn_cell.BasicLSTMCell(10)  # 也可以换成别的，比如GRUCell，BasicRNNCell等等
lstm_multi = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)
state = lstm_multi.zero_state(3, tf.float32)
output, state = tf.nn.dynamic_rnn(lstm_multi, X, initial_state=state, time_major=False)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print (output.get_shape())
    print (sess.run(state))