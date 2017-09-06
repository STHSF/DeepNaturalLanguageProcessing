# coding=utf-8

import tensorflow as tf
import numpy as np

hidden_units = 10
vocab_size = 1000
embedding_size = 4
layers_num = 10
batch_size = 5
num_steps = 4
num_classes = 4
# source_inputs = tf.placeholder(shape=(None, None), dtype=tf.float32, name='source_inputs')
# target_inputs = tf.placeholder(shape=(None, None), dtype=tf.float32, name='y_inputs')
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

source_inputs = tf.placeholder(shape=(batch_size, num_steps), dtype=tf.int32, name='source_inputs')
target_inputs = tf.placeholder(shape=(batch_size, num_steps), dtype=tf.int32, name='y_inputs')

embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
# shape = [batch_size, num_steps, embedding_size]
inputs_embedded = tf.nn.embedding_lookup(embedding, source_inputs)


def lstm_cell():
    cell = tf.contrib.rnn.LSTMCell(hidden_units)
    return cell


cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layers_num)], state_is_tuple=True)
cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layers_num)], state_is_tuple=True)

initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)


((outputs_fw,
  outputs_bw),
 (final_state_fw,
  final_state_bw)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                      cell_bw=cell_bw,
                                                      inputs=inputs_embedded,
                                                      initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      dtype=tf.float32,
                                                      time_major=False))
# shape = [batch_size, num_steps, hidden_units * 2]
outputs = tf.concat((outputs_fw, outputs_bw), 2)
outputs = tf.reshape(outputs, [-1, hidden_units*2])

# final_state_c = tf.concat((final_state_fw.c, final_state_bw.c), 1)
# final_state_h = tf.concat((final_state_fw.h, final_state_bw.h), 1)
# final_state = tf.contrib.rnn.LSTMStateTuple(c=final_state_c,
#                                             h=final_state_h)

with tf.variable_scope('outputs'):
    softmax_w = weight_variable([hidden_units * 2, num_classes])
    softmax_b = bias_variable([num_classes])
    y_pred = tf.matmul(outputs, softmax_w) + softmax_b







