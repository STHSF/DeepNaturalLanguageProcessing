#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
初步思路使用RNN
"""

import tensorflow as tf
import input_data
import globe
import matplotlib.pyplot as plt


# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# hyper_parameters
lr = 0.001
batch_size = 1

n_inputs = 1  # data input size，输入层神经元
n_steps = globe.n_dim  # time steps， w2v 维度
n_hidden_units = 200  # neurons in hidden layer，隐藏层神经元个数
n_classes = 2  # classes 二分类

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

# Define weights
weights = {
    # (1, 200)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (200, 2)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # (200, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (2, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def rnn(input_data, weights, biases):
    keep_prob = 1
    num_layers = 2
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (100 batch * 200 steps, 1 inputs)
    input_data = tf.reshape(input_data, [-1, n_inputs])

    # into hidden
    # data_in = (100 batch * 200 steps, 100 hidden)
    data_in = tf.matmul(input_data, weights['in']) + biases['in']

    # data_in ==> (100 batch, 200 steps, 100 hidden)
    data_in = tf.reshape(data_in, [-1, n_steps, n_hidden_units])

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # DropoutWrapper
    if keep_prob < 1:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    # lstm cell is divided into two parts (c_state, h_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as data_in.
    # Make sure the time_major is changed accordingly.  确保time_major相应改变
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, data_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    outputs_temp = tf.transpose(outputs, [1, 0, 2])

    # # or
    # unpack to list [(batch, outputs)..] * steps
    outputs = tf.unpack(outputs_temp)  # states is the last outputs   tf.transpose转置函数

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # outputs[-1]表示取最后一个

    return results

predict = rnn(x, weights, biases)

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, globe.model_rnn_path)

    data = input_data.read_data_sets_predict()

    for title in data.keys():
        batch_xs = data[title]
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        result = sess.run([predict], feed_dict={x: batch_xs})

        print '【标题】', title
        print '【LabelIndex】', result[0].argmax()
        print '【result】', result[0].tolist()[0], '\n'

