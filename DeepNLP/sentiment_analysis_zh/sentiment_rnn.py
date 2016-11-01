#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用RNN
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import input_data
import globe
import matplotlib.pyplot as plt

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# get data
training_data = input_data.read_data_sets()

# hyper_parameters
lr = 0.001
training_iters = 100000
batch_size = 100

n_inputs = 1  # data input size，输入层神经元
n_steps = globe.n_dim  # time steps， w2v 维度
n_hidden_units = 200  # neurons in hidden layer，隐藏层神经元个数
n_classes = 2  # classes 二分类

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

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


def rnn(input_data, weights, biases, is_training=True):
    keep_prob = 1
    num_layers = 2
    # hidden layer for input to cell
    ########################################

    # print "[前]", input_data

    # transpose the inputs shape from
    # X ==> (100 batch * 200 steps, 1 inputs)
    # x ==> 每次循环提供100篇文档作为输入，每篇文档是一个200维度的向量，
    input_data = tf.reshape(input_data, [-1, n_inputs])

    # into hidden
    # data_in = (100 batch * 200 steps, 100 hidden)
    data_in = tf.matmul(input_data, weights['in']) + biases['in']

    # data_in ==> (100 batch, 200 steps, 100 hidden)
    data_in = tf.reshape(data_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # DropoutWrapper
    if is_training and keep_prob < 1:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    # lstm cell is divided into two parts (c_state, h_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as data_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, data_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


predict = rnn(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, y))
train = tf.train.AdamOptimizer(lr).minimize(cost)

correct_predict = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    acc_array = []

    while step * batch_size < training_iters:
        batch_xs, batch_ys = training_data.train.next_batch(batch_size)
        print batch_xs
        # print '【前】', batch_xs.shape
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

        sess.run([train], feed_dict={x: batch_xs, y: batch_ys})

        # accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        acc_array.append(acc)
        if step % 20 == 0:
            prediction_value = sess.run(predict, feed_dict={x: batch_xs, y: batch_ys})
            # plot the prediction
            # lines = ax.plot(batch_xs, prediction_value, 'r-', lw=2)
            print acc
        step += 1

    # 模型保存

    # saver_path = saver.save(sess, "/home/zhangxin/work/workplace_python/DeepSentiment/data/rnn_model/model.ckpt")
    # print "Model saved in file: ", saver_path

    # plot accuracy
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lines = ax.plot(acc_array, '-', lw=2)
    y_text = ax.ylabel('精度')
    ax.setp(y_text, size='medium', name='helvetica', weight='light', color='r')
    plt.show()

# # 模型保存
# saver = tf.train.Saver()
# saver_path = saver.save(sess, "/home/zhangxin/work/workplace_python/DeepSentiment/data/rnn_model")
# print "Model saved in file: ", saver_path
#
# # plot accuracy
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# lines = ax.plot(acc_array, '-', lw=2)
# y_text = ax.ylabel('Precision')
# ax.setp(y_text, size='medium', name='helvetica', weight='light', color='r')
# plt.show()
