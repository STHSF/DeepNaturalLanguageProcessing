#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import input_data
import globe

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
train_dir = ''
training_data = input_data.read_data_sets(train_dir)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, weight):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def average_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def add_cnn_layer(inputs, patch, inputsize,outputsize):
    weight_conv_layer = weight_variable([5, 5, inputsize, outputsize])  # patch 5x5, input size 1, output size 32
    biases_conv_layer = bias_variable([outputsize])
    hidden_conv_layer = tf.nn.relu(conv2d(inputs, weight_conv_layer) + biases_conv_layer)  # output size 28x28x32
    hidden_pool = max_pool_2x2(hidden_conv_layer)  # output size 14x14x32
    return hidden_pool


def add_func_layer(inputs, inputsize, outputsize):
    height = inputsize[0], width = inputsize[1], depth = inputsize[2]
    weight_func = weight_variable([height * width * depth, outputsize])
    biases_func = bias_variable([outputsize])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    hidden_pool2_flat = tf.reshape(inputs, [-1, height * width * depth])
    hidden_func = tf.nn.relu(tf.matmul(hidden_pool2_flat, weight_func) + biases_func)
    hidden_func_drop = tf.nn.dropout(hidden_func, keep_prob)
    return hidden_func_drop

'''定义节点准备接收数据'''
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 200])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 200, 1, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]


# conv1 layer #
weight_conv_layer1 = weight_variable([5, 5, 1, 32])  # patch 5x5, input size 1, output size 32
biases_conv_layer1 = bias_variable([32])
hidden_conv_layer1 = tf.nn.relu(conv2d(x_image, weight_conv_layer1) + biases_conv_layer1)  # output size 28x28x32
hidden_pool_layer1 = max_pool_2x2(hidden_conv_layer1)  # output size 14x14x32


# conv2 layer #
weight_conv_layer2 = weight_variable([5, 5, 32, 64])  # patch 5x5, input size 32, output size 64
biases_conv_layer2 = bias_variable([64])
hidden_conv_layer2 = tf.nn.relu(conv2d(hidden_pool_layer1, weight_conv_layer2) + biases_conv_layer2)  # output size 14x14x64
hidden_pool_layer2 = max_pool_2x2(hidden_conv_layer2)  # output size 7x7x64


# func1 layer #
weight_func_layer1 = weight_variable([7 * 7 * 64, 1024])
bias_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
hidden_pool2_flat = tf.reshape(hidden_pool_layer2, [-1, 7 * 7 * 64])
hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, weight_func_layer1) + bias_fc1)
hidden_fc1_drop = tf.nn.dropout(hidden_fc1, keep_prob)

# func2 layer #
weight_fc2 = weight_variable([1024, 10])
bias_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(hidden_fc1_drop, weight_fc2) + bias_fc2)

# loss function
# 分类问题
# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = training_data.train.next_batch(100)
    sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            training_data.test.images, training_data.test.labels))
