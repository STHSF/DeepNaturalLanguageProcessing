#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: attention.py
@time: 2020/1/15 3:09 下午
"""
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, axis=2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.contrib.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def attention_test(rnn_output, attention_size):
    # https://github.com/cjymz886/text_rnn_attention/blob/master/text_model.py
    if isinstance(rnn_output, tuple):
        rnn_output = tf.concat(rnn_output, 2)

    # Attention Layer
    with tf.name_scope('attention'):
        input_shape = rnn_output.shape  # (batch_size, sequence_length, hidden_size)
        sequence_size = input_shape[1].value  # the length of sequences processed in the RNN layer
        hidden_size = input_shape[2].value  # hidden size of the RNN layer
        attention_w = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.1),
                                  name='attention_w')
        attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
        attention_u = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name='attention_u')
        z_list = []
        for t in range(sequence_size):
            u_t = tf.tanh(tf.matmul(rnn_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
            z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
            z_list.append(z_t)
        # Transform to batch_size * sequence_size
        attention_z = tf.concat(z_list, axis=1)
        alpha = tf.nn.softmax(attention_z)
        attention_output = tf.reduce_sum(rnn_output * tf.reshape(alpha, [-1, sequence_size, 1]), 1)

    return attention_output

