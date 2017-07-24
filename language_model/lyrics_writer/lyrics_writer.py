#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: language_model.py
@time: 2017/7/20 上午8:41
"""
import time
import numpy as np
import tensorflow as tf

file_path = './data/split.txt'
with open(file_path) as f:
    text = f.read()
data = text.split()
# print(u'词的个数：', len(data))
print(format(data[:10]))

# 使用set对列表去重，并保持列表原来的顺序
vocab = list(set(data))
vocab.sort(key=data.index)

vocab_to_id = {char: id for id, char in enumerate(vocab)}
id_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_id[c] for c in data])


def get_batch(raw_data, batch_size, seq_length):
    data = np.array(raw_data)
    data_length = data.shape[0]
    num_steps = data_length - seq_length + 1
    print('num_steps', num_steps)
    iterations = num_steps // batch_size
    print('iterations', iterations)
    xdata=[]
    ydata=[]
    for i in range(num_steps-1):
        xdata.append(data[i:i+seq_length])
        ydata.append(data[i+1:i+1+seq_length])

    for batch in range(iterations):
        x = np.array(xdata)[batch * batch_size: batch * batch_size + batch_size, :]
        y = np.array(xdata)[batch * batch_size + 1: batch * batch_size + 1 + batch_size, :]
        yield x, y


class language_model():

    def __init__(self, vocab_size, embed_dim, batch_size,
                 seq_length, input_dim, hidden_units, keep_prob,
                 num_layers,learning_rate, grad_clip, is_training):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.num_layers = num_layers
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate

    def input_layer(self):
        self.input_x = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_length], name='input_y')

    def word_embed(self, input_data):
        embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim], -1, 1))
        embed = tf.nn.embedding_lookup(embedding, input_data)
        return embed

    def lstm_cell(self):
        with tf.variable_scope('lstm_cell'):
            single_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units, state_is_tuple=True)

        if self.is_training:
            single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        return single_cell

    def add_multi_cell(self):

        embed = self.word_embed(self.input_x)
        with tf.variable_scope('stacked_cell'):
            stacked_cells = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)],
                                                        state_is_tuple=True)

        with tf.name_scope('initial_state'):
            self.initial_state = stacked_cells.zero_state(self.batch_size, dtype=tf.float32)

        self.cell_outputs, self.final_state = tf.nn.dynamic_rnn(cell=stacked_cells,
                                                                inputs=embed,
                                                                initial_state=self.initial_state)

    def output_layer(self):

        self.logits = tf.contrib.layers.fully_connected(self.cell_outputs, self.vocab_size, activation=None)
        self.prediction = tf.nn.softmax(self.logits, name='prediction')
        return self.logits, self.prediction

    def compute_loss(self):
        targets = self.word_embed(self.input_y)
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, targets, tf.ones())
        return self.loss

    def optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        return self.train_op

