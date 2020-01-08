#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: rnn_model.py
@time: 2018/3/27 下午5:41
"""

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TRNNConfig(object):
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10   # 类别数
    vocab_size = 5000  # 词汇表大小

    num_layers = 2  # 隐藏层层数
    hidden_dim = 128  # 隐藏神经单元个数
    rnn = 'gru'  # lstm 或 gru

    dropout_keep_prob = 0.8
    learning_rate = 1e-3

    batch_size = 128
    num_epochs = 10

    print_per_batch = 100
    save_per_batch = 10


class TextRNN(object):

    def __init__(self, config):
        self.config = config
        self._build_graph()

    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

    def gru_cell(self):
        return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

    def dropout(self):
        if self.config.rnn == 'lstm':
            cell = self.lstm_cell()
        else:
            cell = self.gru_cell()
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

    def _build_graph(self):
        with tf.variable_scope("input_data"):
            # input_x:[batch_size, seq_length]
            self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
            # input_y:[batch_size, num_classes]
            self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.variable_scope('embedding'):
            # embedding:[vocab_size, embedding_dim]
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_imputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            cells = [self.dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            self._outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_imputs, dtype=tf.float32)
            print('shape_of_outputs: %s' % self._outputs.get_shape())
            last = self._outputs[:, -1, :]    # 取最后一个时序输出作为结果
            # print('shape_of_outputs: %s' % last.get_shape())

        with tf.name_scope("score"):
            # 全连接层
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimizer"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
