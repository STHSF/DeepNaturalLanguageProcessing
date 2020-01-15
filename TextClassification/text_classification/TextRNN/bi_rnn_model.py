#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: rnn_model.py
@time: 2018/3/27 下午5:41
"""
import tensorflow as tf
from attention import attention
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TBRNNConfig(object):
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10   # 类别数
    vocab_size = 5000  # 词汇表大小

    num_layers = 2  # 隐藏层层数
    hidden_size = 2
    hidden_dim = 128  # 隐藏神经单元个数
    rnn = 'gru'  # lstm 或 gru

    attention_size = 50

    dropout_keep_prob = 0.8
    learning_rate = 1e-3

    batch_size = 128
    num_epochs = 10

    print_per_batch = 100
    save_per_batch = 10


class TextBiRNN(object):

    def __init__(self, config):
        self.config = config
        self._build_graph()

    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

    def gru_cell(self):
        return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

    def _build_graph(self):
        with tf.variable_scope("Input_data"):
            # input_x:[batch_size, seq_length]
            self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
            # input_y:[batch_size, num_classes]
            self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.device('/cpu:0'):
            # embedding:[vocab_size, embedding_dim]
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            self.embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("BiRNN"):
            # define lstm cess:get lstm cell output
            if self.config.rnn == 'lstm':
                rnn_fw_cell = self.lstm_cell()  # forward direction cell
                rnn_bw_cell = self.lstm_cell()  # backward direction cell
            else:
                rnn_fw_cell = self.gru_cell()  # forward direction cell
                rnn_bw_cell = self.gru_cell()  # backward direction cell
            if self.dropout_keep_prob is not None:
                rnn_fw_cell = tf.contrib.rnn.DropoutWrapper(rnn_fw_cell, output_keep_prob=self.dropout_keep_prob)
                rnn_bw_cell = tf.contrib.rnn.DropoutWrapper(rnn_bw_cell, output_keep_prob=self.dropout_keep_prob)
            # bidirectional_dynamic_rnn:
            # input: [batch_size, max_time, input_size]
            # output: A tuple (outputs, output_states)
            # where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, rnn_bw_cell, self.embedding_inputs, dtype=tf.float32)  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
            print("outputs:===>", outputs)  # outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
            # 3. concat output
            self.output_rnn = tf.concat(outputs, axis=2)  # [batch_size, sequence_length, hidden_size*2]
            # 4.1 average
            # self.output_rnn_last=tf.reduce_mean(output_rnn,axis=1) #[batch_size, hidden_size*2]
            # 4.2 last output
            output_rnn_last = self.output_rnn[:, -1, :]  # [batch_size, hidden_size*2]
            # print("output_rnn_last:", output_rnn_last)  # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
            # 5. logits(use linear layer)

        with tf.name_scope('Attention_layer'):
            # Attention layer
            attention_output, self.alphas = attention(self.output_rnn, self.config.attention_size, return_alphas=True)
            output_rnn_last = tf.nn.dropout(attention_output, self.dropout_keep_prob)

        with tf.name_scope("Score"):
            # 全连接层
            fc = tf.layers.dense(output_rnn_last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("Optimizer"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("Accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
