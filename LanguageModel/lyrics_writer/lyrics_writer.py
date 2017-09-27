#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: LanguageModel.py
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
# print(format(data[:10]))

# 使用set对列表去重，并保持列表原来的顺序
vocab = list(set(data))
vocab.sort(key=data.index)
vocab_to_id = {char: id for id, char in enumerate(vocab)}
id_to_vocab = dict(enumerate(vocab))

encoded = np.array([vocab_to_id[c] for c in data])


# def get_batch(raw_data, batch_size, seq_length):
#     data = np.array(raw_data)
#     data_length = data.shape[0]
#     num_steps = data_length - seq_length + 1
#     iterations = num_steps // batch_size
#     xdata=[]
#     ydata=[]
#     for i in range(num_steps-1):
#         xdata.append(data[i:i+seq_length])
#         ydata.append(data[i+1:i+1+seq_length])
#
#     for batch in range(iterations):
#         x = np.array(xdata)[batch * batch_size: batch * batch_size + batch_size, :]
#         y = np.array(xdata)[batch * batch_size + 1: batch * batch_size + 1 + batch_size, :]
#         yield x, y

def get_batch(raw_data, batch_size, seq_length):
    """
    生成batch数据，
    Args:
        array:
        batch_size:
        seq_length:

    Returns:

    """
    data = np.array(raw_data)
    data_length = data.shape[0]
    num_batches = (data_length - 1) // (batch_size * seq_length)
    assert num_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = num_batches * (batch_size * seq_length)
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, num_batches * seq_length])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, num_batches * seq_length])

    for batch in range(num_batches):
        x = xdata[:, batch * seq_length:(batch + 1) * seq_length]
        y = ydata[:, batch * seq_length:(batch + 1) * seq_length]

        yield x, y


class language_model:

    def __init__(self, vocab_size, embed_dim, batch_size,
                 seq_length, hidden_units, keep_prob,
                 num_layers, learning_rate, grad_clip, is_training):
        tf.reset_default_graph()  # 模型的训练和预测放在同一个文件下时如果没有这个函数会报错。
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_units = hidden_units
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.is_training = is_training

        if self.is_training:
            self.batch_size = batch_size
            self.seq_length = seq_length
        else:
            self.batch_size = 1
            self.seq_length = 1

        with tf.name_scope('add_input_layer'):
            self.input_layer()
        with tf.name_scope('lstm_cell'):
            self.lstm_cell()
        with tf.name_scope('add_multi_cell'):
            self.add_multi_cell()
        with tf.name_scope('build_output'):
            self.output_layer()
        with tf.name_scope('cost'):
            self.compute_loss()
        with tf.name_scope('optimizer'):
            self.optimizer()

    def input_layer(self):
        self.input_x = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_length), name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_length), name='input_y')

    def word_embed(self, input_data):

        # input_data = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_length), name='input_data')
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
        self.logits = tf.contrib.layers.fully_connected(self.cell_outputs, self.vocab_size, activation_fn=None)
        self.prediction = tf.nn.softmax(self.logits, name='prediction')
        return self.logits, self.prediction

    def compute_loss(self):
        # targets = self.word_embed(self.input_y)
        input_data_shape = tf.shape(self.input_x)
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.input_y, tf.ones([input_data_shape[0], input_data_shape[1]]))
        return self.loss

    def optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        return self.train_op


class conf:
    def __init__(self):
        pass

    batch_size = 100  # Sequences per batch
    num_steps = 100  # Number of sequence steps per batch
    lstm_size = 512  # Size of hidden layers in LSTMs
    num_layers = 2  # Number of LSTM layers
    learning_rate = 0.001  # Learning rate
    keep_prob = 0.5  # Dropout keep probability
    grad_clip = 5
    vocab_size = len(id_to_vocab)

    num_epochs = 10
    # 每n轮进行一次变量保存
    save_every_n = 200

model = language_model(conf.vocab_size, 100, conf.batch_size, conf.num_steps,
                       conf.lstm_size, conf.keep_prob, conf.num_layers,
                       conf.learning_rate, conf.grad_clip, True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    for epoch in range(conf.num_epochs):
        print('epoch', epoch)
        for x, y in get_batch(encoded, conf.batch_size, conf.num_steps):
            print('X.shape', np.shape(x))
            print('Y.shape', np.shape(y))
            start = time.time()
            counter += 1
            # if epoch == 0:
            #     feed_dict = {
            #         model.input_x: x,
            #         model.input_y: y
            #     }
            # else:
            #     feed_dict = {
            #         model.input_x: x,
            #         model.input_y: y,
            #         model.initial_state: new_state
            #     }
            # _, batch_loss, new_state, predict = sess.run([model.train_op,
            #                                               model.loss,
            #                                               model.final_state,
            #                                               model.prediction],
            #                                              feed_dict=feed_dict)
            # end = time.time()
            # if counter % 100 == 0:
            #     print(u'轮数: {}/{}... '.format(epoch + 1, conf.num_epochs),
            #           u'训练步数: {}... '.format(counter),
            #           u'训练误差: {:.4f}... '.format(batch_loss),
            #           u'{:.4f} sec/batch'.format((end - start)))
            state = sess.run(model.initial_state)
            feed = {
                model.input_x: x,
                model.input_y: y,
                model.initial_state: state
            }
            train_loss, state, _ = sess.run([model.loss, model.final_state, model.train_op], feed_dict=feed)
            end = time.time()
            if counter % 100 == 0:
                print(u'轮数: {}/{}... '.format(epoch + 1, conf.num_epochs),
                      u'训练步数: {}... '.format(counter),
                      u'训练误差: {:.4f}... '.format(train_loss),
                      u'{:.4f} sec/batch'.format((end - start)))