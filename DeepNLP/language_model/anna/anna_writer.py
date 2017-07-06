#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: anna_writer.py
@time: 2017/7/4 下午2:27
"""
import time
import numpy as np
import tensorflow as tf

file_path = './data/anna.txt'
with open(file_path) as f:
    text = f.read()
# print('text', text)

vocab = set(text)
# print('vocab\n', vocab)
print('len_vocab', len(vocab))

vocab_to_int = {char: i for i, char in enumerate(vocab)}
# print('vocab_to_int\n', vocab_to_int)
int_to_vocab = dict(enumerate(vocab))
# print('int_to_vocab\n', int_to_vocab)

encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
# print('encoded\n', encoded)
# print('encoded_shape\n', np.shape(encoded))


def get_batch(array, batch_size, seq_length):
    num_steps = batch_size * seq_length
    n_batch = int(len(array) / num_steps)

    array = array[:n_batch * num_steps]
    array = array.reshape((batch_size, -1))

    for n in range(0, array.shape[1], seq_length):
        x = array[:, n:(n + seq_length)]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]

        yield x, y


class language_model:
    def __init__(self, num_classes, batch_size=100, seq_length=50, learning_rate=0.01, num_layers=5, hidden_units=128,
                 keep_prob=0.8, grad_clip=5, is_training=True):

        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        self.num_classes = num_classes

        if self.is_training:
            self.batch_size = batch_size
            self.seq_length = seq_length
        else:
            self.batch_size = 1
            self.seq_length = 1

        with tf.name_scope('add_input_layer'):
            self.add_input_layer()
        with tf.variable_scope('lstm_cell'):
            self.add_lstm_cell()
        with tf.name_scope('build_output'):
            self.build_output()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('optimizer'):
            self.optimizer()

    def add_input_layer(self):
        with tf.name_scope("inputs"):
            self.x = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_length), name='inputs')
            self.y = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_length), name='targets')

        self.inputs = tf.one_hot(self.x, self.num_classes)
        # self.inputs = tf.reshape(self.y, [-1, self.num_classes])

        self.targets = tf.one_hot(self.y, self.num_classes)

    def rnn_cell(self):
        # Or GRUCell, LSTMCell(args.hiddenSize)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units,
                                                 state_is_tuple=True)
        if not self.is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                      input_keep_prob=1.0,
                                                      output_keep_prob=self.keep_prob)
        return lstm_cell

    def add_lstm_cell(self):
        lstm_cells = tf.contrib.rnn.MultiRNNCell([self.rnn_cell() for _ in range(self.num_layers)],
                                                 state_is_tuple=True)

        with tf.name_scope('initial_state'):
            self.initial_state = lstm_cells.zero_state(self.batch_size, dtype=tf.float32)

        self.cell_outputs, self.final_state = tf.nn.dynamic_rnn(cell=lstm_cells,
                                                                inputs=self.inputs,
                                                                initial_state=self.initial_state)

    def build_output(self):
        seq_output = tf.concat(self.cell_outputs, axis=1)
        x = tf.reshape(seq_output, [-1, self.hidden_units])

        with tf.name_scope('softmax'):
            sofmax_w = tf.Variable(tf.truncated_normal([self.hidden_units, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))

        with tf.name_scope('wx_plus_b'):
            self.logits = tf.matmul(x, sofmax_w) + softmax_b

        self.prediction = tf.nn.softmax(logits=self.logits, name='prediction')

        return self.prediction, self.logits

    def compute_cost(self):
        # One-hot编码
        y_reshaped = tf.reshape(self.targets, self.logits.get_shape())

        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)
        return self.loss

    def optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

        return self.optimizer


class conf:
    batch_size = 100  # Sequences per batch
    num_steps = 100  # Number of sequence steps per batch
    lstm_size = 512  # Size of hidden layers in LSTMs
    num_layers = 2  # Number of LSTM layers
    learning_rate = 0.001  # Learning rate
    keep_prob = 0.5  # Dropout keep probability
    grad_clip = 5
    num_classes = len(vocab)

    epochs = 20
    # 每n轮进行一次变量保存
    save_every_n = 200


def train(language_model):
    language_model = language_model(conf.num_classes, conf.batch_size, conf.num_steps, conf.learning_rate,
                                    conf.num_layers, conf.lstm_size, conf.keep_prob, conf.grad_clip, is_training=True)
    saver = tf.train.Saver(max_to_keep=100)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run(tf.global_variables_initializer())

    counter = 0
    for e in range(conf.epochs):

        for x, y in get_batch(encoded, conf.batch_size, conf.num_steps):
            counter += 1
            start = time.time()
            if e == 0:
                feed_dict = {
                    language_model.x: x,
                    language_model.y: y,
                }
            else:
                feed_dict = {
                    language_model.x: x,
                    language_model.y: y,
                    language_model.initial_state: new_state
                }

            #
            _, batch_loss, new_state, predict = sess.run(
                [language_model.optimizer, language_model.loss, language_model.final_state, language_model.prediction],
                feed_dict=feed_dict)

            end = time.time()
            # control the print lines
            if counter % 100 == 0:
                print('轮数: {}/{}... '.format(e + 1, conf.epochs),
                      '训练步数: {}... '.format(counter),
                      '训练误差: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end - start)))

            if counter % conf.save_every_n == 0:
                saver.save(sess, 'checkpoints/i{}_l{}.ckpt'.format(counter, conf.lstm_size))
    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, conf.lstm_size))


def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符

    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def generate_samples(language_model, checkpoint, num_samples, prime='The '):

    samples = [char for char in prime]
    language_model = language_model(conf.num_classes,conf.batch_size, conf.num_steps, conf.learning_rate, conf.num_layers,
                                    conf.lstm_size, conf.keep_prob, conf.grad_clip, False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(language_model.initial_state)

        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed_dict = {language_model.x: x,
                         language_model.initial_state: new_state}

            predicts, final_state = sess.run([language_model.prediction,
                                              language_model.final_state],
                                             feed_dict=feed_dict)

        c = pick_top_n(predicts, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(num_samples):
            x[0, 0] = c
            feed_dict = {language_model.x: x,
                         language_model.initial_state: new_state}
            preds, new_state = sess.run([language_model.prediction,
                                         language_model.final_state],
                                        feed_dict=feed_dict)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
    return ''.join(samples)


tf.train.latest_checkpoint('checkpoints')

# 选用最终的训练参数作为输入进行文本生成
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = generate_samples(language_model, checkpoint, 20000, prime="The ")
print(samp)
