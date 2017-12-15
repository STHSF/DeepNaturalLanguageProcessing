#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: 1.0
@author: Li Yu
@license: Apache Licence 
@file: data_processing.py
@time: 2017/7/3 上午9:59
@version: python3 tensorflow1.0
"""
import time
from collections import namedtuple
import numpy as np
import tensorflow as tf

file_path = './data/anna.txt'
with open(file_path, 'r') as f:
    text = f.read()
# 字符集合
vocab = set(text)
# 字符数字映射
vocab_to_int = {char: i for i, char in enumerate(vocab)}
# 数字字符映射
int_to_vocab = dict(enumerate(vocab))

# 对文本进行编码
encode = np.array([vocab_to_int[char] for char in text], dtype=np.int32)
print(text[:100])
print('encode\n', encode[:100])
print('encode.shape\n', np.shape(encode))


def generate_bath(arr, batch_size, seq_length):
    """

    Args:
        arr: input array
        batch_size:
        seq_length:

    Returns:

    """
    num_steps = batch_size * seq_length
    # print('num_steps', num_steps)
    n_iters = int(len(arr) / num_steps)
    # print('num_iters', n_iters)
    arr = arr[: num_steps * n_iters]
    # print('arr_b', arr[:,:15])
    # 重塑
    arr = arr.reshape((batch_size, -1))
    print('arr_a\n',arr[:,:15])
    # print(arr.shape[1])

    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n + seq_length]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

batches = generate_bath(encode, 10, 50)
x, y = next(batches)
print('inputs.shape', np.shape(x))
print('inputs', x[:10, :10])
print('y.shape', np.shape(y))
print('y', y[:10, :10])


class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        tf.reset_default_graph()

        # 如果sampling是True，则采用SGD
        if sampling:
            self.batch_size, self.num_steps = 1, 1
        else:
            self.batch_size, self.num_steps = batch_size, num_steps
        self.num_classes = num_classes
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

    def build_inputs(self, batch_size, seq_length):
        """
        构建输入层
        batch_size: 每个batch中的序列个数
        seq_length: 每个序列包含的字符数
        """
        self.inputs = tf.placeholder(tf.int32, shape=(batch_size, seq_length), name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=(batch_size, seq_length), name='targets')
        # 加入keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def build_lstm(self):
        """
        构建lstm层
        keep_prob
        lstm_size: lstm隐层中结点数目
        num_layers: lstm的隐层数目
        batch_size: batch_size
        """
        # 构建一个基本lstm单元
        cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

        # 添加dropout
        self.cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 堆叠
        # cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(num_layers)])
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, self.num_classes)
        print('x_one_hot.shape', np.shape(x_one_hot))
        # 运行RNN
        self.outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

    def build_output(self):
        """
        构造输出层
        lstm_output: lstm层的输出结果
        in_size: lstm输出层重塑后的size
        out_size: softmax层的size
        """
        # 将lstm的输出按照列concate，例如[[1,2,3],[7,8,9]],
        # tf.concat的结果是[1,2,3,7,8,9]
        seq_output = tf.concat(self.outputs, axis=1)  # tf.concat(concat_dim, values)
        # reshape
        x = tf.reshape(seq_output, [-1, self.lstm_size])

        # 将lstm层与softmax层全连接
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))

        # 计算logits
        self.logits = tf.matmul(x, softmax_w) + softmax_b
        # softmax层返回概率分布
        self.prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        """
        根据logits和targets计算损失
        logits: 全连接层的输出结果（不经过softmax）
        targets: targets
        lstm_size
        num_classes: vocab_size

        """
        # One-hot编码
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())

        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        """
        构造Optimizer
        loss: 损失
        learning_rate: 学习率
        """
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))


batch_size = 100         # Sequences per batch
num_steps = 100          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5         # Dropout keep probability

epochs = 1
# 每n轮进行一次变量保存
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in generate_bath(encode, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer],
                                                feed_dict=feed)

            end = time.time()
            # control the print lines
            if counter % 100 == 0:
                print('轮数: {}/{}... '.format(e + 1, epochs),
                      '训练步数: {}... '.format(counter),
                      '训练误差: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end - start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "acheckpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "acheckpoints/i{}_l{}.ckpt".format(counter, lstm_size))


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


def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    """
    生成新文本

    checkpoint: 某一轮迭代的参数文件
    n_sample: 新闻本的字符长度
    lstm_size: 隐层结点数
    vocab_size
    prime: 起始文本
    """
    # 将输入的单词转换为单个字符组成的list
    samples = [c for c in prime]
    # sampling=True意味着batch的size=1 inputs 1
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        # 添加字符到samples中
        samples.append(int_to_vocab[c])

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

    return ''.join(samples)

tf.train.latest_checkpoint('acheckpoints')

# 选用最终的训练参数作为输入进行文本生成
checkpoint = tf.train.latest_checkpoint('acheckpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="The")
print(samp)