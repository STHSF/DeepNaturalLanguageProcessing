#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: 1.0
@author: Li Yu
@license: Apache Licence 
@file: anna_writer.py
@time: 2017/7/4 下午2:27
"""
import time
import numpy as np
import tensorflow as tf

# 读取训练数据
file_path = './data/anna.txt'
with open(file_path) as f:
    text = f.read()
# print('text', text)
# 生成字符集合
# 使用set对列表去重，并保持列表原来的顺序
vocab = list(set(text))
vocab.sort(key=text.index)
# print('vocab\n', vocab)
print('len_vocab', len(vocab))

# 字符编码

vocab_to_int = {char: i for i, char in enumerate(vocab)}
# print('vocab_to_int\n', vocab_to_int)
int_to_vocab = dict(enumerate(vocab))
# print('int_to_vocab\n', int_to_vocab)

# 文本编码
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
# print('encoded\n', encoded)
# print('encoded_shape\n', np.shape(encoded))


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
    def __init__(self, num_classes, batch_size=100, seq_length=50, learning_rate=0.01, num_layers=5, hidden_units=128,
                 keep_prob=0.8, grad_clip=5, is_training=True):
        tf.reset_default_graph()  # 模型的训练和预测放在同一个文件下时如果没有这个函数会报错。
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
            self.add_multi_cells()
        with tf.name_scope('build_output'):
            self.build_output()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('optimizer'):
            self.optimizer()

    def add_input_layer(self):
        self.x = tf.placeholder(tf.int32,
                                shape=(self.batch_size, self.seq_length), name='inputs')    # [batch_size, seq_length]
        self.y = tf.placeholder(tf.int32,
                                shape=(self.batch_size, self.seq_length), name='targets')   # [batch_size, seq_length]
        # One-hot编码
        self.inputs = tf.one_hot(self.x, self.num_classes)           # [batch_size, seq_length, num_classes]
        # self.inputs = tf.reshape(self.y, [-1, self.num_classes])
        self.targets = tf.one_hot(self.y, self.num_classes)          # [batch_size, seq_length, num_classes]

    def lstm_cell(self):
        # Or GRUCell, LSTMCell(args.hiddenSize)

        with tf.variable_scope('lstm_cell'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units,
                                                state_is_tuple=True)

        # with tf.variable_scope('lstm_cell'):
        #     cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units,
        #                                         state_is_tuple=True,
        #                                         reuse=tf.get_variable_scope().reuse)
        if self.is_training:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 input_keep_prob=1.0,
                                                 output_keep_prob=self.keep_prob)
        return cell

    def add_multi_cells(self):

        # initial_state: [batch_size, hidden_units * num_layers]
        # cell_output: [batch_size, seq_length, hidden_units]
        # final_state: [batch_size, hidden_units * num_layers]
        stacked_cells = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)],
                                                    state_is_tuple=True)

        with tf.name_scope('initial_state'):
            self.initial_state = stacked_cells.zero_state(self.batch_size, dtype=tf.float32)

        self.cell_outputs, self.final_state = tf.nn.dynamic_rnn(cell=stacked_cells,
                                                                inputs=self.inputs,
                                                                initial_state=self.initial_state)

    def build_output(self):
        seq_output = tf.concat(self.cell_outputs, axis=1)
        y0 = tf.reshape(seq_output, [-1, self.hidden_units])    # y0: [batch_size * seq_length, hidden_units]

        with tf.name_scope('weights'):
            softmax_w = tf.Variable(tf.truncated_normal([self.hidden_units, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))

        with tf.name_scope('wx_plus_b'):
            self.logits = tf.matmul(y0, softmax_w) + softmax_b    # logits: [batch_size * seq_length, num_classes]

        self.prediction = tf.nn.softmax(logits=self.logits, name='prediction')

        return self.prediction, self.logits

    def compute_cost(self):
        y_reshaped = tf.reshape(self.targets, self.logits.get_shape())    # y_reshaped: [batch_size * seq_length, num_classes]

        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)  # loss: [batch_size, seq_length]

        self.loss = tf.reduce_mean(loss)
        return self.loss

    def optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        return self.train_op


class conf:
    batch_size = 100  # Sequences per batch
    num_steps = 100  # Number of sequence steps per batch
    lstm_size = 512  # Size of hidden layers in LSTMs
    num_layers = 2  # Number of LSTM layers
    learning_rate = 0.001  # Learning rate
    keep_prob = 0.5  # Dropout keep probability
    grad_clip = 5
    num_classes = len(vocab)

    num_epochs = 1
    # 每n轮进行一次变量保存
    save_every_n = 200


def train():
    """
    语言模型的训练
    Returns:

    """
    model = language_model(conf.num_classes, conf.batch_size, conf.num_steps, conf.learning_rate, conf.num_layers,
                           conf.lstm_size, conf.keep_prob, conf.grad_clip, is_training=True)
    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0
        for epoch in range(conf.num_epochs):
            for x, y in get_batch(encoded, conf.batch_size, conf.num_steps):
                counter += 1
                start = time.time()
                if epoch == 0:
                    feed_dict = {
                        model.x: x,
                        model.y: y
                    }
                else:
                    feed_dict = {
                        model.x: x,
                        model.y: y,
                        model.initial_state: new_state
                    }

                _, batch_loss, new_state, predict = sess.run([model.train_op,
                                                              model.loss,
                                                              model.final_state,
                                                              model.prediction],
                                                             feed_dict=feed_dict)
                end = time.time()
                # control the print lines
                if counter % 100 == 0:
                    print(u'轮数: {}/{}... '.decode('utf-8').format(epoch + 1, conf.num_epochs),
                          u'训练步数: {}... '.decode('utf-8').format(counter),
                          u'训练误差: {:.4f}... '.decode('utf-8').format(batch_loss),
                          u'{:.4f} sec/batch'.decode('utf-8').format((end - start)))

                if counter % conf.save_every_n == 0:
                    saver.save(sess, u'checkpointss/i{}_l{}.ckpt'.decode('utf-8').format(counter, conf.lstm_size))
        saver.save(sess, u"checkpointss/i{}_l{}.ckpt".decode('utf-8').format(counter, conf.lstm_size))


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


def generate_samples(checkpoint, num_samples, prime='The '):

    samples = [char for char in prime]
    model = language_model(conf.num_classes, conf.batch_size, conf.num_steps, conf.learning_rate, conf.num_layers,
                           conf.lstm_size, conf.keep_prob, conf.grad_clip, is_training=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed_dict = {model.x: x}

            predicts = sess.run(model.prediction, feed_dict=feed_dict)

            c = pick_top_n(predicts, len(vocab))
            samples.append(int_to_vocab[c])

        for i in range(num_samples):
            x[0, 0] = c
            feed_dict = {model.x: x,
                         model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction,
                                         model.final_state],
                                        feed_dict=feed_dict)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
    return ''.join(samples)


if __name__ == '__main__':
    train()

    tf.train.latest_checkpoint('checkpointss')

    # 选用最终的训练参数作为输入进行文本生成
    checkpoint = tf.train.latest_checkpoint('checkpointss')
    samp = generate_samples(checkpoint, 20000, prime="The ")
    print(samp)



#  问题1 其中还存在的问题，程序每运行一次vocab_to_int都会改变，导致train和predict不能分开。
