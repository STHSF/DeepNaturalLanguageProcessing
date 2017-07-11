#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time

import input_data

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006


flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


# 数据格式转换
def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class Config(object):
    """ config."""
    input_size = 100
    n_steps = 20

    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    cell_size = 10
    output_size = 2    # num_classes
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class SentimentModel(object):
    """The sentiment model."""

    def __init__(self, is_training, config):
        self.n_steps = n_steps = config.n_steps
        self.input_size = input_size = config.input_size
        self.output_size = output_size = config.output_size
        self.cell_size = config.cell_size
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps

        with tf.name_scope('inputs'):
            self._input_data = tf.placeholder(tf.float32, [None, n_steps, input_size], name='_input_data')
            self._targets = tf.placeholder(tf.float32, [None, n_steps, output_size], name='_targets')

        with tf.variable_scope('in_hidden'):
            self.add_input_layer()

        with tf.variable_scope('LSTM_cell'):
            self.add_cell(is_training, config)

        with tf.variable_scope('out_hidden'):
            self.add_output_layer()

        with tf.name_scope('cost'):
            self.compute_cost_regression()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self):
        data_out = tf.reshape(self._input_data,
                              [-1, self.input_size],
                              name='2_2D')  # (batch*n_step, in_size)

        weights_in = self._weight_variable([self.input_size, self.cell_size])

        bias_in = self._bias_variable([self.cell_size,])

        with tf.name_scope('w_plus_b'):
            data_out = tf.matmul(data_out, weights_in) + bias_in

        self.output_data = tf.reshape(data_out,
                                      [-1, self.n_steps, self.cell_size],
                                      name='2_3D')

    def add_cell(self, is_training, config):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)

        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                      output_keep_prob=config.keep_prob)
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell,
                                                                     self.output_data,
                                                                     initial_state=self.cell_init_state,
                                                                     time_major=False)

    def add_output_layer(self):

        # shape = (batch * steps, cell_size)
        outputs = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')

        weights_out = self._weight_variable([self.cell_size, self.output_size])

        bias_out = self._bias_variable([self.output_size, ])

        # shape = (batch * steps, output_size)
        with tf.name_scope('w_plus_b'):

            # hidden layer for output as the final results
            #############################################
            # self.predict = tf.matmul(outputs, weights_out) + bias_out

            # # or
            # unpack to list [(batch, outputs)..] * steps
            outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
            self.predict = tf.matmul(outputs[-1], weights_out) + bias_out

    def compute_cost_regression(self):

        losses = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(self.predict, [-1], name='reshape_pred')],
            [tf.reshape(self._targets, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )

        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.scalar_summary('cost', self.cost)

    def compute_cost_classify(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict, self._targets), name='')

    def ms_error(self, y_pre, y_target):

        return tf.square(tf.sub(y_pre, y_target))

    def _weight_variable(self, shape, name='weights'):

        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)

        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):

        initializer = tf.constant_initializer(0.1)

        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self.targets

    @property
    def initial_state(self):
        return self.cell_init_state

    @property
    def final_state(self):
        return self.cell_final_state


class SentimentInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


def run_epoch(session, model, data, eval_op, verbose=False):
    """Runs the model on the given data."""

    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                      model.num_steps)):
        fetches = [model.cost, model.final_state, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, state, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "train":
        return Config()
    elif FLAGS.model == "test":
        return Config()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    # 读取ptb原始数据
    # raw_data = reader.ptb_raw_data(FLAGS.data_path)
    # train_data, valid_data, test_data= raw_data

    raw_data = input_data.read_data_sets()
    train_data = raw_data.train
    valid_data = raw_data.validation
    test_data = raw_data.test

    # 获取参数
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    # 构建graph框架， 使用tensorbord
    with tf.Graph().as_default():
        # 使用均匀分布初始化
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        # 定义Train域
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = SentimentModel(is_training=True, config=config)
            tf.scalar_summary("Training Loss", m.cost)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = SentimentModel(is_training=False, config=config)
            tf.scalar_summary("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = SentimentModel(is_training=False, config=eval_config)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_data, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__ == '__main__':
    tf.app.run()  # It's just a very quick wrapper that handles flag parsing and then dispatches to your own main

