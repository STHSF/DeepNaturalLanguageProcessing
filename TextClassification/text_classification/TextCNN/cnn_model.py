#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: cnn_model.py
@time: 2020/1/8 2:51 下午
"""

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    kernel_size_list = [3, 4, 5]
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        self._build_graph()

    def _build_graph(self):
        """CNN模型"""
        with tf.variable_scope("Input_data"):
            # 三个待输入的数据
            self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # 词向量映射
        with tf.device('/cpu:0'), tf.name_scope('Embedding'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # with tf.name_scope("CNN"):
        #     # CNN layer
        #     conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
        #     # global max pooling layer
        #     gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.kernel_size_list):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, filter_size, name='conv',
                                            reuse=True)
                    mp = tf.reduce_max(conv, reduction_indices=[1], name='mp')
                    pooled_outputs.append(mp)
            num_filter_total = self.config.num_filters * len(self.config.kernel_size_list)
            self.h_pool = tf.concat(pooled_outputs, 3)
            gmp = tf.reshape(self.h_pool, [-1, num_filter_total])

        with tf.name_scope("Score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("Optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("Accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
