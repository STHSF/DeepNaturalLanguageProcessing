#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: model_predict.py
@time: 2020/1/8 4:19 下午
"""
from __future__ import print_function

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import os
import argparse
import tensorflow as tf
import tensorflow.contrib.keras as kr

from rnn_model import TRNNConfig, TextRNN
from bi_rnn_model import TBRNNConfig, TextBiRNN
from cnnews_loder import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str


class RNNModel:
    def __init__(self, conf, model, categories, cat_to_id, words, word_to_id):
        self.config = conf
        self.model = model
        self.categories = categories
        self.cat_to_id = cat_to_id
        self.word_to_id = word_to_id

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def init_session(self):
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.dropout_keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        self.session.close()
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', default="RNN", type=str, required=True, choices=['RNN', 'BiRNN'],
                        help="选择的模型类型")
    pass_args = parser.parse_args()
    _model = pass_args.model

    base_dir = '../data/cnews'
    vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
    if _model == "RNN":
        save_dir = './checkpoints/textrnn'
        args = TRNNConfig()
        model = TextRNN(args)
    if _model == "BiRNN":
        save_dir = './checkpoints/textbirnn'
        args = TBRNNConfig()
        model = TextBiRNN(args)
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)

    rnn_model = RNNModel(args, model, categories, cat_to_id, words, word_to_id)
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    for i in test_demo:
        print(rnn_model.predict(i))