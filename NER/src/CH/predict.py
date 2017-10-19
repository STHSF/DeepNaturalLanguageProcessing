#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
from config import Config
from NER_Model import bi_lstm_crf


# 数据导入
with open('data.pkl', 'rb') as pk:
    X = pickle.load(pk)
    y = pickle.load(pk)
    word2id = pickle.load(pk)
    id2word = pickle.load(pk)
    tag2id = pickle.load(pk)
    id2tag = pickle.load(pk)


def file_content_iterator(file_name):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            yield line.strip()


def write_result_to_file(iterator, tags):
    raw_content = next(iterator)
    words = raw_content.split()
    assert len(words) == len(tags)
    for w, t in zip(words, tags):
        print w, '(' + t + ')',
    print
    print '*' * 100


model = bi_lstm_crf(Config)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver()

model_path = '../data/model'
# ckpt = tf.train.get_checkpoint_state(model_path)
best_model_path = tf.train.latest_checkpoint(model_path)
if best_model_path is not None:
    print 'loading pre-trained model from %s.....' % best_model_path
    saver.restore(sess, best_model_path)
else:
    print 'Model not found, please train your model first'


# 获取原文本的iterator
pred_file_path = './data/predict.txt'
file_iter = file_content_iterator(pred_file_path)
while True:
    try:
        for i in file_iter.next().split():
            print(i)
            print(word2id[i])
    except KeyError:
        print"eddro"
        break
# file_iter_ids = tag2id(file_iter.next())
# print(file_iter_ids)

# while True:
#     # batch等于1的时候本来就没有padding，如果批量预测的话，记得这里需要做长度的截取。
#     try:
#         fetches = [model.logits, model.transition_params]
#         feed_dict = {model.source_input: X_batch,
#                      model.is_training: False,
#                      model.lr: 1.0,
#                      model.batch_size: 1,
#                      model.keep_prob: 0.5}
#
#         tf_unary_scores, tf_transition_params = sess.run(
#             [model.logits, model.transition_params])
#     except tf.errors.OutOfRangeError:
#         print 'Prediction finished!'
#         break
#
#     # 把batch那个维度去掉
#     tf_unary_scores = np.squeeze(tf_unary_scores)
#
#     viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
#         tf_unary_scores, tf_transition_params)
#     tags = []
#     for id in viterbi_sequence:
#         tags.append(sess.run(id2tag[tf.constant(id, dtype=tf.int64)]))
#     write_result_to_file(file_iter, tags)