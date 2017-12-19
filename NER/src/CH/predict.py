#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import config
from NER_Model_advanced import bi_lstm_crf


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


# hyper-parameter
num_steps = config.FLAGS.num_steps
vocab_size = config.FLAGS.vocab_size
embedding_size = config.FLAGS.embedding_size
hidden_units = config.FLAGS.hidden_units
layers_num = config.FLAGS.layers_num
num_classes = config.FLAGS.num_classes
max_grad_norm = config.FLAGS.max_grad_norm
model_save_path = config.FLAGS.model_save_path

# Load DNN Model
model = bi_lstm_crf(num_steps, vocab_size, embedding_size, hidden_units, layers_num, num_classes)
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
saver = tf.train.Saver()

# ckpt = tf.train.get_checkpoint_state(model_path)
best_model_path = tf.train.latest_checkpoint(model_save_path)
if best_model_path is not None:
    print 'loading pre-trained model from %s.....' % best_model_path
    saver.restore(sess, best_model_path)
else:
    print 'Model not found, please train your model first'


def viterbi_decoder(x_batch):
    fetches = [model.logits, model.transition_params]
    feed_dict = {model.source_input: x_batch,
                 model._is_training: False,
                 model._batch_size: 1}

    tf_unary_scores, tf_transition_params = sess.run(fetches, feed_dict)

    tf_unary_scores = np.squeeze(tf_unary_scores)

    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores, tf_transition_params)

    tags = []

    for ids in viterbi_sequence:
        tags.append(sess.run(id2tag[tf.constant(ids, dtype=tf.int64)]))

    return tags


# 获取predict 文本
pred_file_path = './data/predict.txt'
file_iter = file_content_iterator(pred_file_path)


def text2ids(text, max_len):
    """把字片段text转为 ids."""
    words = list(text)
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        print u'输出片段超过%d部分无法处理' % max_len
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids)))  # 短则补全
    ids = np.asarray(ids).reshape([-1, max_len])
    return ids


try:
    for i in file_iter:
        print(i)
        # shape = (1, 100)
        text_ids = text2ids(i.split(), 100)
        print(np.shape(text_ids))
        print(text_ids)
        # tags = viterbi(text_ids)
        #
        # write_result_to_file(file_iter, tags)
except KeyError:
    print("eddro")

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



