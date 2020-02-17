#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: model_run.py
@time: 2018/3/27 下午5:10
"""
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import os
import time
import argparse
from datetime import timedelta
import tensorflow as tf
from sklearn import metrics

import numpy as np
# from rnn_model import TextRNN
from rnn_model_attention import TRNNConfig TextRNN
from bi_rnn_model import TBRNNConfig, TextBiRNN
from cnnews_loder import read_vocab, read_category, batch_iter, process_file, build_vocab

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_time_dif(start_time):
    """
    耗时计算
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    """
    将模型需要feed的参数
    :param x_batch:
    :param y_batch:
    :param keep_prob:
    :return: 返回一个数据字典
    """
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_, batch_size):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, batch_size)
    total_loss = 0.0
    total_acc = 0.0
    y_pred_class = None
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        # 在测试时不用进行dropout
        y_pred_class, loss, acc = sess.run([model.y_pred_cls, model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return y_pred_class, total_loss / data_len, total_acc / data_len


def train():
    """
    model training
    :return:
    """
    print("Configuring TensorBoard and Saver...")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    tf.summary.histogram("embedding_var", model.embedding)
    tf.summary.histogram("alphas", model.alphas)

    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data ...")

    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # print(x_train)
    # print(type(x_train), np.shape(x_train))
    # print(y_train)
    # print(type(y_train), np.shape(y_train))
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        merged_summary = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter(tensorboard_dir + '/train', session.graph)
        writer_valid = tf.summary.FileWriter(tensorboard_dir + '/valid')
        # writer_train.add_graph(session.graph)

        print("Training and evaluating...")
        start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        require_improvement = 1000

        flag = False
        for epoch in range(config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = batch_iter(x_train, y_train, config.batch_size)
            for x_batch, y_batch in batch_train:
                feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
                if total_batch % config.save_per_batch == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    # train
                    summary_str = session.run(merged_summary, feed_dict=feed_dict)
                    writer_train.add_summary(summary_str, total_batch)
                    # valid
                    batch_eval = batch_iter(x_val, y_val, config.batch_size)
                    for _x_batch, _y_batch in batch_eval:
                        feed_dict_valid = feed_data(_x_batch, _y_batch, 1.0)
                    summary_valid = session.run(merged_summary, feed_dict=feed_dict_valid)
                    writer_valid.add_summary(summary_valid, total_batch)

                if total_batch % config.print_per_batch == 0:
                    feed_dict[model.dropout_keep_prob] = 1.0
                    loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                    y_pred_cls_1, loss_val, acc_val = evaluate(session, x_val, y_val, config.batch_size)
                    # print("prediction %s" % session.run(tf.cast(tf.arg_max(model.y_pred_cls, 1), tf.int32)))
                    # print("y_batch %s" % session.run(tf.reshape(model.input_y, [-1])))

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                session.run(model.optim, feed_dict=feed_dict)
                total_batch += 1
                # print('y_pre', session.run(model.y_pred_cls, feed_dict=feed_dict))
                # print('input_y', session.run(tf.arg_max(model.input_y, 1), feed_dict=feed_dict))

                if total_batch - last_improved > require_improvement:
                    print("No optimization for a long time ,auto-stoppping...")
                    flag = True
                    break
                if flag:
                    break
    writer_train.close()
    writer_valid.close()


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    # 读取保存的模型

    print('Testing...')
    y_pred_cls_1, loss_test, acc_test = evaluate(session, x_test, y_test, config.batch_size)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = config.batch_size
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.dropout_keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', default="train", type=str, choices=['train', 'test'], help="类型")
    parser.add_argument('--model', dest='model', default="RNN", type=str, required=True, choices=['RNN', 'BiRNN'], help="模型")
    args = parser.parse_args()
    _type = args.type
    _model = args.model

    base_dir = "../data/cnews"
    train_dir = os.path.join(base_dir, "cnews.train.txt")
    test_dir = os.path.join(base_dir, "cnews.test.txt")
    val_dir = os.path.join(base_dir, "cnews.val.txt")
    vocab_dir = os.path.join(base_dir, "cnews.vocab.txt")

    print("Configuring Model Path")
    model_name = ''
    if _model == 'RNN':
        model_name = "textrnn"
    if _model == 'BiRNN':
        model_name = "textbirnn"
    save_dir = './checkpoints/' + model_name
    save_path = os.path.join(save_dir, 'best_validation')
    tensorboard_dir = './tensorboard/' + model_name

    print("Configuring RNN Model")
    config = {}
    if _model == 'RNN':
        config = TRNNConfig()
    if _model == 'BiRNN':
        config = TBRNNConfig()

    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, config.vocab_size)

    if _model == 'RNN':
        # TextRNN
        model = TextRNN(config)
    if _model == 'BiRNN':
        # TextBiRNN
        model = TextBiRNN(config)

    if _type == 'train':
        train()
    if _type == 'test':
        test()
