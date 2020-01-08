#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: model_running.py
@time: 2018/3/27 下午5:10
"""
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import os
import time
from datetime import timedelta
import tensorflow as tf
import numpy as np
from rnn_model import TRNNConfig, TextRNN
from cnnews_loder import read_vocab, read_category, batch_iter, process_file, build_vocab


base_dir = "../data/cnews"
train_dir = os.path.join(base_dir, "cnews.train.txt")
test_dir = os.path.join(base_dir, "cnews.test.txt")
val_dir = os.path.join(base_dir, "cnews.val.txt")
vocab_dir = os.path.join(base_dir, "cnews.vocab.txt")

save_dir = 'checkpoints/textrnn'
save_path = os.path.join(save_dir, 'best_validation')


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
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """
    develop model
    :param sess:
    :param x_:
    :param y_:
    :return:
    """
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss*batch_len
        total_acc += acc*batch_len

    return total_acc / data_len, total_loss / data_len


def train():
    """
    model training
    :return:
    """
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = '../tensorboard/textrnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # tf.summary.scalar("loss", model.loss)
    # tf.summary.scalar("accuracy", model.acc)

    # mergerd_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter(tensorboard_dir)

    # saver = tf.train.Saver()
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
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

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # writer.add_graph(session.graph)

    print("Training and evaluating...")
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch +1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)

                # print("prediction %s" % session.run(tf.cast(tf.arg_max(model.y_pred_cls, 1), tf.int32)))
                # print("y_batch %s" % session.run(tf.reshape(model.input_y, [-1])))

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    # saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1
            print('y_pre', session.run(model.y_pred_cls, feed_dict=feed_dict))
            print('input_y', session.run(tf.arg_max(model.input_y, 1), feed_dict=feed_dict))

            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time ,auto-stoppping...")
                flag = True
                break

            if flag:
                break


if __name__ == '__main__':
    print("Configuring RNN Model")
    config = TRNNConfig()
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextRNN(config)

    train()