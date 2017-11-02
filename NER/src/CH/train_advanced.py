# coding=utf-8
"""
Training model
"""

import time
import pickle
import numpy as np
import tensorflow as tf
import config
from NER_Model_advanced import bi_lstm_crf
from sklearn.model_selection import train_test_split
from batch_generate import BatchGenerator

# 数据导入
with open('data.pkl', 'rb') as pk:
    X = pickle.load(pk)
    y = pickle.load(pk)
    word2id = pickle.load(pk)
    id2word = pickle.load(pk)
    tag2id = pickle.load(pk)
    id2tag = pickle.load(pk)

# 划分训练集、测试集、和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

print('Creating the data generator ...')
data_train = BatchGenerator(X_train, y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
print('Finished creating the data generator.')

decay = 0.85
tr_batch_size = 128
max_epoch = 1
max_max_epoch = 1
display_num = 5  # 每个 epoch 显示是个结果
# model_save_path = 'ckpt/bi-lstm-crf.ckpt'  # 模型保存位置
print('data_train.y.shape[0]', data_train.y.shape[0])
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次

print('tr_batch_num', tr_batch_num)
print('display_batch', display_batch)


def acc_crf(unary_scores, y_batch, transition_params):
    """
    Compute the accuracy of the crf predicted and the target
    :param unary_scores: (batch_size, num_steps, num_classes)
    :param y_batch:
    :param transition_params:
    :return:
    """
    correct_labels = 0  # prediction accuracy
    total_labels = 0
    # accuracy = 0.0
    # shape = (batch_size, num_steps, num_classes)
    # unary_scores = np.reshape(logits, [batch_size, -1, num_classes])
    # iterate over batches [batch_size, num_steps, target_num], [batch_size, target_num]
    for unary_score_, y_ in zip(unary_scores, y_batch):  # unary_score_  :[num_steps, target_num], y_: [num_steps]
        viterbi_prediction = tf.contrib.crf.viterbi_decode(unary_score_, transition_params)
        # viterbi_prediction: tuple (list[id], value)
        # y_: tuple
        correct_labels += np.sum(
            np.equal(viterbi_prediction[0], y_))  # compare prediction sequence with golden sequence
        total_labels += len(y_)
        # print ("viterbi_prediction")
        # print (viterbi_prediction)
    accuracy_crf = 100.0 * correct_labels / float(total_labels)

    return accuracy_crf


class configuration(object):
    # hyper-parameter
    init_scale = 0.04
    batch_size = 128  # size of per batch

    num_steps = config.FLAGS.num_steps
    vocab_size = config.FLAGS.vocab_size
    embedding_size = config.FLAGS.embedding_size
    hidden_units = config.FLAGS.hidden_units
    layers_num = config.FLAGS.layers_num
    num_classes = config.FLAGS.num_classes
    max_grad_norm = config.FLAGS.max_grad_norm
    model_save_path = config.FLAGS.model_save_path
    lr = config.FLAGS.lr
    keep_pro = config.FLAGS.dropout


# # hyper-parameter
# num_steps = config.FLAGS.num_steps
# vocab_size = config.FLAGS.vocab_size
# embedding_size = config.FLAGS.embedding_size
# hidden_units = config.FLAGS.hidden_units
# layers_num = config.FLAGS.layers_num
# num_classes = config.FLAGS.num_classes
# max_grad_norm = config.FLAGS.max_grad_norm
# model_save_path = config.FLAGS.model_save_path
#
# # DNN Model
# model = bi_lstm_crf(num_steps, vocab_size, embedding_size, hidden_units, layers_num, num_classes)


def run_epoch(sess, dataset, batch_size, model, train_op):
    """Runs the model on the given data."""
    start_time = time.time()
    _batch_size = batch_size

    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    # batch_num = int(((data_size // _batch_size) - 1) // model.num_steps)
    _costs = 0.0
    _accs = 0.0
    for batch in range(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)

        if model.is_training:
            # fetches = [model.accuracy, model.logits, model.transition_params, model.cost]
            fetches = [model.logits, model.transition_params, model.cost, train_op]
            feed_dict = {model.source_input: X_batch,
                         model.target_input: y_batch,
                         model.batch_size: _batch_size}

            _logits, _transition_params, _cost, _ = sess.run(fetches, feed_dict)

        else:

            fetches = [model.logits, model.transition_params, model.cost, tf.no_op()]
            feed_dict = {model.source_input: X_batch,
                         model.target_input: y_batch,
                         model.batch_size: _batch_size}
            _logits, _transition_params, _cost, _ = sess.run(fetches, feed_dict)

        accuracy = acc_crf(_logits, y_batch, _transition_params)
        # _accs += _acc
        _accs += accuracy
        _costs += _cost

        if batch % (batch_num // 10) == 10:
            print("Accuracy: %.2f%%" % (_accs / batch_num))

        if model.is_training and batch % (batch_num // 10) == 10:  # 每 3 个 epoch 保存一次模型
            save_path = model.saver.save(sess, config.FLAGS.model_save_path, global_step=(batch + 1))
            print("Model Saved... at time step " + str(batch))
            print('the save path is ', save_path)

    mean_acc = _accs / batch_num
    mean_cost = np.exp(_costs / batch_num)

    if model.is_training:
        print('\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
        print('Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch'
              % (data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time))

    return mean_cost


train_config = configuration()
eval_config = configuration()
eval_config.batch_size = 1
# eval_config.num_steps = 1

# 设置cpu按需增长
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
init = tf.global_variables_initializer()
# all_vars = tf.trainable_variables()
# saver = tf.traylin.Saver(all_vars)  # 最多保存的模型数量

merged = tf.summary.merge_all()


with tf.Session(config=conf) as sess:
    initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                                train_config.init_scale)
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_model = bi_lstm_crf(is_training=True, config=train_config)
        tf.summary.scalar("Training Loss", train_model.cost)
        tf.summary.scalar("Learning Rate", train_model.lr)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_model = bi_lstm_crf(is_training=False, config=train_config)
        tf.summary.scalar("Validation Loss", valid_model.cost)

    with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            test_model = bi_lstm_crf(is_training=False, config=eval_config)

    # CheckPoint State
    ckpt = tf.train.get_checkpoint_state(train_config.model_save_path)
    if ckpt:
        print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
        train_model.saver.restore(sess, tf.train.latest_checkpoint(train_config.model_save_path))
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('data/model/tensorflowlogs', sess.graph)
    #
    # for epoch in range(max_max_epoch):
    #     lr_decay = 1e-4
    #     if epoch > max_epoch:
    #         lr_decay = lr_decay * (decay ** (epoch - max_epoch))
    #         train_model.assign_lr(sess, train_config.lr * lr_decay)
    #     print('EPOCH %d， lr=%g' % (epoch + 1, lr_decay))
    #
    #     train_perplexity = run_epoch(sess, data_train, train_config.batch_size, train_model, train_model.train_op)
    #     print("Epoch: %d Train Perplexity: %.3f" % (epoch + 1, train_perplexity))
    #     valid_perplexity = run_epoch(sess, data_valid, 500, valid_model, tf.no_op())
    #     print("Epoch: %d Valid Perplexity: %.3f" % (epoch + 1, valid_perplexity))
    #
    # test_perplexity = run_epoch(sess, data_test, 500, test_model, tf.no_op())
    # print("Test Perplexity: %.3f" % test_perplexity)

