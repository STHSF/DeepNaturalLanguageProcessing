# coding=utf-8
"""
Training model
"""

import pickle
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import config
from batch_generate import BatchGenerator
from NER_Model_advanced import bi_lstm_crf

# 数据导入
# data_path = "/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/NER/src/CH/"
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
        # print("y_")
        # print(y_)
    accuracy_crf = 100.0 * correct_labels / float(total_labels)
    return accuracy_crf


class configuration(object):
    # hyper-parameter
    init_scale = 0.04
    batch_size = config.FLAGS.batch_size  # size of per batch
    num_steps = config.FLAGS.max_sequence
    vocab_size = config.FLAGS.vocab_size
    embedding_size = config.FLAGS.embedding_size
    hidden_units = config.FLAGS.hidden_units
    layers_num = config.FLAGS.layers_num
    num_classes = config.FLAGS.num_classes
    max_grad_norm = config.FLAGS.max_grad_norm
    model_save_path = config.FLAGS.model_save_path
    lr = config.FLAGS.lr
    keep_pro = config.FLAGS.dropout


def run_epoch(dataset, session, model, summary_writer):
    start_time = time.time()
    display_batch = 5
    _batch_size = model.batch_size
    print("model_batch_size in run_epoch", _batch_size)
    data_size = dataset.y.shape[0]
    print("data_size %s", data_size)
    batch_num = int(data_size / _batch_size)
    print("batch_num: %d", batch_num)

    if model.is_training:
        print("=========================== Train ===============================")
        fetches = [model.logits, model.transition_params, model.cost, model.train_op]
    else:
        print("======================= Valid or Test ============================")
        fetches = [model.logits, model.transition_params, model.cost, tf.no_op()]

    iters = 0
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0

    for batch in range(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {model.source_input: X_batch,
                     model.target_input: y_batch}
        _logits, _transition_params, _cost, _ = session.run(fetches, feed_dict)

        accuracy = acc_crf(_logits, y_batch, _transition_params)
        _accs += accuracy
        _costs += _cost
        iters += model.num_steps

        if batch % 10 == 0:
            summary = session.run(model.merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, batch)

        if model.is_training:
            show_accs += accuracy
            show_costs += _cost
            if (batch + 1) % display_batch == 0:
                print('batch: %s', batch + 1)
                print('\t training acc: %g, cost: %g, speed: %.0f wps' %
                      (show_accs / display_batch, show_costs / display_batch,
                       (iters * model.batch_size) / (time.time() - start_time)))
                show_accs, show_costs = 0.0, 0.0
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num

    # save model
    if model.is_training:
        pass

    return mean_acc, mean_cost


train_config = configuration()
eval_config = configuration()
eval_config.batch_size = 1
# eval_config.num_steps = 1

decay = 0.85
max_epoch = 3
max_max_epoch = 10

merged = tf.summary.merge_all()


# sv = tf.train.Supervisor(logdir=config.FLAGS.log_path)
# with sv.managed_session() as session:

def main():
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                                    train_config.init_scale)
        with tf.name_scope("Train") as train_scope:
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_model = bi_lstm_crf(train_scope, is_training=True, config=train_config)
            tf.summary.scalar("Training Loss", train_model.cost)
            tf.summary.scalar("Learning Rate", train_model.lr)

        with tf.name_scope("Valid") as valid_scope:
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_model = bi_lstm_crf(valid_scope, is_training=False, config=train_config)
            tf.summary.scalar("Validation Loss", valid_model.cost)

        with tf.name_scope("Test") as test_scope:
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = bi_lstm_crf(test_scope, is_training=False, config=eval_config)

        with tf.Session() as session:
            train_summary_writer = tf.summary.FileWriter('data/model/tensorflowlogs/train', session.graph)
            valid_summary_writer = tf.summary.FileWriter('data/model/tensorflowlogs/test', session.graph)
            test_summary_writer = tf.summary.FileWriter('data/model/tensorflowlogs/test', session.graph)

            # CheckPoint State
            ckpt = tf.train.get_checkpoint_state(train_config.model_save_path)
            if ckpt:
                print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
                train_model.saver.restore(session, tf.train.latest_checkpoint(train_config.model_save_path))
            else:
                print("Created model with fresh parameters.")
                session.run(tf.global_variables_initializer())

            for epoch in range(max_max_epoch):
                lr_decay = 1e-4
                if epoch > max_epoch:
                    lr_decay = lr_decay * (decay ** (epoch - max_epoch))
                    train_model.assign_lr(session, train_config.lr * lr_decay)
                print('EPOCH %d， lr=%g' % (epoch + 1, lr_decay))

                train_perplexity = run_epoch(data_train, session, train_model, train_summary_writer)
                print("Epoch: %d Train acc: %.3f, Train cost: %.3f " % (
                    epoch + 1, train_perplexity[0], train_perplexity[1]))
                valid_perplexity = run_epoch(data_valid, session, valid_model, valid_summary_writer)
                print("Epoch: %d Valid acc: %.3f, Valid cost: %.3f " % (
                    epoch + 1, valid_perplexity[0], valid_perplexity[1]))

            test_perplexity = run_epoch(data_test, session, test_model, test_summary_writer)
            print("Test acc: %.3f, Test cost:  %.3f" % (test_perplexity[0], test_perplexity[1]))


if __name__ == '__main__':
    main()
