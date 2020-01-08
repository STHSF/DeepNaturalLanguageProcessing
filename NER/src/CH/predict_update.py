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
from sequence_labelling_ner_crf import SequenceLabelingModel


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

print('X_train.shape={}, y_train.shape={}; '
      '\nX_valid.shape={}, y_valid.shape={};'
      '\nX_test.shape={}, y_test.shape={}'.format(X_train.shape,
                                                  y_train.shape,
                                                  X_valid.shape,
                                                  y_valid.shape,
                                                  X_test.shape,
                                                  y_test.shape))

print('Creating the data generator ...')
data_train = BatchGenerator(X_train, y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
print('Finished creating the data generator.')


class configuration(object):
    # hyper-parameter
    init_scale = config.FLAGS.init_scale
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


train_config = configuration()
eval_config = configuration()
eval_config.batch_size = 1
decay = 0.85
max_epoch = 3
max_max_epoch = 10


def main():
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                                    train_config.init_scale)
        with tf.name_scope("Train") as train_scope:
            with tf.variable_scope("Train_Model", reuse=None, initializer=initializer):
                train_model = SequenceLabelingModel(train_scope, is_training=True, config=train_config)
                tf.summary.scalar("Training Loss", train_model.loss)
                tf.summary.scalar("Learning Rate", train_model.lr)

        with tf.name_scope("Valid") as valid_scope:
            with tf.variable_scope("Valid_Model", reuse=True, initializer=initializer):
                valid_model = SequenceLabelingModel(valid_scope, is_training=False, config=train_config)
                tf.summary.scalar("Validation Loss", valid_model.loss)

        with tf.Session() as session:
            train_summary_writer = tf.summary.FileWriter('data/model/tensorflowlogs/train', session.graph)
            valid_summary_writer = tf.summary.FileWriter('data/model/tensorflowlogs/valid', session.graph)
            session.run(tf.global_variables_initializer())

            for epoch in range(max_max_epoch):
                _batch_size = train_model.batch_size
                print("model_batch_size in run_epoch", _batch_size)
                data_size = data_train.y.shape[0]
                print("data_size %s", data_size)
                batch_num = int(data_size / _batch_size)
                print("batch_num: %d", batch_num)
                for batch in range(batch_num):
                    fetches = [train_model.logits, train_model.transition_params, train_model.loss, train_model.train_op]

                    X_batch, y_batch = data_train.next_batch(_batch_size)
                    feed_dict = {train_model.source_input: X_batch,
                                 train_model.target_input: y_batch}
                    _logits, _transition_params, _loss, _ = session.run(fetches, feed_dict)
                    print _loss


if __name__ == '__main__':
    main()