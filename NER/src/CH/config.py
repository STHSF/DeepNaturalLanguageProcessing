#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_string("src_file", 'data/source.txt', "Training data.")
tf.app.flags.DEFINE_string("tgt_file", 'data/target.txt', "labels.")
# 希望做命名识别的数据
tf.app.flags.DEFINE_string("pred_file", 'data/predict.txt', "test data.")
tf.app.flags.DEFINE_string("src_vocab_file", 'data/source_vocab.txt', "source vocabulary.")
tf.app.flags.DEFINE_string("tgt_vocab_file", 'data/target_vocab.txt', "targets.")
tf.app.flags.DEFINE_string("word_embedding_file", 'data/wiki.zh.vec', "extra word embeddings.")
tf.app.flags.DEFINE_string("model_path", 'data/model/', "model save path")
# 这里默认词向量的维度是300, 如果自行训练的词向量维度不是300,则需要该这里的值。
tf.app.flags.DEFINE_integer("embeddings_size", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("max_sequence", 100, "max sequence length.")

tf.app.flags.DEFINE_integer("batch_size", 128, "batch size.")
tf.app.flags.DEFINE_integer("epoch", 30000, "epoch.")
tf.app.flags.DEFINE_float("dropout", 0.6, "drop out")

tf.app.flags.DEFINE_string("action", 'train', "train | predict")
FLAGS = tf.app.flags.FLAGS


class Config():
    num_steps = 10
    vocab_size = 5188
    embedding_size = 128
    is_training = True
    hidden_units = 300
    keep_pro = 0.0001
    layers_num = 32
    batch_size = 128
    num_classes = 5
    lr = 0.001
    max_grad_norm = 1

    # glove_filename = "data/embedding/news_tensite_ch_clean.model".format(dim)
    # trimmed_filename = "data/news_tensite_ch_clean_{}d.trimmed.npz".format(dim)
    words_filename = "data/words.txt"
    tags_filename = "data/tags.txt"

    dev_filename = "data/msra/msr_training.utf8.val"
    test_filename = "data/msra/msr_training.utf8.test"
    train_filename = "data/msra/msr_training.utf8.train"
    max_iter = None
    lowercase = True
    train_embeddings = False
    nepochs = 20
    dropout = 0.5
    lr_decay = 0.9
    nepoch_no_imprv = 3
    crf = True  # if crf, training is 1.7x slower
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
