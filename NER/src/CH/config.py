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
tf.app.flags.DEFINE_string("model_save_path", 'ckpt/bi-lstm-crf.ckpt', "model save path")
tf.app.flags.DEFINE_string("log_path", 'model/log/', "log save path")
# 这里默认词向量的维度是300, 如果自行训练的词向量维度不是300,则需要该这里的值。
tf.app.flags.DEFINE_integer("vocab_size", 20955, "Size of vocabulary.")

tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("max_sequence", 100, "max sequence length.")
tf.app.flags.DEFINE_integer("hidden_units", 300, "hidden units in lstm.")
tf.app.flags.DEFINE_integer("layers_num", 32, "hidden layers in lstm.")
tf.app.flags.DEFINE_integer("num_steps", 100, "max sequence length.")
tf.app.flags.DEFINE_integer("num_classes", 16, "num classes.")


tf.app.flags.DEFINE_integer("batch_size", 128, "batch size.")
tf.app.flags.DEFINE_integer("epoch", 30000, "epoch.")
tf.app.flags.DEFINE_float("dropout", 0.6, "drop out")
tf.app.flags.DEFINE_float("lr", 0.0001, "learning rate")
tf.app.flags.DEFINE_float("max_grad_norm", 1.0, "max_grad_norm.")

tf.app.flags.DEFINE_string("action", 'train', "train | predict")
FLAGS = tf.app.flags.FLAGS
