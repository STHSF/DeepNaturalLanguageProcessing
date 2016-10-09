"""Tests for word2vec module."""

import os

import tensorflow as tf

from tensorflow.models.embedding import word2vec

flags = tf.app.flags

FLAGS = flags.FLAGS


class Word2Vec_model(tf.test.TestCase):

    def setUp(self):
        # FLAGS.train_data = os.path.join(self.get_temp_dir() + "text8")
        FLAGS.train_data = "/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/embedding/text8"
        # FLAGS.eval_data = os.path.join(self.get_temp_dir() + "questions-words.txt")
        FLAGS.eval_data = "/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing" \
                          "/DeepNLP/embedding/questions-words.txt"
        # FLAGS.save_path = self.get_temp_dir()
        FLAGS.save_path = "/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/embedding/tmp"

    def word2vec_model(self):
        FLAGS.batch_size = 5
        FLAGS.num_neg_samples = 10
        FLAGS.epochs_to_train = 1
        FLAGS.min_count = 0
        word2vec.main([])


Word2Vec_model.setUp()
Word2Vec_model.word2vec_model()




