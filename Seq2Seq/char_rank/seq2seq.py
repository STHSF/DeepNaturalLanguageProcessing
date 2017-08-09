# coding=utf-8
import tensorflow as tf


class seq2seqModel():
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_units, decoder_hidden_units):
        tf.reset_default_graph()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        # encoder_inputs:[batch, max_time]
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        # decoder_inputs:[batch, max_time]
        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
        # decoder_targets:[batch, max_time]
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

        self.get_encoder_layer()
        self.get_decoder_layer()
        self.compute_loss()
        self.optimizer()

    def embedding(self, input_data):
        embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0), dtype=tf.float32)
        embedded = tf.nn.embedding_lookup(embedding, input_data)
        return embedded

    def get_encoder_layer(self):
        # encoder_inputs_embedded:[batch_size, max_time, embedding_dim]
        encoder_inputs_embedded = self.embedding(self.encoder_inputs)
        with tf.variable_scope('encoder_cell'):
            encoder_cell = tf.contrib.rnn.LSTMCell(self.encoder_hidden_units)
        # encoder_output:[batch_size, max_time, encoder_hidden_units]
        # encoder_final_state:[batch_size, encoder_hidden_units]
        self.encoder_output, self.encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                                          encoder_inputs_embedded,
                                                                          dtype=tf.float32,
                                                                          time_major=True,
                                                                          scope='encode_cell')
        del self.encoder_output

    def get_decoder_layer(self):
        # decoder_inputs_embedded:[batch_size, max_time, embedding_dim]
        decoder_inputs_embedded = self.embedding(self.decoder_inputs)
        with tf.variable_scope('decoder_cell'):
            decoder_cell = tf.contrib.rnn.LSTMCell(self.decoder_hidden_units)
        # decoder_outputs:[batch_size, max_time, decoder_hidden_units]
        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                                           decoder_inputs_embedded,
                                                                           initial_state=self.encoder_final_state,
                                                                           dtype=tf.float32,
                                                                           time_major=True,
                                                                           scope='decode_cell')

    def compute_loss(self):
        # 使用fully_connected作为输出层的输出。
        # decoder_logits:[batch_size, max_time, vocab_size]
        self.decoder_logits = tf.contrib.layers.fully_connected(self.decoder_outputs, self.vocab_size)

        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.decoder_targets,
                                                                                           depth=self.vocab_size,
                                                                                           dtype=tf.float32),
                                                                         logits=self.decoder_logits)
        self.loss = tf.reduce_mean(stepwise_cross_entropy)

    def optimizer(self):
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)