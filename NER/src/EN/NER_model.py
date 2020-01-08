# coding=utf-8
"""
"""

import tensorflow as tf


class NERModel(object):
    def __init__(self):
        self.vocab_size = 5199
        self.word_embedding_size = 128
        self.char_size = 52
        self.char_embedding_size = 60
        self.hidden_size_char = 32
        self.num_classes = 53

        # shape = (batch_size, max_length of sentence)
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="word_to_ids")
        self.target_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="target_to_ids")

        # shape = (batch_size)
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name="max_length_of_sentence")

        # shape = (batch_size, max_length of sentence, max_length of word)
        self.char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="char_to_ids")
        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(dtype=tf.int32, shape=[None, None], name="word_length")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.max_grad_norm = tf.placeholder(dtype=tf.float32, shape=[], name="max_grad_norm")

        self.word_embeddings = word_embedding(self.vocab_size, self.word_embedding_size, self.word_ids,
                                              self.char_ids, self.char_size, self.char_embedding_size,
                                              self.word_lengths, self.hidden_size_char)

        self.logits = add_logits_op(self.word_embeddings, self.hidden_size_char, self.sequence_length,
                                    self.dropout, self.num_classes)

        self.loss, self.trans_params = add_loss_op(self.logits, self.target_ids, self.sequence_length)

        self.train_op = add_train_op(self.loss, self.lr, self.max_grad_norm)


def word_embedding(vocab_size, word_embedding_size, word_ids, char_ids, char_size,
                   char_embedding_size, word_lengths, hidden_size_char):
    """Character Embedding using Bidirectional_RNN"""

    with tf.variable_scope("word_embedding"):
        _word_embedding = tf.Variable(tf.random_normal([vocab_size, word_embedding_size], -1.0, 1.0),
                                      dtype=tf.float32, name="_word_embedding")
        # shape = (batch_size, max_length of sentence, word_embedding_size)
        word_embeddings = tf.nn.embedding_lookup(_word_embedding, word_ids, name="word_embeddings")

    with tf.variable_scope("char_embedding"):
        _char_embedding = tf.Variable(tf.random_normal([char_size, char_embedding_size], -1.0, 1.0),
                                      dtype=tf.float32, name="_char_embedding")
        # shape = (batch_size, max_length of sentence, max_length of word, char_embedding_size)
        char_embeddings = tf.nn.embedding_lookup(_char_embedding, char_ids, name="char_embedding")

    # put the time dimension on axis=1
    s = tf.shape(char_embeddings)
    char_embeddings = tf.reshape(char_embeddings,
                                 shape=[s[0] * s[1], s[-2], char_embedding_size])
    word_lengths = tf.reshape(word_lengths, shape=[s[0] * s[1]])

    with tf.variable_scope("bi_lstm"):
        # bi lstm on chars
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_char, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_char, state_is_tuple=True)
        _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings,
                                                  sequence_length=word_lengths, dtype=tf.float32)

        # read and concat output
        _, ((_, output_fw), (_, output_bw)) = _output
        output = tf.concat([output_fw, output_bw], axis=-1)

    # shape = (batch_size, max_length of sentence, char_hidden_size * 2)
    output = tf.reshape(output, shape=[s[0], s[1], 2 * hidden_size_char])
    # shape = (batch_size, max_length of sentence, word_embedding_size + char_hidden_size * 2)
    word_embeddings = tf.concat([word_embeddings, output], axis=-1)

    return word_embeddings


def add_logits_op(word_embeddings, hidden_size_lstm, sequence_lengths, dropout, num_classes):
    """Defines self.logits
    For each word in each sentence of the batch, it corresponds to a vector
    of scores, of dimension equal to the number of tags.
    """
    with tf.variable_scope("bi-lstm"):
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, word_embeddings,
            sequence_length=sequence_lengths, dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.nn.dropout(output, dropout)

    with tf.variable_scope("output"):
        w = tf.get_variable("W", dtype=tf.float32,
                            shape=[2*hidden_size_lstm, num_classes])

        b = tf.get_variable("b", shape=[num_classes],
                            dtype=tf.float32, initializer=tf.zeros_initializer())

        n_steps = tf.shape(output)[1]
        output = tf.reshape(output, [-1, 2*hidden_size_lstm])
        pred = tf.matmul(output, w) + b
        logits = tf.reshape(pred, [-1, n_steps, num_classes])

    return logits


def add_loss_op(logits, labels, sequence_lengths):
    """Defines the loss"""
    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(logits, labels, sequence_lengths)
    trans_params = trans_params  # need to evaluate it for decoding
    loss = tf.reduce_mean(-log_likelihood)

    return loss, trans_params


def add_train_op(cost, lr, max_grad_norm):
    """optimize"""
    tvars = tf.trainable_variables()  # 获取模型的所有参数
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # 优化器

    # 梯度下降计算
    train_op = optimizer.apply_gradients(zip(grads, tvars),
                                         global_step=tf.contrib.framework.get_or_create_global_step())
    return train_op
