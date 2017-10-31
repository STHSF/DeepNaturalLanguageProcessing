# coding=utf-8
"""
Concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
Concatenate this representation to a standard word vector representation (GloVe here)
Run a bi-lstm on each sentence to extract contextual representation of each word
Decode with a linear chain CRF
"""
import tensorflow as tf
import numpy as np


class bi_lstm_crf(object):
    def __init__(self, num_steps, vocab_size, embedding_size, hidden_units, layers_num, num_classes):
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.layers_num = layers_num
        self.num_classes = num_classes

        # 在训练和测试的时候，我们想用不同batch_size的数据，所以将batch_size也采用占位符的形式
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        self.is_training = tf.placeholder(tf.bool, [])
        self.lr = tf.placeholder(tf.float32, [])
        self.max_grad_norm = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        # shape = (batch_size, num_steps)
        self.source_input = tf.placeholder(dtype=tf.int32, shape=(None, self.num_steps), name="source_input")
        # shape = (batch_size, num_steps)
        self.target_input = tf.placeholder(dtype=tf.int32, shape=(None, self.num_steps), name="labels")

        with tf.variable_scope("embedding"):
            _embedding = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     dtype=tf.float32, name="embedding")
            # shape = (batch_size, num_steps, embedding_size)
            target_inputs_embedding = tf.nn.embedding_lookup(_embedding, self.source_input)
        # shape = [batch_size * num_steps, hidden_units * 2]
        bi_cell_outputs = bi_RNN(target_inputs_embedding, self.is_training, self.hidden_units,
                                 self.keep_prob, self.layers_num, self.batch_size)
        # shape = [batch_size, num_steps, num_classes]
        self.logits = add_output_layer(bi_cell_outputs, self.hidden_units, self.num_classes)
        # print('shape of self.unary_scores', np.shape(self.unary_scores))
        # print('shape of self.target_inputs', np.shape(self.target_input))
        # print('shape of self.batch_size', np.shape(self.batch_size), self.batch_size)
        # print('shape of self.num_steps', np.shape(self.num_steps), self.num_steps)

        self.cost, self.transition_params = cost_crf(self.logits, self.target_input, self.batch_size,
                                                     self.num_steps, self.num_classes)

        # self.accuracy = acc(self.logits, self.target_input, self.transition_params, self.batch_size, self.num_classes)
        # self.accuracy = (self.logits, self.target_input)

        self.train_op = train_operation(self.cost, self.lr, self.max_grad_norm)
        # 模型保存
        self.saver = tf.train.Saver(tf.global_variables())


def fw_rnn_cell(is_training, hidden_units, keep_prob):
    fw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                      reuse=tf.get_variable_scope().reuse,
                                      state_is_tuple=True)
    if is_training is not None:
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,
                                                input_keep_prob=1.0,
                                                output_keep_prob=keep_prob)
    return fw_cell


def bw_rnn_cell(is_training, hidden_units, keep_prob):
    bw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                      reuse=tf.get_variable_scope().reuse,
                                      state_is_tuple=True)
    if is_training is not None:
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,
                                                input_keep_prob=1.0,
                                                output_keep_prob=keep_prob)
    return bw_cell


def bi_RNN(inputs, is_training, hidden_units, keep_prob, layers_num, batch_size):
    with tf.variable_scope("fw_cell"):
        multi_fw_cell = tf.contrib.rnn.MultiRNNCell(
            [fw_rnn_cell(is_training, hidden_units, keep_prob) for _ in range(layers_num)],
            state_is_tuple=True)

    with tf.variable_scope("bw_cell"):
        multi_bw_cell = tf.contrib.rnn.MultiRNNCell(
            [bw_rnn_cell(is_training, hidden_units, keep_prob) for _ in range(layers_num)],
            state_is_tuple=True)

    with tf.variable_scope("init_state_fw"):
        initial_state_fw = multi_fw_cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope("init_state_bw"):
        initial_state_bw = multi_bw_cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope("bi_RNN"):
        ((outputs_fw,
          outputs_bw),
         (output_state_fw,
          output_state_bw)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_fw_cell,
                                                               cell_bw=multi_bw_cell,
                                                               inputs=inputs,
                                                               initial_state_fw=initial_state_fw,
                                                               initial_state_bw=initial_state_bw,
                                                               dtype=tf.float32,
                                                               time_major=False))

    # shape = [batch_size, num_steps, hidden_units * 2]
    with tf.name_scope('outputs'):
        _outputs = tf.concat((outputs_fw, outputs_bw), 2, name='_outputs')

        # final_state_c = tf.concat((final_state_fw.c, final_state_bw.c), 1)
        # final_state_h = tf.concat((final_state_fw.h, final_state_bw.h), 1)
        # final_state = tf.contrib.rnn.LSTMStateTuple(c=final_state_c,
        #                                             h=final_state_h)

    # shape = [batch_size * num_steps, hidden_units * 2]
    outputs = tf.reshape(_outputs, [-1, hidden_units * 2], name='predict')
    print ("LSTM NN layer output size:")
    print (outputs.get_shape())
    return outputs


def add_output_layer(outputs, hidden_units, num_classes):
    """
    Computing Tag Scores
    """
    with tf.variable_scope("output_layer"):
        softmax_w = tf.Variable(tf.truncated_normal(shape=[hidden_units * 2, num_classes], stddev=0.1), name="softmax_w")
        softmax_b = tf.Variable(tf.constant(1.0, shape=[num_classes]), name="softmax_b")

    with tf.variable_scope("logits"):
        # shape = (batch_size * num_steps, num_classes)
        logits = tf.matmul(outputs, softmax_w) + softmax_b
        print('Size of logits', logits.get_shape())

    return logits


def cost_crf(logits, labels, batch_size, sequence_length, num_classes):
    """
    Decoding the score with crf
    :param logits: [batch_size * max_seq_len，num_tags]
    :param labels:  [batch_size，max_seq_len]
    :param batch_size: batch_size
    :param sequence_length: max_seg_len
    :param num_classes: num_classes
    :return:
    transition_params: [num_tags, num_tags]
    """
    # shape = (batch_size, num_steps, num_classes)
    unary_scores = tf.reshape(logits, [batch_size, -1, num_classes])
    print('size of unary_scores', np.shape(unary_scores))
    print('size of labels', np.shape(labels))

    with tf.variable_scope("crf"):
        # shape: [batch_size], value: [T-1, T-1,...]
        _sequence_lengths = tf.tile([sequence_length], [batch_size])
        # _sequence_lengths = tf.constant(np.full(batch_size, sequence_length, dtype=np.int32))
        print('size of sequence_length', np.shape(_sequence_lengths))

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, labels, _sequence_lengths)

    # with tf.variable_scope('verterbi_decode'):
    #     # Compute the highest scoring sequence.
    #     viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params)

    with tf.variable_scope("cost"):
        cost = tf.reduce_mean(-log_likelihood)

    return cost, transition_params


def train_operation(cost, lr, max_grad_norm):
    # ***** 优化求解 *******
    tvars = tf.trainable_variables()  # 获取模型的所有参数
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # 优化器

    # 梯度下降计算
    train_op = optimizer.apply_gradients(zip(grads, tvars),
                                         global_step=tf.contrib.framework.get_or_create_global_step())
    return train_op


# def acc(logits, target_inputs):
#     # Evaluate word-level accuracy.
#     correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(target_inputs, [-1]))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     # correct_prediction = tf.equal(viterbi_sequence, target_input)
#     # accuracy = tf.reducacce_mean(tf.cast(correct_prediction, tf.float32))
#
#     return accuracy


# def acc(logits, target_inputs, transition_params, batch_size, num_classes):
#     accuracy = 0
#     correct_labels = 0  # prediction accuracy
#     total_labels = 0
#     # shape = (batch_size, num_steps, num_classes)
#     unary_scores = tf.reshape(logits, [batch_size, -1, num_classes])
#     # iterate over batches [batch_size, num_steps, target_num], [batch_size, target_num]
#     for unary_score_, y_ in zip(unary_scores, target_inputs):  # unary_score_  :[num_steps, target_num], y_: [num_steps]
#         viterbi_prediction = tf.contrib.crf.viterbi_decode(unary_score_, transition_params)
#         # viterbi_prediction: tuple (list[id], value)
#         # y_: tuple
#         correct_labels += np.sum(
#             np.equal(viterbi_prediction[0], y_))  # compare prediction sequence with golden sequence
#         total_labels += len(y_)
#         print ("viterbi_prediction")
#         print (viterbi_prediction)
#         accuracy = 100.0 * correct_labels / float(total_labels)
#
#     return accuracy
