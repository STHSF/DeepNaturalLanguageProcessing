# coding=utf-8
"""
Concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
Concatenate this representation to a standard word vector representation (GloVe here)
Run a bi-lstm on each sentence to extract contextual representation of each word
Decode with a linear chain CRF
"""
import tensorflow as tf


class bi_lstm_crf(object):
    def __init__(self, config):
        self.num_steps = config.num_steps
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.is_training = config.is_training
        self.hidden_units = config.hidden_units
        self.keep_pro = config.keep_pro
        self.layers_num = config.layers_num
        self.batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.lr = config.lr
        self.max_grad_norm = config.max_grad_norm
        # shape = (batch_size, num_steps)
        self.source_input = tf.placeholder(tf.int32, shape=[None, self.num_steps], name="source_input")
        # shape = (batch_size, num_steps)
        self.target_input = tf.placeholder(tf.int32, shape=[None, self.num_steps], name="labels")

        with tf.variable_scope("embedding"):
            _embedding = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     dtype=tf.float32, name="embedding")
            # shape = (batch_size, num_steps, embedding_size)
            target_inputs_embedding = tf.nn.embedding_lookup(_embedding, self.source_input)

        self.bi_cell_outputs = bi_RNN(target_inputs_embedding, self.is_training, self.hidden_units,
                                      self.keep_pro, self.layers_num, self.batch_size)

        self.logits = add_output_layer(self.bi_cell_outputs, self.hidden_units, self.num_classes)

        self.cost = cost_crf(self.logits, target_inputs_embedding, self.num_steps)

        self.train_op = train_operation(self.cost, self.lr, self.max_grad_norm)


def fw_rnn_cell(is_training, hidden_units, keep_prob):
    fw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                      reuse=tf.get_variable_scope().reuse,
                                      state_is_tuple=True)
    if is_training:
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,
                                                input_keep_prob=1.0,
                                                output_keep_prob=keep_prob)
    return fw_cell


def bw_rnn_cell(is_training, hidden_units, keep_prob):
    bw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                      reuse=tf.get_variable_scope().reuse,
                                      state_is_tuple=True)
    if is_training:
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
    return outputs


def add_output_layer(outputs, hidden_units, num_classes):
    """
    Computing Tag Scores
    """
    with tf.variable_scope("output_layer"):
        softmax_w = tf.Variable(tf.truncated_normal(shape=[hidden_units * 2, num_classes], stddev=0.1),
                                name="softmax_w")
        softmax_b = tf.Variable(tf.constant(1.0, shape=[num_classes]), name="softmax_b")

    with tf.variable_scope("logits"):
        logits = tf.multinomial(outputs, softmax_w) + softmax_b

    return logits


def cost_crf(logits, labels, sequence_length):
    """
    Decoding the score with crf
    """
    with tf.variable_scope("crf"):
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, labels, sequence_length)

    with tf.variable_scope("cost"):
        cost = tf.reduce_mean(-log_likelihood)

    return cost


def train_operation(cost, lr, max_grad_norm):
    # ***** 优化求解 *******
    tvars = tf.trainable_variables()  # 获取模型的所有参数
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # 优化器

    # 梯度下降计算
    train_op = optimizer.apply_gradients(zip(grads, tvars),
                                         global_step=tf.contrib.framework.get_or_create_global_step())
    return train_op
