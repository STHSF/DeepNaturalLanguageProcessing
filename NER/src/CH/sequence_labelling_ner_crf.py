# coding=utf-8

import tensorflow as tf
import numpy as np


class SequenceLabelingModel(object):
    def __init__(self, scope_name, config, is_training):
        self._num_steps = config.num_steps  # 序列长度
        self._vocab_size = config.vocab_size  # 词汇量个数， 用于embedding_up
        self._embedding_size = config.embedding_size  #
        self._hidden_units = config.hidden_units  # 隐藏单元的个数
        self._layers_num = config.layers_num  # 神经网络的层数
        self._num_classes = config.num_classes  # 分类的类别数
        self._max_grad_norm = config.max_grad_norm
        self._keep_prob = config.keep_pro
        self._batch_size = config.batch_size
        self._is_training = is_training

        with tf.variable_scope(scope_name):
            self._build_graph()

    def _build_graph(self):
        # 在训练和测试的时候，我们想用不同batch_size的数据，所以将batch_size也采用占位符的形式
        # self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")  # 注意类型必须为 tf.int32
        # self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        # self.lr = tf.placeholder(tf.float32, [], name="lr")
        # self.max_grad_norm = tf.placeholder(tf.float32, [], name="max_grad_norm")
        # self.keep_prob = tf.placeholder(tf.float32, [], name="keep_pro")
        with tf.variable_scope('source_input'):
            # shape = (batch_size, num_steps)
            self.source_input = tf.placeholder(dtype=tf.int32, shape=(None, self._num_steps), name="source_input")
        with tf.variable_scope('target_input'):
            # shape = (batch_size, num_steps)
            self.target_input = tf.placeholder(dtype=tf.int32, shape=(None, self._num_steps), name="labels")

        with tf.variable_scope("embedding"):
            _embedding = tf.Variable(tf.random_normal([self._vocab_size, self._embedding_size], -1.0, 1.0),
                                     dtype=tf.float32, name="embedding_initial")

        with tf.variable_scope("embedding_lookup"):
            # shape = (batch_size, num_steps, embedding_size)
            source_inputs_embedding = tf.nn.embedding_lookup(_embedding, self.source_input, name="inputs_embedding")

        with tf.variable_scope('bidirecrional_LSTM'):
            # shape = [batch_size * num_steps, hidden_units * 2]
            bi_cell_outputs = self._bidirectional_rnn(source_inputs_embedding, self._is_training, self._hidden_units,
                                                      self._keep_prob, self._layers_num, self._batch_size)
            bi_cell_outputs = tf.nn.dropout(bi_cell_outputs, keep_prob=1.0-self._keep_prob, name='lstm_dropout')

        # shape = [batch_size, num_steps, num_classes]
        with tf.variable_scope('output_layer'):
            self.logits = self._output_layer(bi_cell_outputs, self._hidden_units, self._batch_size, self._num_classes)

        with tf.variable_scope('crf'):
            self.loss, self.transition_params = self._loss_crf(self.logits, self.target_input, self._batch_size,
                                                               self._num_steps, self._num_classes)

        # self.accuracy = acc(self.logits, self.target_input, self.transition_params, self.batch_size, self.num_classes)
        # self.accuracy = (self.logits, self.target_input)

        if not self._is_training:
            return

        # 优化求解
        self.lr = tf.Variable(0.0001, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          self._max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                  global_step=tf.train.get_or_create_global_step())

        # self.train_op = train_operation(self.cost, self.lr, self.max_grad_norm)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self._new_lr)

        with tf.name_scope('saver'):
            # 模型保存
            self.saver = tf.train.Saver(tf.global_variables())

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def _fw_cell(self, is_training, hidden_units, keep_prob):

        fw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                          reuse=tf.get_variable_scope().reuse,
                                          state_is_tuple=True)

        if is_training:
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,
                                                    input_keep_prob=1.0,
                                                    output_keep_prob=keep_prob)

        # fw_cell_out = tf.cond(is_training, lambda: fw_cell_drop, lambda: fw_cell, name="f_conf")
        return fw_cell

    def _bw_cell(self, is_training, hidden_units, keep_prob):
        bw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                          reuse=tf.get_variable_scope().reuse,
                                          state_is_tuple=True)
        if is_training:
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,
                                                    input_keep_prob=1.0,
                                                    output_keep_prob=keep_prob)
        # bw_cell_out = tf.cond(is_training, lambda: bw_cell_drop, lambda: bw_cell, name="cond")
        return bw_cell

    def _bidirectional_rnn(self, inputs, is_training, hidden_units, keep_prob, layers_num, batch_size):
        with tf.variable_scope("fw_cell", reuse=tf.AUTO_REUSE):
            multi_fw_cell = tf.contrib.rnn.MultiRNNCell(
                [self._fw_cell(is_training, hidden_units, keep_prob) for _ in range(layers_num)],
                state_is_tuple=True)

        with tf.variable_scope("bw_cell", reuse=tf.AUTO_REUSE):
            multi_bw_cell = tf.contrib.rnn.MultiRNNCell(
                [self._bw_cell(is_training, hidden_units, keep_prob) for _ in range(layers_num)],
                state_is_tuple=True)

        with tf.variable_scope("init_state_fw", reuse=tf.AUTO_REUSE):
            initial_state_fw = multi_fw_cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope("init_state_bw", reuse=tf.AUTO_REUSE):
            initial_state_bw = multi_bw_cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope("dynamic_rnn", reuse=tf.AUTO_REUSE):
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

        # shape = [batch_size * num_steps, hidden_units * 2]
        outputs = tf.reshape(_outputs, [-1, hidden_units * 2], name='predict')
        tf.summary.histogram('outputs', outputs)
        print ("LSTM NN layer output size:")
        print (outputs.get_shape())
        return outputs

    def _output_layer(self, outputs, hidden_units, batch_size, num_classes):
        """
        Computing Tag Scores
        """
        with tf.variable_scope("w_plus_b", reuse=tf.AUTO_REUSE):
            with tf.variable_scope('weight', reuse=tf.AUTO_REUSE):
                softmax_w = tf.get_variable(name='softmax_w',
                                            initializer=tf.truncated_normal(shape=[hidden_units * 2, num_classes],
                                                                            stddev=0.1))
            with tf.variable_scope('bais', reuse=tf.AUTO_REUSE):
                softmax_b = tf.get_variable(name='softmax_b',
                                            initializer=tf.zeros(shape=[num_classes]))
            # softmax_b = tf.get_variable('softmax_b', initializer=tf.ones(shape=[num_classes]))

        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
            # shape = (batch_size * num_steps, num_classes)
            _logits = tf.matmul(outputs, softmax_w) + softmax_b
            logits = tf.reshape(_logits, [batch_size, -1, num_classes])
            print('Size of logits', logits.get_shape())

        return logits

    def _loss_crf(self, logits, labels, batch_size, sequence_length, num_classes):
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

        with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
            # shape: [batch_size], value: [T-1, T-1,...]
            _sequence_lengths = tf.tile([sequence_length], [batch_size], name="_sequence_lengths")
            # _sequence_lengths = tf.constant(np.full(batch_size, sequence_length, dtype=np.int32))
            print('size of sequence_length', np.shape(_sequence_lengths))

            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores,
                                                                                  labels,
                                                                                  _sequence_lengths)

        # with tf.variable_scope('verterbi_decode'):
        #     # Compute the highest scoring sequence.
        #     viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params)

        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            loss = tf.reduce_mean(-log_likelihood, name="reduce_mean")

        return loss, transition_params

    @property
    def batch_size(self):
        return self._batch_size




