# coding=utf-8
import tensorflow as tf


class bi_lstm(object):
    """
    Construct a Bidirection_RNN for word ChineseSegmentation.
    For Chinese word ChineseSegmentation.
    """
    def __init__(self, is_training=False, hidden_units=128, timestep_size=32, vocab_size=5159,
                 embedding_size=64, num_classes=5, hidden_size=128, layers_num=2, max_grad_norm=5.0):
        # tf.reset_default_graph()  # 模型的训练和预测放在同一个文件下时如果没有这个函数会报错。
        self.is_training = is_training
        self.hidden_units = hidden_units
        self.n_timesteps = timestep_size
        self.max_len = timestep_size  # 句子长度
        self.vocab_size = vocab_size  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
        self.embedding_size = embedding_size  # 字向量长度
        self.num_classes = num_classes
        self.hidden_units = hidden_size  # 隐含层节点数
        self.layers_num = layers_num  # bi-lstm 层数
        self.max_grad_norm = max_grad_norm  # 最大梯度（超过此值的梯度将被裁剪）

        # 在训练和测试的时候，我们想用不同batch_size的数据，所以将batch_size也采用占位符的形式
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        # shape = [batch_size, num_steps]
        self._source_inputs = tf.placeholder(shape=(None, self.n_timesteps), dtype=tf.int32, name='source_inputs')
        # shape = [batch_size, num_steps]
        self._target_inputs = tf.placeholder(shape=(None, self.n_timesteps), dtype=tf.int32, name='target_inputs')

        with tf.device("/cpu:0"):
            _embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     dtype=tf.float32, name='embedding')
            # shape = [batch_size, num_steps, embedding_size]
            inputs_embedded = tf.nn.embedding_lookup(_embedding, self._source_inputs)

        bi_lstm_output = bi_RNN(self.is_training, inputs_embedded, self.layers_num,
                                self.batch_size, self.hidden_units, self.keep_prob)

        self._logits = add_output_layer(bi_lstm_output, self.hidden_units, self.num_classes)

        self._cost = cost_compute(self._logits, self.target_inputs, self.num_classes)

        correct_prediction = tf.equal(tf.cast(tf.argmax(self._logits, 1), tf.int32), tf.reshape(self._target_inputs, [-1]))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._train_op = train_operation(self._cost, self.lr, self.max_grad_norm)

        # 模型保存
        self.saver = tf.train.Saver(tf.global_variables())

    @property
    def source_inputs(self):
        return self._source_inputs

    @property
    def target_inputs(self):
        return self._target_inputs

    @property
    def logits(self):
        return self._logits

    @property
    def loss(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def accuracy(self):
        return self._accuracy


def lstm_fw_cell(hidden_units, is_training, keep_prob):
    fw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                      reuse=tf.get_variable_scope().reuse,
                                      state_is_tuple=True)
    if is_training:
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,
                                                input_keep_prob=1.0,
                                                output_keep_prob=keep_prob)
    return fw_cell


def lstm_bw_cell(hidden_units, is_training, keep_prob):
    bw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                      reuse=tf.get_variable_scope().reuse,
                                      state_is_tuple=True)
    if is_training:
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,
                                                input_keep_prob=1.0,
                                                output_keep_prob=keep_prob)
    return bw_cell


def bi_RNN(is_training, inputs, layers_num, batch_size, hidden_units, keep_prob):

    with tf.variable_scope('multi_cell_fw'):
        multi_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell(hidden_units, is_training, keep_prob) for _ in range(layers_num)],
                                                    state_is_tuple=True)
    with tf.variable_scope('multi_cell_bw'):
        multi_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell(hidden_units, is_training, keep_prob) for _ in range(layers_num)],
                                                    state_is_tuple=True)

    with tf.variable_scope('init_state_fw'):
        initial_state_fw = multi_cell_fw.zero_state(batch_size, tf.float32)
    with tf.variable_scope('init_state_bw'):
        initial_state_bw = multi_cell_bw.zero_state(batch_size, tf.float32)

    with tf.variable_scope('bi_RNN'):
        ((outputs_fw,
          outputs_bw),
        (final_state_fw,
         final_state_bw)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_cell_fw,
                                                             cell_bw=multi_cell_bw,
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
    with tf.variable_scope('output_layer'):
        softmax_w = tf.Variable(tf.truncated_normal(shape=[hidden_units * 2, num_classes], stddev=0.1),
                                name="soft_max_w")
        softmax_b = tf.Variable(tf.constant(1.0, shape=[num_classes]), name="soft_max_b")
        # softmax_w = self.weight_variable([self.hidden_units * 2, self.num_classes])
        # softmax_b = self.bias_variable([self.num_classes])

        # shape = [batch_size * num_steps, num_classes]
        with tf.name_scope('logits'):
            logits = tf.matmul(outputs, softmax_w) + softmax_b
    return logits


def add_crf_layer(logits, labels, sequence_lengths):
    with tf.variable_scope("crf"):
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, labels, sequence_lengths)

    loss = tf.reduce_mean(-log_likelihood)

    return loss, transition_params


def cost_compute(logits, target_inputs, num_classes):
    # shape = [batch_size * num_steps, ]
    # labels'shape = [batch_size * num_steps, num_classes]
    # logits'shape = [shape = [batch_size * num_steps, num_classes]]
    # 这里可以使用tf.nn.sparse_softmax_cross_entropy_with_logits()和tf.nn.softmax_cross_entropy_with_logits()两种方式来计算rnn
    # 但要注意labels的shape。
    # eg.1
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(target_inputs, [-1]),
    #                                                       logits=logits, name='loss')

    # eg.2
    targets = tf.one_hot(target_inputs, num_classes)  # [batch_size, seq_length, num_classes]
    # 不能使用logit.get_shape(), 因为在定义logit时shape=[None, num_steps], 这里使用会报错
    # y_reshaped = tf.reshape(targets, logits.get_shape())  # y_reshaped: [batch_size * seq_length, num_classes]
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1, num_classes]),
                                                   logits=logits, name='loss')

    cost = tf.reduce_mean(loss, name='cost')
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


print('Finished creating the bi-lstm model.')
