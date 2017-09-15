# coding=utf-8
import tensorflow as tf


class bi_lstm():
    """
    For Chinese word segmentation.
    """
    def __init__(self, is_training=True, hidden_units=128, timestep_size=32, vocab_size=5159, embedding_size=64,
                 num_classes=5, hidden_size=128, layers_num=2, max_grad_norm=5.0):
        # tf.reset_default_graph()  # 模型的训练和预测放在同一个文件下时如果没有这个函数会报错。
        self.is_training = is_training
        self.hidden_units = hidden_units
        self.num_steps = timestep_size
        self.max_len = self.max_len = timestep_size  # 句子长度
        self.vocab_size = vocab_size  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
        self.input_size = self.embedding_size = embedding_size  # 字向量长度
        self.num_classes = num_classes
        self.hidden_units = hidden_size  # 隐含层节点数
        self.layers_num = layers_num  # bi-lstm 层数
        self.max_grad_norm = max_grad_norm  # 最大梯度（超过此值的梯度将被裁剪）

        # 在训练和测试的时候，我们想用不同batch_size的数据，所以将batch_size也采用占位符的形式
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        # shape = [batch_size, num_steps]
        self._source_inputs = tf.placeholder(shape=(None, self.num_steps), dtype=tf.int32, name='source_inputs')
        # shape = [batch_size, num_steps]
        self._target_inputs = tf.placeholder(shape=(None, self.num_steps), dtype=tf.int32, name='target_inputs')

        with tf.device("/cpu:0"):
            _embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     dtype=tf.float32, name='embedding')
            # shape = [batch_size, num_steps, embedding_size]
            inputs_embedded = tf.nn.embedding_lookup(_embedding, self._source_inputs)

        self._cost, self._logits = bidi_lstm(inputs_embedded,
                                             self._target_inputs,
                                             self.layers_num,
                                             self.batch_size,
                                             self.hidden_units,
                                             self.num_classes)

        correct_prediction = tf.equal(tf.cast(tf.argmax(self._logits, 1), tf.int32), tf.reshape(self._target_inputs, [-1]))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # ***** 优化求解 *******
        tvars = tf.trainable_variables()  # 获取模型的所有参数
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), self.max_grad_norm)  # 获取损失函数对于每个参数的梯度
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)  # 优化器

        # 梯度下降计算
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())

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


def lstm_fw_cell(hidden_units):
    fw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                      reuse=tf.get_variable_scope().reuse,
                                      state_is_tuple=True)
    return fw_cell


def lstm_bw_cell(hidden_units):
    bw_cell = tf.contrib.rnn.LSTMCell(hidden_units,
                                      reuse=tf.get_variable_scope().reuse,
                                      state_is_tuple=True)
    return bw_cell


def bidi_lstm(inputs, target_inputs, layers_num, batch_size, hidden_units, num_classes):

    with tf.variable_scope('cell_fw'):
        cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell(hidden_units) for _ in range(layers_num)], state_is_tuple=True)
    with tf.variable_scope('cell_bw'):
        cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell(hidden_units) for _ in range(layers_num)], state_is_tuple=True)

    with tf.variable_scope('init_state_fw'):
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    with tf.variable_scope('init_state_bw'):
        initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    with tf.variable_scope('bi-lstm'):
        ((outputs_fw,
          outputs_bw),
        (final_state_fw,
         final_state_bw)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                             cell_bw=cell_bw,
                                                             inputs=inputs,
                                                             initial_state_fw=initial_state_fw,
                                                             initial_state_bw=initial_state_bw,
                                                             dtype=tf.float32,
                                                             time_major=False))

    # shape = [batch_size, num_steps, hidden_units * 2]
    with tf.name_scope('outputs'):
        _outputs = tf.concat((outputs_fw, outputs_bw), 2)

        # final_state_c = tf.concat((final_state_fw.c, final_state_bw.c), 1)
        # final_state_h = tf.concat((final_state_fw.h, final_state_bw.h), 1)
        # final_state = tf.contrib.rnn.LSTMStateTuple(c=final_state_c,
        #                                             h=final_state_h)

    # shape = [batch_size * num_steps, hidden_units * 2]
    outputs = tf.reshape(_outputs, [-1, hidden_units * 2], name='predict')

    with tf.variable_scope('output_layer'):
        softmax_w = tf.Variable(tf.truncated_normal(shape=[hidden_units * 2, num_classes], stddev=0.1),
                                name="soft_max_w")
        softmax_b = tf.Variable(tf.constant(1.0, shape=[num_classes]), name="soft_max_b")
        # softmax_w = self.weight_variable([self.hidden_units * 2, self.num_classes])
        # softmax_b = self.bias_variable([self.num_classes])

        # shape = [batch_size * num_steps, hidden_units * 2]
        with tf.name_scope('logits'):
            logits = tf.matmul(outputs, softmax_w) + softmax_b

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(target_inputs, [-1]),
                                                          logits=logits,
                                                          name='loss')
    cost = tf.reduce_mean(loss, name='cost')
    return cost, logits


print 'Finished creating the bi-lstm model.'
