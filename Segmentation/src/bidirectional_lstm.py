# coding=utf-8
import tensorflow as tf


class bi_lstm():
    """
    For Chinese word Segmentation.
    """
    def __init__(self, hidden_units=128,
                 timestep_size=32, vocab_size=5159, embedding_size=64,
                 num_classes=5, hidden_size=128, layers_num=2,
                 max_grad_norm=5.0):
        # tf.reset_default_graph()  # 模型的训练和预测放在同一个文件下时如果没有这个函数会报错。
        self.hidden_units = hidden_units
        self.timestep_size = timestep_size
        self.max_len = self.max_len = timestep_size  # 句子长度
        self.vocab_size = vocab_size  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
        self.input_size = self.embedding_size = embedding_size # 字向量长度
        self.num_classes = num_classes
        self.hidden_units = hidden_size  # 隐含层节点数
        self.layers_num = layers_num  # bi-lstm 层数
        self.max_grad_norm = max_grad_norm  # 最大梯度（超过此值的梯度将被裁剪）

        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        self.source_inputs = tf.placeholder(shape=(None, self.timestep_size), dtype=tf.int32, name='source_inputs')
        self.target_inputs = tf.placeholder(shape=(None, self.timestep_size), dtype=tf.int32, name='y_inputs')

        with tf.variable_scope("embedding"):
            self.embedding()
        with tf.variable_scope("lstm_cell"):
            self.lstm_cell()
        with tf.variable_scope("bi_lstm"):
            self.bi_lstm()
        with tf.variable_scope("output_layer"):
            self.output_layer()
        with tf.variable_scope("loss_compute"):
            self.loss_compute()
        with tf.name_scope('train'):
            self.train_operater()
        with tf.variable_scope("loss_compute"):
            self.acc()

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def embedding(self):
        embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), dtype=tf.float32, name='embedding')
        # shape = [batch_size, num_steps, embedding_size]
        self.inputs_embedded = tf.nn.embedding_lookup(embedding, self.source_inputs)
        return self.inputs_embedded

    def lstm_cell(self):
        self.cell = tf.contrib.rnn.LSTMCell(self.hidden_units,
                                            reuse=tf.get_variable_scope().reuse,
                                            state_is_tuple=True)
        return self.cell

    def bi_lstm(self):
        cell_fw = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layers_num)], state_is_tuple=True)
        cell_bw = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layers_num)], state_is_tuple=True)

        initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('bi-lstm'):
            ((outputs_fw,
              outputs_bw),
             (final_state_fw,
              final_state_bw)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                  cell_bw=cell_bw,
                                                                  inputs=self.embedding(),
                                                                  initial_state_fw=initial_state_fw,
                                                                  initial_state_bw=initial_state_bw,
                                                                  dtype=tf.float32,
                                                                  time_major=False))

        # shape = [batch_size, num_steps, hidden_units * 2]
        with tf.name_scope('outputs'):
            self._outputs = tf.concat((outputs_fw, outputs_bw), 2)

        # final_state_c = tf.concat((final_state_fw.c, final_state_bw.c), 1)
        # final_state_h = tf.concat((final_state_fw.h, final_state_bw.h), 1)
        # final_state = tf.contrib.rnn.LSTMStateTuple(c=final_state_c,
        #                                             h=final_state_h)

    def output_layer(self):
        # shape = [batch_size * num_steps, hidden_units * 2]
        outputs = tf.reshape(self._outputs, [-1, self.hidden_units * 2])

        with tf.variable_scope('output_layer'):
            softmax_w = tf.Variable(tf.truncated_normal(shape=[self.hidden_units * 2, self.num_classes], stddev=0.1), name="soft_max_w")
            softmax_b = tf.Variable(tf.constant(1.0, shape=[self.num_classes]), name="soft_max_b")
            # softmax_w = self.weight_variable([self.hidden_units * 2, self.num_classes])
            # softmax_b = self.bias_variable([self.num_classes])

            # shape = [batch_size * num_steps, hidden_units * 2]
            with tf.name_scope('pred'):
                self._y_pred = tf.matmul(outputs, softmax_w) + softmax_b

    def loss_compute(self):
        self._loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.target_inputs, [-1]), logits=self._y_pred))

    def train_operater(self):
        # ***** 优化求解 *******
        tvars = tf.trainable_variables()  # 获取模型的所有参数
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), self.max_grad_norm)  # 获取损失函数对于每个参数的梯度
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)  # 优化器

        # 梯度下降计算
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())

    def acc(self):
        correct_prediction = tf.equal(tf.cast(tf.arg_max(self._y_pred, 1), tf.int32), tf.reshape(self.target_inputs, [-1]))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def accuracy(self):
        return self._accuracy

    print 'Finished creating the bi-lstm model.'
