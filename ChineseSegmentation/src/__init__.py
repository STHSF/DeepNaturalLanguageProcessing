# coding=utf-8


import pickle
import time

import tensorflow as tf
from sklearn.model_selection import train_test_split

from ChineseSegmentation.src.batch_generate import BatchGenerator

# 数据导入
with open('data.pkl', 'rb') as pk:
    X = pickle.load(pk)
    y = pickle.load(pk)
    word2id = pickle.load(pk)
    id2word = pickle.load(pk)
    tag2id = pickle.load(pk)
    id2tag = pickle.load(pk)

# 划分训练集、测试集、和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print 'X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

print 'Creating the data generator ...'
data_train = BatchGenerator(X_train, y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
print 'Finished creating the data generator.'

'''
For Chinese word ChineseSegmentation.
'''
# ##################### config ######################
decay = 0.85
max_epoch = 5
hidden_units = 128
timestep_size = max_len = 32  # 句子长度
vocab_size = 5159  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64  # 字向量长度
num_classes = 5
hidden_size = 128  # 隐含层节点数
layers_num = 2  # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置


# source_inputs = tf.placeholder(shape=(None, None), dtype=tf.float32, name='source_inputs')
# target_inputs = tf.placeholder(shape=(None, None), dtype=tf.float32, name='y_inputs')


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


source_inputs = tf.placeholder(shape=(None, timestep_size), dtype=tf.int32, name='source_inputs')
target_inputs = tf.placeholder(shape=(None, timestep_size), dtype=tf.int32, name='y_inputs')

embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
# shape = [batch_size, num_steps, embedding_size]
inputs_embedded = tf.nn.embedding_lookup(embedding, source_inputs)


def lstm_cell():
    cell = tf.contrib.rnn.LSTMCell(hidden_units)
    return cell


cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layers_num)], state_is_tuple=True)
cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layers_num)], state_is_tuple=True)

initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

((outputs_fw,
  outputs_bw),
 (final_state_fw,
  final_state_bw)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                      cell_bw=cell_bw,
                                                      inputs=inputs_embedded,
                                                      initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      dtype=tf.float32,
                                                      time_major=False))

# shape = [batch_size, num_steps, hidden_units * 2]
outputs = tf.concat((outputs_fw, outputs_bw), 2)

# shape = [batch_size * num_steps, hidden_units * 2]
outputs = tf.reshape(outputs, [-1, hidden_units * 2])

# final_state_c = tf.concat((final_state_fw.c, final_state_bw.c), 1)
# final_state_h = tf.concat((final_state_fw.h, final_state_bw.h), 1)
# final_state = tf.contrib.rnn.LSTMStateTuple(c=final_state_c,
#                                             h=final_state_h)

with tf.variable_scope('outputs'):
    softmax_w = weight_variable([hidden_units * 2, num_classes])
    softmax_b = bias_variable([num_classes])

    # shape = [batch_size * num_steps, hidden_units * 2]
    y_pred = tf.matmul(outputs, softmax_w) + softmax_b

correct_prediction = tf.equal(tf.cast(tf.arg_max(y_pred, 1), tf.int32), tf.reshape(target_inputs, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(target_inputs, [-1]), logits=y_pred))

# ***** 优化求解 *******
tvars = tf.trainable_variables()  # 获取模型的所有参数
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # 优化器

# 梯度下降计算
train_op = optimizer.apply_gradients(zip(grads, tvars),
                                     global_step=tf.contrib.framework.get_or_create_global_step())
print 'Finished creating the bi-lstm model.'


def test_epoch(dataset):
    """Testing or valid."""
    _batch_size = 500
    fetches = [accuracy, loss]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    for i in xrange(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {source_inputs: X_batch,
                     target_inputs: y_batch,
                     lr: 1e-5,
                     batch_size: _batch_size,
                     keep_prob: 1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

tr_batch_size = 128
max_max_epoch = 6
display_num = 5  # 每个 epoch 显示是个结果

tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
saver = tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量
for epoch in xrange(max_max_epoch):
    _lr = 1e-4
    if epoch > max_epoch:
        _lr = _lr * (decay ** (epoch - max_epoch))
    print 'EPOCH %d， lr=%g' % (epoch + 1, _lr)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0
    for batch in xrange(tr_batch_num):
        fetches = [accuracy, loss, train_op]
        X_batch, y_batch = data_train.next_batch(tr_batch_size)
        feed_dict = {source_inputs: X_batch,
                     target_inputs: y_batch,
                     lr: _lr,
                     batch_size: tr_batch_size,
                     keep_prob: 0.5}
        _acc, _cost, _ = sess.run(fetches, feed_dict)  # the cost is the mean cost of one batch
        _accs += _acc
        _costs += _cost
        show_accs += _acc
        show_costs += _cost
        if (batch + 1) % display_batch == 0:
            valid_acc, valid_cost = test_epoch(data_valid)  # valid
            print '\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                                            show_costs / display_batch, valid_acc,
                                                                            valid_cost)
            show_accs = 0.0
            show_costs = 0.0
    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num
    if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
        save_path = saver.save(sess, model_save_path, global_step=(epoch + 1))
        print 'the save path is ', save_path
    print '\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost)
    print 'Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch'\
          % (data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time)

# testing
print '**TEST RESULT:'
test_acc, test_cost = test_epoch(data_test)
print '**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost)
