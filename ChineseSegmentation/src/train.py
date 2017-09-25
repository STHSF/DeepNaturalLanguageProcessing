# coding=utf-8
import time
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from batch_generate import BatchGenerator
from bi_lstm_advanced import bi_lstm

# 数据导入
with open('data.pkl', 'rb') as pk:
    X = pickle.load(pk)
    y = pickle.load(pk)
    word2id = pickle.load(pk)
    id2word = pickle.load(pk)
    tag2id = pickle.load(pk)
    id2tag = pickle.load(pk)
    zy = pickle.load(pk)

# 划分训练集、测试集、和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

print('Creating the data generator ...')
data_train = BatchGenerator(X_train, y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
print('Finished creating the data generator.')

# bi_lstm模型
model = bi_lstm()
# 设置cpu按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()
# all_vars = tf.trainable_variables()
# saver = tf.train.Saver(all_vars)  # 最多保存的模型数量
# summary_writer = tf.train.SummaryWriter('/tmp/tensorflowlogs')
saver = tf.train.Saver()  # 最多保存的模型数量

decay = 0.85
tr_batch_size = 128
max_epoch = 1
max_max_epoch = 1
display_num = 5  # 每个 epoch 显示是个结果
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置

tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次


def run_epoch(dataset):
    """Testing or valid."""
    _batch_size = 500
    fetches = [model.accuracy, model.loss]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    _costs = 0.0
    _accs = 0.0
    for i in range(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {model.source_inputs: X_batch,
                     model.target_inputs: y_batch,
                     model.lr: 1e-5,
                     model.batch_size: _batch_size,
                     model.keep_prob: 1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost


with tf.Session(config=config) as sess:
    sess.run(init)
    for epoch in range(max_max_epoch):
        _lr = 1e-4
        if epoch > max_epoch:
            _lr = _lr * (decay ** (epoch - max_epoch))
        print('EPOCH %d， lr=%g' % (epoch + 1, _lr))
        start_time = time.time()
        _costs = 0.0
        _accs = 0.0
        show_accs = 0.0
        show_costs = 0.0
        for batch in range(tr_batch_num):
            fetches = [model.accuracy, model.loss, model.train_op]
            X_batch, y_batch = data_train.next_batch(tr_batch_size)
            feed_dict = {model.source_inputs: X_batch,
                         model.target_inputs: y_batch,
                         model.lr: _lr,
                         model.batch_size: tr_batch_size,
                         model.keep_prob: 0.5}
            _acc, _cost, _ = sess.run(fetches, feed_dict)  # the cost is the mean cost of one batch
            _accs += _acc
            _costs += _cost
            show_accs += _acc
            show_costs += _cost
            if (batch + 1) % display_batch == 0:
                valid_acc, valid_cost = run_epoch(data_valid)  # valid
                print('\t training acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                                                 show_costs / display_batch, valid_acc,
                                                                                 valid_cost))
                show_accs = 0.0
                show_costs = 0.0
        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num
        if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
            save_path = model.saver.save(sess, model_save_path, global_step=(epoch + 1))
            print('the save path is ', save_path)
        print('\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
        print('Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' \
              % (data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time))

    # testing
    print('**TEST RESULT:')
    test_acc, test_cost = run_epoch(data_test)
    print('**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost))
