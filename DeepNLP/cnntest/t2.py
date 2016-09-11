# coding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

plt.figure(1)
plt.plot(x_data, y_data, "r.")
# plt.show()


def add_layers(inputs, in_size, out_size, layer_name, keep_prob, activation_function=None):

    # add one more layer and return the output of this layer
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases

    # here to dropout
    # 在 wx_plus_b 上drop掉一定比例
    # keep_prob 保持多少不被drop，在迭代时在 sess.run 中 feed
    wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)

    tf.histogram_summary(layer_name + '/outputs', outputs)

    return outputs


def add_layer(inputs, in_size, out_size, activation_function=None):

    # add one more layer and return the output of this layer
    # 区别：大框架，定义层 layer，里面有 小部件
    with tf.name_scope('layer'):
        # 区别：小部件
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)

            # here to dropout, 在 wx_plus_b 上drop掉一定比例, keep_prob 保持多少不被drop，在迭代时在 sess.run 中 feed
            wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob=1)

        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b, )

        return outputs

# define placeholder for inputs to network
# 区别：大框架，里面有 inputs x，y
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
layer1 = add_layer(xs, 1, 3, activation_function=tf.nn.relu)
layer2 = add_layer(layer1, 3, 5, activation_function=tf.nn.softplus)

# add output layer
prediction1 = add_layer(layer2, 5, 2, activation_function=tf.nn.relu)
prediction = add_layer(prediction1, 2, 1, activation_function=tf.nn.softplus)

# the error between prediction and real data
# 区别：定义框架 loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# 区别：定义框架 train
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 分类问题的loss函数cross entropy
# loss 函数用 cross entropy
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# session对象
sess = tf.Session()
init = tf.initialize_all_variables()

# 区别：sess.graph 把所有框架加载到一个文件中放到文件夹"logs/"里
# 接着打开terminal，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir='logs/'
# 会返回一个地址，然后用浏览器打开这个地址，在 graph 标签栏下打开
writer = tf.train.SummaryWriter("logs/", sess.graph)

# important step
sess.run(init)

# 训练
for step in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if step % 20 == 0:
        sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

# Save to file
# remember to define the same dtype and shape when restore
#
# saver = tf.train.Saver()
#
# 用 saver 将所有的 variable 保存到定义的路径
# with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess, "my_net/save_net.ckpt")
#    print("Save to path: ", save_path)


# ################################################
#
# # restore variables
# # redefine the same shape and same type for your variables
# W1 = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# b1 = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
#
# # not need init step
# saver = tf.train.Saver()
# # 用 saver 从路径中将 save_net.ckpt 保存的 W 和 b restore 进来
# with tf.Session() as sess:
#     saver.restore(sess, "my_net/save_net.ckpt")
#     print("weights:", sess.run(W1))
#     print("biases:", sess.run(b1))