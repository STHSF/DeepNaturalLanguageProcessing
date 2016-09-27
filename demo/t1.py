#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np


"""单层神经网络"""
# 导入随机定义训练的数据x和y
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 定义参数weight biases 拟合公式y 误差公式loss
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y - y_data))

# 选择gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.5)

# 神经网络的key idea
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 激活结构
sess = tf.Session()

# 激活
sess.run(init)

# 训练
for step in range(100):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
