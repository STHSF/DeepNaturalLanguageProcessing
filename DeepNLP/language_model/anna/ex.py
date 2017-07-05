#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: ex.py
@time: 2017/7/4 上午11:11
"""
import time
from collections import namedtuple
import numpy as np
import tensorflow as tf

file_path = './data/anna.txt'
with open(file_path, 'r') as f:
    text = f.read()
# 字符集合
vocab = set(text)
# 字符数字映射
vocab_to_int = {char:i for i, char in enumerate(vocab)}
# 数字字符映射
int_to_vocab = dict(enumerate(vocab))

# 对文本进行编码
encode = np.array([vocab_to_int[char] for char in text], dtype=np.int32)
print(text[:100])
print('encode\n', encode[:100])


def generate_bath(arr, batch_size, seq_length):
    """

    Args:
        arr: input array
        batch_size:
        seq_length:

    Returns:

    """
    num_steps = batch_size * seq_length
    # print('num_steps', num_steps)
    n_iters = int(len(arr) / num_steps)
    # print('num_iters', n_iters)
    arr = arr[: num_steps * n_iters]
    # print('arr_b', arr[:,:15])
    # 重塑
    arr = arr.reshape((batch_size, -1))
    print('arr_a\n',arr[:,:15])
    # print(arr.shape[1])

    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n + seq_length]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

batches = generate_bath(encode, 10, 50)
x, y = next(batches)
print('inputs.shape', np.shape(x))
print('inputs', x[:10, :10])
print('y.shape', np.shape(y))
print('y', y[:10, :10])

aa = tf.one_hot(x, 10)

print('aa.shape', np.shape(aa))
print('aa', aa)