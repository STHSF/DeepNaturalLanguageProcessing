#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: predict.py
@time: 2017/7/3 下午2:26
"""
import time
from collections import namedtuple
import numpy as np
import tensorflow as tf


import time
from collections import namedtuple
import numpy as np
import tensorflow as tf

file_path = './data/anna.txt'
with open(file_path, 'r') as f:
    text = f.read()

vocab = set(text)
# print(vocab)
# 字符数字映射
vocab_to_int = {c: i for i, c in enumerate(vocab)}
# print(vocab_to_int)
# 数字字符映射
int_to_vocab = dict(enumerate(vocab))
# print(int_to_vocab)

# 对文本进行编码
encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
print(text[:100])
print('encode\n', encode[:100])


def generate_bath(arr, batch_size, seq_length):
    num_steps = batch_size * seq_length
    # print('num_steps', num_steps)
    n_iters = int(len(arr) / num_steps)
    # print('num_iters', n_iters)
    arr = arr[: num_steps * n_iters]
    # print('arr_b', arr[:,:15])
    # 重塑
    arr = arr.reshape((batch_size, -1))
    # print('arr_a\n',arr[:,:15])
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


def get_batch(raw_data, batch_size, seq_length):
    """
    生成batch数据，
    Args:
        array:
        batch_size:
        seq_length:

    Returns:

    """
    data = np.array(raw_data)
    data_length = data.shape[0]
    num_batches = (data_length - 1) // (batch_size * seq_length)
    assert num_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = num_batches * (batch_size * seq_length)
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, num_batches * seq_length])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, num_batches * seq_length])

    for batch in range(num_batches):
        x = xdata[:, batch * seq_length:(batch + 1)*seq_length]
        y = ydata[:, batch * seq_length:(batch + 1)*seq_length]
        yield x, y


batches = generate_bath(encode, 10, 50)
x, y = next(batches)
print('2inputs.shape', np.shape(x))
print('2inputs', x[:10, :10])
print('2y.shape', np.shape(y))
print('2y', y[:10, :10])
