#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: __init__.py.py
@time: 2017/7/5 下午2:47
"""

import numpy as np

batch_size = 4
seq_length = 3
raw_data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]


def get_batch2(raw_data, batch_size, seq_length):
    data = np.array(raw_data)
    data_length = data.shape[0]
    print('data_length', data_length)
    iterations = (data_length - 1) // (batch_size * seq_length)
    print('iterations',iterations)
    round_data_len = iterations * batch_size * seq_length
    print('round_data_len',round_data_len)
    xdata = data[:round_data_len].reshape(batch_size, iterations*seq_length)
    print(xdata)
    ydata = data[1:round_data_len+1].reshape(batch_size, iterations*seq_length)
    print(ydata)

    for i in range(iterations):
        x = xdata[:, i*seq_length:(i+1)*seq_length]
        y = ydata[:, i*seq_length:(i+1)*seq_length]
        yield x, y


def get_batch(raw_data, batch_size, seq_length):
    data = np.array(raw_data)
    data_length = data.shape[0]
    num_steps = data_length - seq_length + 1
    print('num_steps', num_steps)
    iterations = num_steps // batch_size
    print('iterations', iterations)
    xdata=[]
    ydata=[]
    for i in range(num_steps-1):
        xdata.append(data[i:i+seq_length])
        ydata.append(data[i+1:i+1+seq_length])

    for batch in range(iterations):
        x = np.array(xdata)[batch * batch_size: batch * batch_size + batch_size, :]
        y = np.array(xdata)[batch * batch_size + 1: batch * batch_size + 1 + batch_size, :]
        yield x, y


if __name__ == '__main__':
    get_batch2(raw_data, batch_size, seq_length)
    for x, y in get_batch2(raw_data, batch_size, seq_length):
        print(x)
        print(y)



