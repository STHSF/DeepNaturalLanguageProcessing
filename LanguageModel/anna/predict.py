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
print(vocab)
# # 字符数字映射
# vocab_to_int = {c: i for i, c in enumerate(vocab)}
#
# # print(vocab_to_int)
# # 数字字符映射
# int_to_vocab = dict(enumerate(vocab))
# # print(int_to_vocab)
#
# # 对文本进行编码
# encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
# print(text[:100])
# print('encode\n', encode[:100])
