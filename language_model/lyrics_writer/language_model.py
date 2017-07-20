#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: language_model.py
@time: 2017/7/20 上午8:41
"""

import numpy as np

file_path = './data/split.txt'

with open(file_path) as f:
    text = f.read()
data = text.split()
# print(u'词的个数：', len(data))
print(format(data[:10]))

# 使用set对列表去重，并保持列表原来的顺序
vocab = list(set(data))
vocab.sort(key=data.index)

# print('不同词的个数',len(text))
# print(text)

vocab_to_id = {char: id for id, char in enumerate(vocab)}
# print(vocab_to_id)

id_to_vocab = dict(enumerate(vocab))

encoded = np.array([vocab_to_id[c] for c in data])

print(encoded)
