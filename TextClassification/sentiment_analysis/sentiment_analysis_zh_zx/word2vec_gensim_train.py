#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""生成词向量空间"""

# import modules & set up logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from gensim.models import Word2Vec
import numpy as np
import logging
import os

x_train = ''
x_test = ''
n_dim = 300

word2vec_model = Word2Vec.load('/tmp/mymodel')

# model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
# #using gzipped/bz2 input works too, no need to unzip:
# model = Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)

# word2vec_model.most_similar(positive=['中国', '美国'], negative=['中国人'], topn=1)
#
# word2vec_model.doesnt_match("我爱中国".split())
#
# word2vec_model.similarity('男人', '女人')


# 将一篇文章中的词向量的平均值作为输入文本的向量
def build_word2vec(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += word2vec_model[word].reshape(1, size)
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

# 训练集文本向量
train_vecs = np.concatenate([build_word2vec(z, n_dim) for z in x_train])
train_vecs = scale(train_vecs)

# 测试集处理

word2vec_model.train(x_test)

test_vecs = np.concatenate([build_word2vec(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)