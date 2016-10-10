#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""生成文本向量空间"""

# import modules & set up logging
from sklearn.preprocessing import scale
from gensim.models import Word2Vec
import numpy as np
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# 将一篇文章中的词向量的平均值作为输入文本的向量
def build_word2vec(text, size, word2vec_model):
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


def text_vecs(x_train, x_test, n_dim, word2vec_model):

    # 训练集文本向量
    train_vecs = np.concatenate([build_word2vec(z, n_dim, word2vec_model) for z in x_train])
    train_vecs = scale(train_vecs)
    # 测试集处理
    word2vec_model.train(x_test)
    test_vecs = np.concatenate([build_word2vec(z, n_dim, word2vec_model) for z in x_test])
    test_vecs = scale(test_vecs)

    res = (train_vecs, test_vecs)
    return res


def model_load_test():

    model_path = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/word2vecmodel/mymodel'
    w2c_model = Word2Vec.load(model_path)
    # model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
    # #using gzipped/bz2 input works too, no need to unzip:
    # model = Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)
    # res = w2c_model.most_similar(positive=['纤维', '批次'], negative=['成分'], topn=1)
    #
    # w2c_model.doesnt_match("我 爱 中国".split())
    #
    # var = w2c_model.similarity('纤维', '批次')
    # print var
    res = w2c_model.most_similar("批次")
    for i in res:
        print i[0],

if __name__ == "__main__":
    model_load_test()
