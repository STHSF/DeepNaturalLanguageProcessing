#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""生成文本向量空间"""

# import modules & set up logging
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
import logging
import globe

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
    train_vecs = scale(train_vecs)  # 归一化
    # 测试集处理
    word2vec_model.train(x_test)
    test_vecs = np.concatenate([build_word2vec(z, n_dim, word2vec_model) for z in x_test])
    test_vecs = scale(test_vecs)

    res = (train_vecs, test_vecs)
    return res


# 训练集转向量空间模型
def text_vecs_zx():
    w2v_model = Word2Vec.load(globe.model_path)
    train_data = globe.train_data
    doc_vec = []
    for d in train_data:
        label = d[0]
        data = open(d[1])
        for doc in data:
            word = doc.split(",")
            doc_vec_temp = build_word2vec(word, globe.n_dim, w2v_model)
            doc_vec_temp = scale(doc_vec_temp)  # 归一化
            doc_vec.append((label, doc_vec_temp))
    return doc_vec


# 文档转向量空间模型
# def doc_vecs_zx(file_seg, word2vec_model):
#     file_vec = {}
#     for key_title in file_seg.keys:
#         doc = file_seg[key_title]
#         word = doc.split(",")
#         doc_vec = build_word2vec(word, globe.n_dim, word2vec_model)
#         doc_vec = scale(doc_vec)  # 归一化
#         file_vec[key_title] = doc_vec
#     return file_vec

# 文档转向量空间模型
def doc_vecs_zx(doc, word2vec_model):
    word = doc.split(",")
    doc_vec = build_word2vec(word, globe.n_dim, word2vec_model)
    # print doc_vec

    # doc_vec = scale(doc_vec)  # 归一化 ，，如果进行归一化，词向量全部变为 0，诡异！！
    # print doc_vec

    return doc_vec


def model_load_test():
    model_path = globe.model_path
    w2c_model = Word2Vec.load(model_path)

    print '[中国] ', " ".join([word[0] for word in w2c_model.most_similar("中国")])
    print '[万科] ', " ".join([word[0] for word in w2c_model.most_similar("万科")])
    print '[猪肉] ', " ".join([word[0] for word in w2c_model.most_similar("猪肉")])
    print '[股市] ', " ".join([word[0] for word in w2c_model.most_similar("股市")])
    print '[涨] ', " ".join([word[0] for word in w2c_model.most_similar("涨")])
    print '[地产] ', " ".join([word[0] for word in w2c_model.most_similar("地产")])
    print '[基金] ', " ".join([word[0] for word in w2c_model.most_similar("基金")])


if __name__ == "__main__":
    model_load_test()

    # doc_vec = text_vecs_zx()
    # count = 1
    # for d in doc_vec:
    #     print count, d[0], len(d[1])
    #     count += 1
