#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""生成词向量空间"""

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import os

import sys

reload(sys)
sys.setdefaultencoding('utf8')


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# sentencestest = [['中国', '人'], ['美国', '人']]
# # train word2vec on the two sentences
# model = gensim.models.Word2Vec(sentences, min_count=1)
#
# print model["中国"]


# class MySentences1(object):
#
#     def __init__(self, dir_name):
#         self.dir_name = dir_name
#         self.do()
#
#     def do(self):
#         res = []
#         for file_name in os.listdir(self.dir_name):
#             for line in open(os.path.join(self.dir_name, file_name)).readlines():
#                 res.append(line.strip())
#         return res

# #  a memory-friendly iterator
# sentences = MySentences('/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data')
# sentences = MySentences('/Users/li/Kunyan/DataSet/trainingSets')  # a memory-friendly iterator

# 读取文件夹中的所有数据
class MySentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for line in open(self.dir_name):
            yield line.split(",")


# 按照标签读取数据
def read_data(pos_file_path, neg_file_path):
    with open(pos_file_path) as input_file:
        pos_file = input_file.readlines()
        tmp = []
        for i in pos_file:
            tmp.append(i.split(","))

    with open(neg_file_path) as input_file:
        neg_file = input_file.readlines()
        tmp = []
        for i in pos_file:
            tmp.append(i.split(","))

    res = (pos_file, neg_file)
    return res


# 数据预处理,设置标签,训练集测试集准备
def data_split(pos_file, neg_file):
    # 标签
    label = np.concatenate((np.ones(len(pos_file)), np.zeros(len(neg_file))))

    # 训练集,测试集
    train_data, test_data, train_labels, test_labels = train_test_split(np.concatenate((pos_file, neg_file)), label,
                                                                        test_size=0.5)

    res = (train_data, test_data, train_labels, test_labels)
    return res


def text_clean(corpus):
    corpus = [z.lower().replace('\n', ' ').split(',') for z in corpus]
    return corpus


# 测试
def do():
    sen = MySentences("/home/zhangxin/work/DeepSentiment/data/tagging/result.txt")
    count = 1
    for s in sen:
        print count, " ".join(s)
        count += 1


if __name__ == "__main__":
    do()
