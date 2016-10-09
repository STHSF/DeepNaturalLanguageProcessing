#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""生成词向量空间"""

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sentencestest = [['中国', '人'], ['美国', '人']]
# # train word2vec on the two sentences
# model = gensim.models.Word2Vec(sentences, min_count=1)
#
# print model["中国"]


class MySentences1(object):

    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.do()

    def do(self):
        res = []
        for file_name in os.listdir(self.dir_name):
            for line in open(os.path.join(self.dir_name, file_name)).readlines():
                res.append(line.strip())
        return res


# 读取文件夹中的所有数据
class MySentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for file_name in os.listdir(self.dir_name):
            for line in open(os.path.join(self.dir_name, file_name)):
                yield line.split(",")

#  a memory-friendly iterator
sentences = MySentences('/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data')
# sentences = MySentences('/Users/li/Kunyan/DataSet/trainingSets')  # a memory-friendly iterator


# 读入数据
pos_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
neg_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'

with open(pos_file_path, "r") as input_file:
    pos_file = input_file.readlines()

with open(neg_file_path, 'r') as input_file:
    neg_file = input_file.readlines()

# 标签
label = np.concatenate((np.ones(len(pos_file)), np.zeros(len(neg_file))))

# 训练集,测试集
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_file, neg_file)), label, test_size=0)


def text_clean(corpus):
    corpus = [z.lower().replace('\n', " ").split(",") for z in corpus]
    return corpus

x_train = text_clean(x_train)
# x_test = text_clean(x_test)

n_dim = 200
min_count = 2
word2vec_model = Word2Vec(size=n_dim, min_count=min_count, workers=4)

word2vec_model.build_vocab(x_train)

word2vec_model.train(x_train)

word2vec_model.save('/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/word2vecmodel/mymodel')

print(unicode(word2vec_model["纤维"]))
dd = word2vec_model.most_similar("纤维")
for i in dd:
    print i,

