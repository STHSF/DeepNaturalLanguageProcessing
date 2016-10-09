#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""生成词向量空间"""

# import modules & set up logging
from gensim.models.word2vec import Word2Vec
import logging
import os

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sentencestest = [['中国', '人'], ['美国', '人']]
# model = gensim.models.Word2Vec(sentences, min_count=1)


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


class MySentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for file_name in os.listdir(self.dir_name):
            for line in open(os.path.join(self.dir_name, file_name)):
                yield line.split(",")

#  a memory-friendly iterator
sentences = MySentences('/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data')
# sentences = MySentences('/Users/li/Kunyan/DataSet/trainingSets')
# model = Word2Vec(sentences, min_count=2, workers=4)

# print model["保险"]
# dd = model.most_similar("中国")

for words in sentences:
    for word in words:
        print word,



# model.save('/tmp/mymodel')
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')