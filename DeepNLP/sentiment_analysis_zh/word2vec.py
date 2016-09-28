#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""生成词向量空间"""

# import modules & set up logging
import gensim
import logging
import os

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['中国', '人'], ['美国', '人']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)

print model["中国"]



# class MySentences(object):
#     def __init__(self, dir_name):
#         self.dir_name = dir_name
#
#     def __iter__(self):
#         for file_name in os.listdir(self.dir_name):
#             for line in open(os.path.join(self.dir_name, file_name)):
#                 yield line.split(",")
#
#
# sentences = MySentences('/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data')  # a memory-friendly iterator
# model = gensim.models.Word2Vec(sentences, min_count=2, workers=4)



# model.save('/tmp/mymodel')
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')