#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""生成词向量空间"""

# import modules & set up logging
import gensim
import logging
import os


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


def do():
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentencestest = [['中国', '人'], ['美国', '人']]
    # # train word2vec on the two sentences
    # model = gensim.models.Word2Vec(sentences, min_count=1)
    #
    # print model["中国"]

    #  a memory-friendly iterator
    # sentences = MySentences('/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data')
    sentences = MySentences(
        '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data')  # a memory-friendly iterator

    count = 1
    for sen in sentences:
        print count, " ".join(sen)
        count += 1

    # train
    model = gensim.models.Word2Vec(sentences, min_count=2, size=200, workers=4)

    # 测试输出
    print(unicode(model["纤维"]))

    dd = model.most_similar("纤维")
    for i in dd:
        for j in i:
            print j,

    # 模型保存
    # model.save('/tmp/mymodel')

    # 模型加载
    # new_model = gensim.models.Word2Vec.load('/tmp/mymodel')

if __name__ == '__main__':
    do()

