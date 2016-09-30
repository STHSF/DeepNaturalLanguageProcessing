# coding=utf-8

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from gensim.models.word2vec import Word2Vec
import numpy as np
import matplotlib.pyplot as plt


# 读入数据
with open("path", "r") as input_file:
    pos_file = input_file.readlines()

with open('/path', 'r') as input_file:
    neg_file = input_file.readlines()

# 标签
label = np.concatenate((np.ones(len(pos_file)), np.zeros(len(neg_file))))

# 训练集,测试集
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_file, neg_file)), label, test_size=0.2)


# def text_clean(corpus):
#     corpus = [z.lower().replace('\n', " ").split() for z in corpus]
#     return corpus
#
# x_train = text_clean(x_train)
# x_test = text_clean(x_test)

n_dim = 300
imdb_w2c = Word2Vec(n_dim, min_count=100)
imdb_w2c.build_vocab(x_train)

imdb_w2c.train(x_train)

imdb_w2c.save('/tmp/mymodel')






