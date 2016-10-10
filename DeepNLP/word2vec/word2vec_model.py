# coding=utf-8

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from gensim.models.word2vec import Word2Vec
import numpy as np

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
#
word2vec_model.build_vocab(x_train)

word2vec_model.train(x_train)

word2vec_model.save('/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/word2vecmodel/mymodel')






