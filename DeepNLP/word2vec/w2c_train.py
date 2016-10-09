# coding=utf-8

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from gensim.models.word2vec import Word2Vec
import numpy as np

import matplotlib.pyplot as plt


# 读入数据
pos_file_path = ''
neg_file_path = ''
with open(pos_file_path, "r") as input_file:
    pos_file = input_file.readlines()

with open(neg_file_path, 'r') as input_file:
    neg_file = input_file.readlines()

# 标签
label = np.concatenate((np.ones(len(pos_file)), np.zeros(len(neg_file))))

# 训练集,测试集
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_file, neg_file)), label, test_size=0.2)


# load model
n_dim = 300
model_path = '/tmp/mymodel'

new_model = Word2Vec.load(model_path)


# 将一篇文章中的词向量的平均值作为输入文本的向量
def build_word2vec(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += new_model[word].reshape(1, size)
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


train_vecs = np.concatenate([build_word2vec(z, n_dim) for z in x_train])
train_vecs = scale(train_vecs)

# 测试集处理

new_model.train(x_test)

test_vecs = np.concatenate([build_word2vec(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)


# 分类训练
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))


pred_probas = lr.predict_proba(test_vecs)[:, 1]

fpr, tpr, _ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' %roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')

plt.show()