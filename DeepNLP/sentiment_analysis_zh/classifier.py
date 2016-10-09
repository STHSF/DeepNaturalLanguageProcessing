# coding=utf-8
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import data_processing
# 读入数据
pos_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
neg_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'

tmp = data_processing.read_data(pos_file_path, neg_file_path)
res = data_processing.data_split(tmp[0], tmp[1])
x_train = res[0]
x_test = res[1]
label_train = res[2]
label_test = res[3]



# 分类训练
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, label_train)

print('Test Accuracy: %.2f' % lr.score(test_vecs, label_test))

pred_probas = lr.predict_proba(test_vecs)[:, 1]

fpr, tpr, _ = roc_curve(label_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' %roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')

plt.show()