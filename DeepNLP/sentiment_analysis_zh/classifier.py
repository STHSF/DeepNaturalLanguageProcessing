# coding=utf-8
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

train_vecs = ''
label_train = ''
test_vecs = ''
label_test = ''

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