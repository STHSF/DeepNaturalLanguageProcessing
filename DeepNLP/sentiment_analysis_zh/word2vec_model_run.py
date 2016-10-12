# coding=utf-8
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import globe
import data_processing
import word2vec_gensim_model
import word2vec_gensim_train


# 读入数据
# pos_file_path = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
# neg_file_path = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'
pos_file_path = '/Users/li/workshop/DataSet/sentiment/train/result_pos.txt'
neg_file_path = '/Users/li/workshop/DataSet/sentiment/train/result_neg.txt'

tmp = data_processing.read_data(pos_file_path, neg_file_path)
res = data_processing.data_split(tmp[0], tmp[1])
x_train = res[0]
x_test = res[1]
label_train = res[2]
label_test = res[3]
x_train = data_processing.text_clean(x_train)
x_test = data_processing.text_clean(x_test)

# 生成文本向量
n_dim = globe.n_dim
# model_path = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/word2vecmodel/mymodel'
model_path = globe.model_path

word2vec_model = Word2Vec.load(model_path)
vecs = word2vec_gensim_train.text_vecs(x_train, x_test, n_dim, word2vec_model)
train_vecs = vecs[0]
test_vecs = vecs[1]


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

