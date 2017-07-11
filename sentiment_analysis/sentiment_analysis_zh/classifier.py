# coding=utf-8
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import data_processing
import word2vec_gensim_train as train
import globe
import numpy as np


# 分类流程 liyu
def run_li():
    # 读入数据
    # pos_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
    # neg_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'

    pos_file_path = globe.file_pos
    neg_file_path = globe.file_neg

    tmp = data_processing.read_data(pos_file_path, neg_file_path)
    res = data_processing.data_split(tmp[0], tmp[1])
    train_vecs = res[0]
    test_vecs = res[1]
    label_train = res[2]
    label_test = res[3]

    # 分类训练
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, label_train)

    print('Test Accuracy: %.2f' % lr.score(test_vecs, label_test))

    pred_probas = lr.predict_proba(test_vecs)[:, 1]

    fpr, tpr, _ = roc_curve(label_test, pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')

    plt.show()


def run_zx():
    # 读入数据并处理 train模块
    doc_vec_label = train.text_vecs_zx()

    # 数据分割并设置标签
    doc_vec_label_split = train.train_test(doc_vec_label)
    train_data = doc_vec_label_split[0]
    test_data = doc_vec_label_split[1]

    y_train = np.array([r[0] for r in train_data])
    x_train = np.array([r[1] for r in train_data])

    y_test = np.array([r[0] for r in test_data])
    x_test = np.array([r[1] for r in test_data])

    # 三维变二维，要再研究一下！！！ 2836 * 1 *200
    nsamples, nx, ny = x_train.shape
    d2_train_dataset = x_train.reshape((nsamples, nx * ny))

    nsamples, nx, ny = x_test.shape
    d2_test_dataset = x_test.reshape((nsamples, nx * ny))

    # 分类训练
    # lr = SGDClassifier(loss='log', penalty='l1')
    # lr =LR()   # Logistics
    lr = SVC()
    lr.fit(d2_train_dataset, y_train)

    # 测试精确度
    print('Test Accuracy: %.2f' % lr.score(d2_test_dataset, y_test))

    # 可视化输出
    # pred_probas = lr.predict_proba(d2_test_dataset)[:, 1]
    pred = lr.predict(d2_test_dataset)  # [:, 1]

    fpr, tpr, _ = roc_curve(y_test, pred)  # 标签y只能为 0 或者 1
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')

    plt.show()


if __name__ == "__main__":
    run_zx()
