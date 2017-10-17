# coding=utf-8
"""
Training model
"""
import tensorflow as tf
from config import Config
from NER_Model import bi_lstm_crf

import time
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from batch_generate import BatchGenerator

# 数据导入
with open('data.pkl', 'rb') as pk:
    X = pickle.load(pk)
    y = pickle.load(pk)
    word2id = pickle.load(pk)
    id2word = pickle.load(pk)
    tag2id = pickle.load(pk)
    id2tag = pickle.load(pk)

# 划分训练集、测试集、和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))

print('Creating the data generator ...')
data_train = BatchGenerator(X_train, y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
print('Finished creating the data generator.')

model = bi_lstm_crf(Config)


decay = 0.85
tr_batch_size = 128
max_epoch = 1
max_max_epoch = 1
display_num = 5  # 每个 epoch 显示是个结果
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置

tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次

print('tr_batch_num', tr_batch_num)
print('display_batch', display_batch)

