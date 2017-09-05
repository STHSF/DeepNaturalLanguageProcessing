# coding=utf-8
import pickle
import numpy as np
import pandas as pd
import re
import time
from itertools import chain

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

with open("msr_train.txt") as f:
    texts = f.read().decode('gbk')
sentences = texts.split('\r\n')

print len(sentences)
print sentences[:10]





print 'Length of texts is %d' % len(texts)
print 'Example of texts: \n', texts[:2]

# 重新以标点来划分
sentences = re.split(u'[，。！？、‘’“”]/[bems]', texts)
print 'Sentences number:', len(sentences)
print 'Sentence Example:\n', sentences[0]
#
# def get_Xy(sentence):
#     """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
#     words_tags = re.findall('(.)/(.)', sentence)
#     if words_tags:
#         words_tags = np.asarray(words_tags)
#         words = words_tags[:, 0]
#         tags = words_tags[:, 1]
#         return words, tags  # 所有的字和tag分别存为 data / label
#     return None
#
# datas = list()
# labels = list()
# print 'Start creating words and tags data ...'
# for sentence in iter(sentences):
#     result = get_Xy(sentence)
#     if result:
#         datas.append(result[0])
#         labels.append(result[1])
#
# print 'Length of datas is %d' % len(datas)
# print 'Example of datas: ', datas[0]
# print 'Example of labels:', labels[0]
#
#
# df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
# # 句子长度
# df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
# print df_data.head(2)
#
# pkl_file1 = open('tag_to_id.pkl', 'rb')
# pkl_file2 = open('word_to_id.pkl', 'rb')
#
# tag2id = pickle.load(pkl_file1)
# word2id = pickle.load(pkl_file2)
#
# max_len = 32
#
#
# def X_padding(words):
#     """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
#     ids = list(word2id[words])
#     if len(ids) >= max_len:  # 长则弃掉
#         return ids[:max_len]
#     ids.extend([0]*(max_len-len(ids))) # 短则补全
#     return ids
#
#
# def y_padding(tags):
#     """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
#     ids = list(tag2id[tags])
#     if len(ids) >= max_len:  # 长则弃掉
#         return ids[:max_len]
#     ids.extend([0]*(max_len-len(ids)))  # 短则补全
#     return ids
#
# df_data['X'] = df_data['words'].apply(X_padding)
# df_data['y'] = df_data['tags'].apply(y_padding)
# print df_data.head(2)
#
# # 最后得到了所有的数据
# X = np.asarray(list(df_data['X'].values))
# y = np.asarray(list(df_data['y'].values))
# print 'X.shape={}, y.shape={}'.format(X.shape, y.shape)
# print 'Example of words: ', df_data['words'].values[0]
# print 'Example of X: ', X[0]
# print 'Example of tags: ', df_data['tags'].values[0]
# print 'Example of y: ', y[0]