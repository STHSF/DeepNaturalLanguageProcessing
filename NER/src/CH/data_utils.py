#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import config


src_file = config.FLAGS.src_file
tgt_file = config.FLAGS.tgt_file
# 只有在预测结果时使用。
pred_file = config.FLAGS.pred_file
src_vocab_file = config.FLAGS.src_vocab_file
tgt_vocab_file = config.FLAGS.tgt_vocab_file
max_len = config.FLAGS.max_sequence

print('building word index...')
datas = list()
labels = list()
word_list = []
with open(src_file) as source:
    for line in source.readlines():
        line.strip()
        if line != '':
            values = line.split()
            datas.append(values)
            word_list.extend(values)
# predict file 中含有training file中没有碰到的词，所以在构建word2id的时候需要将predict中国的词也放进去。
with open(pred_file) as predicts:
    for line in predicts.readlines():
        line.strip()
        if line != '':
            values = line.split()
            word_list.extend(values)


word_set = list(set(word_list))
word_set.insert(0, '$UNK$')  # 添加$UNK$标识符，如果predict中存在训练集中没有的词汇则使用$UNK$表示。
word_set.insert(1, '$PADDING$')  # 添加padding标识符，用于固定长度的字符补全。区别与其他的tags

print('length of word', len(word_set))
set_ids = range(2, len(word_set) + 2)
print(set_ids)

word2id = pd.Series(set_ids, index=word_set)
id2word = pd.Series(word_set, index=set_ids)
print('word2id\n', word2id.head())
print('id2word\n', id2word.head())

print('building tag index...')
tag_list = []
with open(tgt_file, 'r') as source:
    for line in source.readlines():
        line = line.strip()
        if line != '':
            word_arr = line.split()
            labels.append(word_arr)
            tag_list.extend(word_arr)
tags_set = list(set(tag_list))
tags_set.insert(0, 'Padding')  # 添加padding标识符，用于固定长度的字符补全。区别与其他的tags
print('length of tags', len(tags_set))
tag_ids = range(len(tags_set))
tag2id = pd.Series(tag_ids, index=tags_set)
id2tag = pd.Series(tags_set, index=tag_ids)

print('tag2id\n', tag2id.head(100))
print('id2tag\n', id2tag.head(100))

df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
# 句子长度
df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
print('df_data\n', df_data.head(10))


def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids)))  # 短则补全，使用编号0补全
    return ids


def y_padding(tags):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    ids = list(tag2id[tags])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids)))  # 短则补全，id=0的tags为Padding
    return ids


df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)
print('df_data\n', df_data.head(10))

X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['y'].values))
print 'X.shape={}, y.shape={}'.format(X.shape, y.shape)
print 'Example of words: ', df_data['words'].values[0]
print 'Example of X: ', X[0]
print 'Example of tags: ', df_data['tags'].values[0]
print 'Example of y: ', y[0]


# 数据保存成pickle的格式。
with open('data.pkl', 'wb') as outp:
    pickle.dump(X, outp)
    pickle.dump(y, outp)
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
print '** Finished saving the data.'

