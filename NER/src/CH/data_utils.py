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


datas = list()
labels = list()
print('building word index...')
with open(src_file) as source:
    a = []
    for line in source.readlines():
        line.strip()
        if line != '':
            values = line.split()
            datas.append(values)
            a.extend(values)
    word = list(set(a))
    set_ids = range(1, len(word)+1)
    print(set_ids)

    word2id = pd.Series(set_ids, index=word)
    id2word = pd.Series(word, index=set_ids)
    print(word2id.head())
    print(id2word.head())

print('building tag index...')
with open(tgt_file, 'r') as source:
    list_word = []
    for line in source.readlines():
        line = line.strip()
        if line != '':
            word_arr = line.split()
            labels.append(word_arr)
            list_word.extend(word_arr)
    tags = list(set(list_word))
    tags.insert(0, 'Padding')
    print(tags)
    tag_ids = range(len(tags))
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)

    print(tag2id.head(100))
    print(id2tag.head(100))

df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
# 句子长度
df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
print df_data.head(10)


def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids)))  # 短则补全
    return ids


def y_padding(tags):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    ids = list(tag2id[tags])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids)))  # 短则补全
    return ids


df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)
print df_data.head(10)

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

