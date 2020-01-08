# coding=utf-8
"""
数据预处理，将训练集转化成需要的格式。
"""
import numpy as np
import pandas as pd
import re
import pickle
from itertools import chain

with open("../dataset/msr_train.txt") as f:
    texts = f.read().decode('gbk')
sentences = texts.split('\r\n')

print(np.shape(sentences))
print(sentences[0])


# 将不规范的内容（如每行的开头）去掉
def clean(s):
    if u'“/s' not in s:  # 句子中间的引号不应去掉
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s


texts = u''.join(map(clean, sentences))  # 把所有的词拼接起来
print 'Length of texts is %d' % len(texts)
print 'Example of texts: \n', texts[:300]

# file_object = open('train_clean.txt', 'w')
# file_object.write(str(texts.decode('utf-8')))
# file_object.close()

# 重新以标点来划分
sentences = re.split(u'[，。！？、‘’“”]/[bems]', texts)
print 'Sentences number:', len(sentences)
print 'Sentence Example:\n', sentences[0]


def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        return words, tags  # 所有的字和tag分别存为 data / label
    return None


datas = list()
labels = list()
print 'Start creating words and tags data ...'
for sentence in iter(sentences):
    result = get_Xy(sentence)
    if result:
        datas.append(result[0])
        labels.append(result[1])

print 'Length of datas is %d' % len(datas)
print 'Example of datas: ', datas[0]
print 'Example of labels:', labels[0]


df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
# 句子长度
df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
print df_data.head(2)


# 1.用 chain(*lists) 函数把多个list拼接起来
all_words = list(chain(*df_data['words'].values))
# 2.统计所有 word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1)  # 注意从1开始，因为我们准备把0作为填充值
print(set_ids)

tags = ['x', 's', 'b', 'm', 'e']
tag_ids = range(len(tags))

# 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

vocab_size = len(set_words)
print 'vocab_size={}'.format(vocab_size)


max_len = 15


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
# # 将word2id和tag2id保存下来
# word2id.to_pickle("word_to_id.pkl")
# tag2id.to_pickle("tag_to_id.pkl")


# 利用 labels（即状态序列）来统计转移概率
# 因为状态数比较少，这里用 dict={'I_tI_{t+1}'：p} 来实现
# A统计状态转移的频数
A = {
    'sb': 0,
    'ss': 0,
    'be': 0,
    'bm': 0,
    'me': 0,
    'mm': 0,
    'eb': 0,
    'es': 0
}

# zy 表示转移概率矩阵
zy = dict()
for label in labels:
    for t in xrange(len(label) - 1):
        key = label[t] + label[t + 1]
        A[key] += 1.0

zy['sb'] = A['sb'] / (A['sb'] + A['ss'])
zy['ss'] = 1.0 - zy['sb']
zy['be'] = A['be'] / (A['be'] + A['bm'])
zy['bm'] = 1.0 - zy['be']
zy['me'] = A['me'] / (A['me'] + A['mm'])
zy['mm'] = 1.0 - zy['me']
zy['eb'] = A['eb'] / (A['eb'] + A['es'])
zy['es'] = 1.0 - zy['eb']
keys = sorted(zy.keys())
print 'the transition probability: '
for key in keys:
    print key, zy[key]

zy = {i: np.log(zy[i]) for i in zy.keys()}


# 数据保存成pickle的格式。
data_path = "/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/ChineseSegmentation/dataset/"

with open('data.pkl', 'wb') as outp:
    pickle.dump(X, outp)
    pickle.dump(y, outp)
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
    pickle.dump(zy, outp)
print '** Finished saving the data.'




