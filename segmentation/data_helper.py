# coding=utf-8
import numpy as np
import pandas as pd
import re
import time
from itertools import chain

with open("msr_train.txt") as f:
    texts = f.read().decode('gbk')
sentences = texts.split('\r\n')


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

file_object = open('train_clean.txt', 'w')
file_object.write(str(texts.decode('utf-8')))
file_object.close()

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

tags = ['x', 's', 'b', 'm', 'e']
tag_ids = range(len(tags))

# 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

vocab_size = len(set_words)
print 'vocab_size={}'.format(vocab_size)

# 将word2id和tag2id保存下来
word2id.to_pickle("word_to_id.pkl")
tag2id.to_pickle("tag_to_id.pkl")






