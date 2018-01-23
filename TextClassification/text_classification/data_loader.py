# coding=utf-8

import os
import sys
import numpy as np
from collections import Counter

reload(sys)
sys.setdefaultencoding("utf-8")


class config(object):
    train_file_patch = './data/cnews/cnews.train.txt'
    val_file_patch = './data/cnews/cnews.val.txt'
    test_file_path = './data/cnews/cnews.test.txt'


def read_file(file_path):
    """
    读取文件内容，并将文件中的label和content分开存储
    :param file_path:
    :return:
    """
    labels, contents = [], []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            try:
                label, data = line.strip().split('\t')
                labels.append(label)
                contents.append(data)
            except:
                pass
    return labels, contents


def build_vocab(file_patch, vocab_size=5000):
    _, contents = read_file(file_patch)
    all_data = []
    # 拼接
    for content in contents:
        all_data.extend(list(content.decode('utf-8')))
    # 统计中文字符出现的次数
    counter = Counter(all_data)
    # 挑选前vocab_size个中文字符
    counter_dict = counter.most_common(vocab_size - 1)

    words, _ = list(zip(*counter_dict))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    # 将筛选出来的中文字符写入文件
    with open('rr.txt', 'w') as f:
        f.write('\n'.join(words) + '\n')


build_vocab(config.val_file_patch)



