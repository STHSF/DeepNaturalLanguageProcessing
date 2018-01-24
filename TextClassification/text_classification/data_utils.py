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
    vocab_path = "./data/vocab_list.txt"


class data_utils(object):
    def __init__(self):
        pass

    def read_file(self, file_path):
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

    def build_vocab(self, file_patch, vocab_size=5000):
        """
        统计输入文件中的中文字符的出现次数，将出现次数最多的前vocab_size个中文字符保留下来，保存到文件中。
        :param file_patch:
        :param vocab_size:
        :return:
        """
        _, contents = self.read_file(file_patch)
        all_data = []
        # 拼接
        for content in contents:
            all_data.extend(list(content.decode('utf-8')))
        # 统计中文字符出现的次数
        counter = Counter(all_data)
        # 挑选前vocab_size个中文字符
        counter_lists = counter.most_common(vocab_size - 1)
        # 巧用zip函数
        words, _ = map(list, zip(*counter_lists))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<PAD>'] + words

        # 将筛选出来的中文字符写入文件
        with open(config.vocab_path, 'w') as f:
            f.write('\n'.join(words) + '\n')

    def build_word(self, vocab_path):
        """
        读取词汇表，并生成word_to_id表。
        :param vocab_path:
        :return:
        """
        with open(vocab_path) as f:
            words = [word.encode('utf-8').strip() for word in f.readlines()]
        word_to_id = list(zip(words, range(len(words))))

        return words, word_to_id

    def build_category(self, categories=None):
        """
        生成标签列表，以及标签编号列表
        :return:
        """
        if categories is None:
            categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        else:
            categories = categories

        cate_to_id = list(zip(categories, range(len(categories))))

        return categories, cate_to_id




def main():
    data_processing = data_utils()
    # data_processing.build_vocab(config.train_file_patch)
    # data_processing.build_word(config.vocab_path)
    data_processing.build_category()


if __name__ == '__main__':
    main()



