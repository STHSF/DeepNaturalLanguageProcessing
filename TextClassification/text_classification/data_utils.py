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
    test_path = "./data/cnews/test.txt"


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
        读取词汇表，并生成{word: id}字典。
        :param vocab_path:
        :return:
        """
        with open(vocab_path) as f:
            words = [word.decode('utf-8').strip() for word in f.readlines()]
        word_to_id = dict(zip(words, range(len(words))))

        return words, word_to_id

    def build_category(self, categories=None):
        """
        生成标签列表，以及{category: id}表
        :return:
        """
        if categories is None:
            categories = [u'体育', u'财经', u'房产', u'家居', u'教育', u'科技', u'时尚', u'时政', u'游戏', u'娱乐']
        else:
            categories = categories

        cate_to_id = dict(zip(categories, range(len(categories))))

        return categories, cate_to_id

    def to_words(self, words, content):
        """
        将id转化成中文字符。
        :param words: list
        :param content: list
        :return: str
        """

        return ''.join(words[x] for x in content)

    def padding(self, sentence, word_to_id, length):
        """
        将sentence padding成固定长度的list
        :param sentence:
        :param word_to_id:
        :param length:
        :return:
        """
        if sentence is list:
            content_list = sentence.decode('utf-8')
        else:
            content_list = list(sentence.decode('utf-8'))
        # 现将中文字符转化成id
        padding_result = []
        for word in content_list:
            if word in word_to_id:
                padding_result.append(word_to_id[word])
            else:
                padding_result.append(0)
        # print 'padding_pre', padding_result

        length_pre = len(padding_result)
        # print 'length_re', length_pre
        if length_pre < length:
            padd = np.zeros([length - length_pre], dtype=int)
            padding_result.extend(padd)
        elif length_pre > length:
            padding_result = padding_result[:length]
        # print 'padding_after', padding_result
        # print 'padding_after_length', len(padding_result)
        return padding_result

    def contents_padding(self, word_to_id, input_file=config.test_path):
        """
        将input_file中的正文内容进行编码和padding
        :param word_to_id: 
        :param input_file: 
        :return: 
        """
        result = []
        with open(input_file) as f:
            for line in f.readlines():
                _, content = line.strip().split('\t')
                content_padding_list = self.padding(content, word_to_id, 10)
                result.append(content_padding_list)
        # print(np.shape(result))
        print('result{}'.format(result))

    def batch_generater(self):
        pass


def main():
    data_processing = data_utils()
    # data_processing.build_vocab(config.train_file_patch)
    word, word_to_id = data_processing.build_word(config.vocab_path)
    categories, cate_to_id = data_processing.build_category()

    # 测试padding模块
    # test_data = u'首先根据文本的存储格式，将标签和正堍文分别提取出来，处理过程中注意中文的编码.'
    # a = data_processing.padding(test_data, word_to_id, 50)
    # print a
    # print np.shape(a)

    # 测试contents_padding模块
    # result = []
    # with open(config.test_path) as f:
    #     for line in f.readlines():
    #         _, content = line.strip().split('\t')
    #         content_padding_list = data_processing.padding(content, word_to_id, 10)
    #         result.append(content_padding_list)
    # print np.shape(result)
    # print result
    data_processing.contents_padding(word_to_id)


if __name__ == '__main__':
    main()
