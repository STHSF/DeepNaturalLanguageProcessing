# coding=utf-8

import os


class config(object):
    train_file_patch = './data/cnews/cnews.train.txt'
    val_file_patch = './data/cnews/cnews.val.txt'
    test_file_path = './data/cnews/cnews.test.txt'


def read_file(file_path):
    """
    读取文件被容
    :param file_path:
    :return:
    """
    labels, datas = [], []
    with open(file_path) as f:
        for line in f.readlines():
            try:
                label, data = line.strip().split('\t')
                labels.append(label)
                datas.append(data)
            except:
                pass
    return labels, datas


