#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: config.py
@time: 2020/1/10 3:02 下午
"""


class config(object):
    base_dir = "./data/cnews"
    train_dir = os.path.join(base_dir, "cnews.train.txt")
    test_dir = os.path.join(base_dir, "cnews.test.txt")
    val_dir = os.path.join(base_dir, "cnews.val.txt")
    vocab_dir = os.path.join(base_dir, "cnews.vocab.txt")

    train_file_patch = './data/cnews/cnews.train.txt'
    val_file_patch = './data/cnews/cnews.val.txt'
    test_file_path = './data/cnews/cnews.test.txt'
    vocab_path = "./data/vocab_list.txt"
    test_path = "./data/cnews/test.txt"
    stop_words_path = './data/stopwords.txt'