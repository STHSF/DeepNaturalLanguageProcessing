#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   中文预处理包括繁简转换，中文分词，去除非utf-8字符等
   暂定使用jieba分词
"""

import jieba
import jieba.posseg as pseg
import os

def process():
    """
    文本处理成gensim可输入格式
    """
    file_parent_path = "/home/zhangxin/work/DeepSentiment/data"
    for file_name in os.listdir(file_parent_path):
        file_path = os.path.join(file_parent_path, file_name)

        # 读出数据并处理
        data = open(file_path)
        result = []
        for line in data:

            print line

        data.close()

        # 处理结果写入原文件


def do():
    sen = '我来自北京清华大学'
    res = jieba.cut(sen)
    print "Default Mode:", "/ ".join(res)

    pres = pseg.cut(sen)
    for w in pres:
        print w.word,w.flag

if __name__ == '__main__':
    process()