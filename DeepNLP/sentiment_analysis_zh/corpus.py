#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   中文预处理包括繁简转换，中文分词，去除非utf-8字符等
   暂定使用jieba分词
"""

import jieba

input_txt = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/input/myInput.txt'
output_txt = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/input/myOutput.txt'


def split_sentence(input_file, output_file):
    fin = open(input_file, 'r')  # 以读的方式打开文件
    fout = open(output_file, 'w')  # 以写得方式打开文件

    for eachLine in fin:
        line = eachLine.strip().decode('utf-8', 'ignore')  # 去除每行首尾可能出现的空格，并转为Unicode进行处理
        # 用结巴分词，对每行内容进行分词,jieba.cut()返回的结构是一个可迭代的generator，可以用list(jieba.cut(...))转化为list
        word_list = list(jieba.cut(line))
        out_str = ''
        for word in word_list:
            out_str += word
            out_str += ','
        fout.write(out_str.strip().encode('utf-8') + '\n')  # 将分词好的结果写入到输出文件
    fin.close()
    fout.close()


split_sentence(input_txt, output_txt)
