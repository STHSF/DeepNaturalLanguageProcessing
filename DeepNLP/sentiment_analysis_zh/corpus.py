#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   中文预处理包括繁简转换，中文分词，去除非utf-8字符等
   暂定使用jieba分词
"""

import jieba
import globe
import os
import sys

reload(sys)
sys.setdefaultencoding('utf8')


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


# 文本处理
def sentence(file_parent_path):
    file_seg = {}
    for file_name in os.listdir(file_parent_path):
        file_path = os.path.join(file_parent_path, file_name)
        data = open(file_path)
        result = ""
        for d in data:
            temp = d.replace(" ", "").strip()
            result += filter_stop_word(list(jieba.cut(temp)))
        file_seg[file_name] = result
        data.close()
    return file_seg

count = 0


# 文本处理并写出
def sentence_out(file_parent_path, out):
    global count
    for file_name in os.listdir(file_parent_path):
        count += 1
        print '正在处理：', count
        file_path = os.path.join(file_parent_path, file_name)
        data = open(file_path)
        result = ""
        for d in data:
            temp = d.replace(" ", "").strip()
            result += filter_stop_word(list(jieba.cut(temp)))
        out.write(result + "\n")
        out.flush()
        data.close()


# 过滤停用词
def filter_stop_word(cut_result):
    stopwords = {}.fromkeys([line.rstrip() for line in open(globe.stopword)])
    final = []
    for seg in cut_result:
        seg = seg.encode('utf-8')
        if seg not in stopwords:
            final.append(seg)
    final_str = ",".join(final)
    return final_str


def do():
    input_txt = globe.input_txt
    output_txt = globe.output_txt
    # split_sentence(input_txt, output_txt)

    out_neg = open("/home/zhangxin/work/DeepSentiment/data/tagging/result_neg.txt", "wb")  # 写出文件
    out_neu = open("/home/zhangxin/work/DeepSentiment/data/tagging/result_neu.txt", "wb")  # 写出文件
    out_pos = open("/home/zhangxin/work/DeepSentiment/data/tagging/result_pos.txt", "wb")  # 写出文件

    sentence("/home/zhangxin/work/DeepSentiment/data/tagging/neg", out_neg)
    sentence("/home/zhangxin/work/DeepSentiment/data/tagging/neu", out_neu)
    sentence("/home/zhangxin/work/DeepSentiment/data/tagging/pos", out_pos)


if __name__ == "__main__":
    result = sentence('/home/zhangxin/work/workplace_python/DeepSentiment/data/predict_test/')
    for r in result:
        print r