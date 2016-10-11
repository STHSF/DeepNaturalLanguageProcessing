#!/usr/bin/env python
# coding:utf-8
# -*- coding: utf-8 -*-

# 全局变量模块

# corpus
input_txt = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/input/myInput.txt'
output_txt = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/input/myOutput.txt'
stopword = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/stopword.txt'

# input_txt = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/input/myInput.txt'
# output_txt = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/input/myOutput.txt'

data_patent = ''
data_process_result = '/home/zhangxin/work/DeepSentiment/data/tagging/result.txt'

# train
train_data = ('neg', '/home/zhangxin/work/DeepSentiment/data/train/result_neg.txt')
train_neu = ('neu', '/home/zhangxin/work/DeepSentiment/data/train/result_neu.txt')
train_pos = ('pos', '/home/zhangxin/work/DeepSentiment/data/train/result_pos.txt')

train_data = [('neg', '/home/zhangxin/work/DeepSentiment/data/train/result_neg.txt'),
              ('neu', '/home/zhangxin/work/DeepSentiment/data/train/result_neu.txt'),
              ('pos', '/home/zhangxin/work/DeepSentiment/data/train/result_pos.txt')]
#
file_parent = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data'
file_neg = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
file_pos = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'
model_path = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/word2vecmodel/mymodel'

# w2v模型的参数
n_dim = 200
min_count = 2
