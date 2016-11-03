#!/usr/bin/env python
# coding:utf-8
# -*- coding: utf-8 -*-

# 全局变量模块

# corpus
# input_txt = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/input/myInput.txt'
# output_txt = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/input/myOutput.txt'
# stopword = 'data/stopword.txt'
#
# data_patent = ''
# data_process_result = '/home/zhangxin/work/workplace_python/DeepSentiment/data/tagging/result.txt'
#
# # train
#
# train_data = ('neg', '/home/zhangxin/work/DeepSentiment/data/train/result_neg.txt')
# train_neu = ('neu', '/home/zhangxin/work/DeepSentiment/data/train/result_neu.txt')
# train_pos = ('pos', '/home/zhangxin/work/DeepSentiment/data/train/result_pos.txt')

# train_data = [(1, '/home/zhangxin/work/DeepSentiment/data/train/result_neg.txt'),
#               (2, '/home/zhangxin/work/DeepSentiment/data/train/result_neu.txt'),
#               (3, '/home/zhangxin/work/DeepSentiment/data/train/result_pos.txt')]

# train_data = [(0, '/home/zhangxin/work/DeepSentiment/data/train/result_neg.txt'),
#               (1, '/home/zhangxin/work/DeepSentiment/data/train/result_pos.txt')]
#
# file_parent = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data'
# file_pos = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
# file_neg = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'
# model_path = '/home/zhangxin/work/workplace_python/DeepSentiment/data/word2vec_model/'
# model_path = 'data/model_w2v/mymodel'


#================================================================================
# train_data = ('neg', '/home/zhangxin/work/DeepSentiment/data/train/result_neg.txt')
# train_neu = ('neu', '/home/zhangxin/work/DeepSentiment/data/train/result_neu.txt')
# train_pos = ('pos', '/home/zhangxin/work/DeepSentiment/data/train/result_pos.txt')

# train_data = [('neg', '/home/zhangxin/work/DeepSentiment/data/train/result_neg.txt'),
#               ('neu', '/home/zhangxin/work/DeepSentiment/data/train/result_neu.txt'),
#               ('pos', '/home/zhangxin/work/DeepSentiment/data/train/result_pos.txt')]

# file_parent = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data'
# file_neg = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
# file_pos = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'
# model_path = '/home/zhangxin/work/workplace_python/DeepNaturalLanguageProcessing/DeepNLP/word2vecmodel/mymodel'

# input_txt = '/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/input/myInput.txt'
# output_txt = '/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/input/myOutput.txt'
# stopword = '/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/stopword.txt'

# data_process_result = '/Users/li/workshop/DataSet/sentiment/tagging/result.txt'

# train_data = ('neg', '/Users/li/workshop/DataSet/sentiment/train/result_neg.txt')
# train_neu = ('neu', '/Users/li/workshop/DataSet/sentiment/train/result_neu.txt')
# train_pos = ('pos', '/Users/li/workshop/DataSet/sentiment/train/result_pos.txt')
#
# train_data = [('neg', '/Users/li/workshop/DataSet/sentiment/train/result_neg.txt'),
#               ('neu', '/Users/li/workshop/DataSet/sentiment/train/result_neu.txt'),
#               ('pos', '/Users/li/workshop/DataSet/sentiment/train/result_pos.txt')]

# file_parent = '/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data'
# file_neg = '/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
# file_pos = '/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'
model_path = '/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/word2vecmodel/model'

# w2v模型的参数
n_dim = 200
min_count = 2

pos_file_path = '/Users/li/workshop/DataSet/sentiment/train/result_pos.txt'
neg_file_path = '/Users/li/workshop/DataSet/sentiment/train/result_neg.txt'

# pos_file_path = '/home/zhangxin/work/workplace_python/DeepSentiment/data/train/result_pos.txt'
# neg_file_path = '/home/zhangxin/work/workplace_python/DeepSentiment/data/train/result_neg.txt'


#预测
# model_rnn_path = 'data/model_rnn/model.ckpt'
# predict_parent_file = 'data/text_predict'