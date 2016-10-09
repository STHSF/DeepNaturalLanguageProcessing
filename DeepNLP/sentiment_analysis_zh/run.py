# coding=utf-8

import data_processing
# 读入数据
pos_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
neg_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'

tmp = data_processing.read_data(pos_file_path, neg_file_path)
res = data_processing.data_split(tmp[0], tmp[1])
