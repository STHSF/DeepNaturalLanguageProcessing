# coding=utf-8

import data_processing
# 读入数据
pos_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test3.txt'
neg_file_path = '/Users/li/Kunyan/MyRepository/DeepNaturalLanguageProcessing/DeepNLP/data/test2.txt'

tmp = data_processing.read_data(pos_file_path, neg_file_path)
res = data_processing.data_split(tmp[0], tmp[1])
x_train = res[0]
x_test = res[1]
label_train = res[2]
label_test = res[3]
x_train = data_processing.text_clean(x_train)
