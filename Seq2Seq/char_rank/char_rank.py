# coding=utf-8
import tensorflow as tf
import seq2seq

file_path = './data/'
with open(file_path + 'letters_source.txt') as f:
    source_data = f.read().split('\n')

with open(file_path + 'letters_target.txt') as f:
    target_data = f.read().split('\n')


def charset(data):
    special_char = ['<EOS>', '<GO>', '<UNK>', '<PAD>']
    char_set = list(set(char for vocab in data for char in vocab)) + special_char
    char_to_int = {char: ids for ids, char in enumerate(char_set)}
    int_to_char = dict(enumerate(char_set))
    return char_to_int, int_to_char


# 这里需要注意的是encoding和decoding需要分别编码，因为本文中
source_char_to_int, source_int_to_char = charset(source_data)
source_letters_to_int = [[source_char_to_int.get(char, source_char_to_int['<UNK>'])
                          for char in vocab] for vocab in source_data]

target_char_to_int, target_int_to_char = charset(target_data)
target_letters_to_int = [[target_char_to_int.get(char, target_char_to_int['<UNK>'])
                          for char in vocab] + [target_char_to_int['<EOS>']] for vocab in target_data]





model = seq2seq.seq2seqModel(10, 20, 512, 512)
