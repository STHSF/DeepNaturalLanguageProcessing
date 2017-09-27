# coding: utf-8

import jieba


# 文本预处理（字符替换）
def preprocess(data):
    """
    对文本中的字符进行替换
    :param text:
    :return:
    """
    data = data.replace('《', '')
    data = data.replace('》', '')
    data = data.replace('【', '')
    data = data.replace('】', '')
    data = data.replace(' ', ';')
    data = data.replace('\n', '.')

    words = jieba.lcut(data, cut_all=False)  # 全模式切词

    return words


def write_file(words, fname):
    # 写入文件
    with open(fname, 'a') as f:
        for w in words:
            f.write(w.encode('utf-8') + '\n')

    print('Done')


def get_batch():
    pass

if __name__ == "__main__":
    dir = "/Users/li/workshop/MyRepository/DeepNaturalLanguageProcessing/LanguageModel/lyrics_writer/data/"
    # 加载文本
    with open(dir + "lyrics.txt") as f:
        text = f.read()
    print(text)
    words = preprocess(text)
    write_file(words, dir + "split.txt")