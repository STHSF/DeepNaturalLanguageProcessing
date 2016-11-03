# coding=utf-8
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
import logging
import globe
import data_processing
import word2vec_gensim_train
import corpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def _data_read(pos_file_path, neg_file_path, model_path):
    """read data word2vec model from file path,
    Args:
        pos_file_path: Positive file path.
        neg_file_path: Negative file path.
    Returns:
        A list contains training data with labels and test data with labels.
    Raises:
           IOError: An error occurred accessing the bigtable.Table object.
    """

    tmp = data_processing.read_data(pos_file_path, neg_file_path)
    res = data_processing.data_split(tmp[0], tmp[1])
    (train_data, test_data, train_labels, test_labels) = (res[0], res[1], res[2], res[3])

    # print train_labels[0]
    train_data = data_processing.text_clean(train_data)
    test_data = data_processing.text_clean(test_data)

    # 词向量的维度
    n_dim = globe.n_dim
    # load word2vec model from model path
    text_vecs = []
    try:
        word2vec_model = Word2Vec.load(model_path)

        text_vecs = word2vec_gensim_train.text_vecs(train_data, test_data, n_dim, word2vec_model)
    except IOError:
        pass
    # 生成文本向量
    train_data_vecs = text_vecs[0]
    # print train_data_vecs.shape
    test_data_vecs = text_vecs[1]
    # print test_data_vecs.shape

    return train_data_vecs, train_labels, test_data_vecs, test_labels


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    with open(filename, 'rb') as bytestream:
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    # print labels_dense.dtype
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(labels, one_hot=False, num_classes=2):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    # print labels.shape
    if one_hot:
        return dense_to_one_hot(labels.astype(np.uint8), num_classes)
    return labels


class DataSet(object):
    """Construct a DataSet.
    Attributes:
        data: text vectors
        labels: labels of every data
    """

    def __init__(self, data, labels):
        """inits DataSet.
        """
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = data.shape[0]

        # 数据归一化

    @property
    def length(self):
        return self.length

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]

    def next_batch_data(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            # self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end]  # , self._labels[start:end]


def read_data_sets():

    # 读入数据
    pos_file_path = globe.pos_file_path
    neg_file_path = globe.neg_file_path
    w2c_model_path = globe.model_path

    # train_data = '/Users/li/workshop/DataSet/sentiment/train/result_pos.txt'
    # train_labels = '/Users/li/workshop/DataSet/sentiment/train/result_pos.txt'
    # test_data = '/Users/li/workshop/DataSet/sentiment/train/result_pos.txt'
    # test_labels = '/Users/li/workshop/DataSet/sentiment/train/result_pos.txt'
    #
    # train_data = base.load_csv_without_header(train_data, train_dir,
    #                                           features_dtype=tf.float32)
    #
    # train_labels_file = base.load_csv_without_header(train_labels, train_dir,
    #                                                  features_dtype=tf.float32)
    #
    # train_labels = extract_labels(train_labels_file, one_hot=one_hot)
    #
    # test_data = base.load_csv_without_header(test_data, train_dir,
    #                                          features_dtype=tf.float32)
    # test_labels_file = base.load_csv_without_header(test_labels, train_dir,
    #                                                 features_dtype=tf.float32)
    # test_labels = extract_labels(test_labels_file, one_hot=one_hot)

    raw_data = _data_read(pos_file_path, neg_file_path, w2c_model_path)

    train_data = raw_data[0]
    # train_label = np.reshape(raw_data[1], (raw_data[1].shape[0],))
    # print train_label.shape
    train_labels = extract_labels(raw_data[1], one_hot=True)
    # for l in train_labels:
    #     print 'L ',l

    test_data = raw_data[2]
    # print test_data.shape
    # test_label = np.reshape(raw_data[3], (raw_data[1].shape[0], 1))
    test_labels = extract_labels(raw_data[3], one_hot=True)
    # print train_label.shape

    validation_size = 500
    validation_data = train_data[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_data = train_data[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_data, train_labels)
    # print train.raw_data[0], train.labels[0]
    validation = DataSet(validation_data, validation_labels)
    test = DataSet(test_data, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


def read_data_sets_predict():

    # 读入数据,并切词\去停处理
    predict_parent_file = globe.predict_parent_file
    file_seg = corpus.sentence(predict_parent_file)

    # 构建word2vec词向量
    w2c_model_path = globe.model_path

    text_vecs = {}
    try:
        word2vec_model = Word2Vec.load(w2c_model_path)

        for title in file_seg.keys():
            # print '【标题】', title
            # print '【正文】', file_seg[title]

            doc = file_seg[title]
            doc_vec = word2vec_gensim_train.doc_vecs_zx(doc, word2vec_model)
            # text_vecs.append(doc_vec)
            text_vecs[title] = doc_vec
    except IOError:
        pass

    return text_vecs


# def load_data():
#     return read_data_sets()

if __name__ == '__main__':
    read_data_sets()


