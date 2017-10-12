#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Config():
    num_steps = 10
    vocab_size = 5188
    embedding_size = 128
    is_training = True
    hidden_units = 300
    keep_pro = 0.0001
    layers_num = 32
    batch_size = 128
    num_classes = 5
    lr = 0.001
    max_grad_norm = 1

    # glove_filename = "data/embedding/news_tensite_ch_clean.model".format(dim)
    # trimmed_filename = "data/news_tensite_ch_clean_{}d.trimmed.npz".format(dim)
    words_filename = "data/words.txt"
    tags_filename = "data/tags.txt"

    dev_filename = "data/msra/msr_training.utf8.val"
    test_filename = "data/msra/msr_training.utf8.test"
    train_filename = "data/msra/msr_training.utf8.train"
    max_iter = None
    lowercase = True
    train_embeddings = False
    nepochs = 20
    dropout = 0.5
    lr_decay = 0.9
    nepoch_no_imprv = 3
    crf = True  # if crf, training is 1.7x slower
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
