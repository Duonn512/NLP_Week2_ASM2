# -*- coding: utf-8 -*-
import os
import pandas as pd
import string
from pyvi import ViTokenizer
from gensim.models import Word2Vec

# path data
pathdata = './datatrain.txt'

def read_data(path):
    traindata = []
    sents = open(pathdata, 'r', encoding='utf-8').readlines()
    for sent in sents:
        traindata.append(sent.split())
    return traindata


if __name__ == '__main__':
    train_data = read_data(pathdata)
    model = Word2Vec(vector_size=150, window=10, min_count=2, workers=4, sg=0)

    model.build_vocab(train_data)

    # Check if vocabulary is built
    if not model.wv.key_to_index:
        raise RuntimeError("Vocabulary is empty. Ensure your data is properly formatted and non-empty.")

    # Train the model
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

    model.wv.save("../model/word2vec_skipgram.model")
