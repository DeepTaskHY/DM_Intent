import pickle
import random
import re
import numpy as np
from random import shuffle
from konlpy.tag import Twitter, Kkma
from gensim.models import KeyedVectors, Word2Vec

kkma = Kkma()
PAD = 0


class Batch:
    def __init__(self):
        self.inputs = []
        self.inputs_length = []
        self.targets = []


def make_embedding_from_input(line, word2id):
    words = kkma.pos(line)
    idlist = []
    for word in words:
        vocab = word[0]
        if vocab in word2id.keys():
            idlist.append(word2id[vocab])
        else:
            idlist.append(word2id['<UNK>'])

    sentence_length = len(words)

    return idlist, sentence_length


def load_data(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
        id2word = data['id2word']
        word2id = data['word2id']
        trainingSamples_list = data['trainingSamples_list']
        validSamples = data['validSamples']
        testSamples = data['testSamples']
        max_length = data['max_length']

        return max_length, id2word, word2id, trainingSamples_list, validSamples, testSamples


def create_batch(samples, maximum_word_size):
    batch = Batch()
    batch.inputs_length = [len(sample[0]) for sample in samples]
    for sample in samples:
        # source = list(sample[0])
        source = sample[0]
        pad = [PAD] * (maximum_word_size - len(source))
        batch.inputs.append(source + pad)
        batch.targets.append(sample[1])
    return batch


def get_batches_norandom(data, batch_size, max_length):
    batches = []
    data_len = len(data)

    def generate_next_samples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in generate_next_samples():
        batch = create_batch(samples, max_length)
        batches.append(batch)

    return batches
