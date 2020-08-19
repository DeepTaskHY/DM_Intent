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


def max_length(samples):
    max = 0
    for sample in samples:
        length = len(sample[0])
        if length > max:
            max = length
    return max


def split_train_valid_test(dialogue_list):
    shuffle(dialogue_list)
    class_0 = []
    class_1 = []
    class_2 = []
    class_3 = []
    class_4 = []
    class_5 = []
    class_6 = []

    for dialogue in dialogue_list:
        label = dialogue[-2]
        if label == '0':
            class_0.append(dialogue)
        elif label == '1':
            class_1.append(dialogue)
        elif label == '2':
            class_2.append(dialogue)
        elif label == '3':
            class_3.append(dialogue)
        elif label == '4':
            class_4.append(dialogue)
        elif label == '5':
            class_5.append(dialogue)
        elif label == '6':
            class_6.append(dialogue)

    print(len(class_0))
    print(len(class_1))
    print(len(class_2))
    print(len(class_3))
    print(len(class_4))
    print(len(class_5))
    print(len(class_6))

    train = []
    train.extend(class_0[:int(len(class_0) * 14 / 20)])
    train.extend(class_1[:int(len(class_1) * 14 / 20)])
    train.extend(class_2[:int(len(class_2) * 14 / 20)])
    train.extend(class_3[:int(len(class_3) * 14 / 20)])
    train.extend(class_4[:int(len(class_4) * 14 / 20)])
    train.extend(class_5[:int(len(class_5) * 14 / 20)])
    train.extend(class_6[:int(len(class_6) * 14 / 20)])
    valid = []
    valid.extend(class_0[int(len(class_0) * 14 / 20):int(len(class_0) * 17 / 20)])
    valid.extend(class_1[int(len(class_1) * 14 / 20):int(len(class_1) * 17 / 20)])
    valid.extend(class_2[int(len(class_2) * 14 / 20):int(len(class_2) * 17 / 20)])
    valid.extend(class_3[int(len(class_3) * 14 / 20):int(len(class_3) * 17 / 20)])
    valid.extend(class_4[int(len(class_4) * 14 / 20):int(len(class_4) * 17 / 20)])
    valid.extend(class_5[int(len(class_5) * 14 / 20):int(len(class_5) * 17 / 20)])
    valid.extend(class_6[int(len(class_6) * 14 / 20):int(len(class_6) * 17 / 20)])
    test = []
    test.extend(class_0[int(len(class_0) * 17 / 20):])
    test.extend(class_1[int(len(class_1) * 17 / 20):])
    test.extend(class_2[int(len(class_2) * 17 / 20):])
    test.extend(class_3[int(len(class_3) * 17 / 20):])
    test.extend(class_4[int(len(class_4) * 17 / 20):])
    test.extend(class_5[int(len(class_5) * 17 / 20):])
    test.extend(class_6[int(len(class_6) * 17 / 20):])

    return train, valid, test


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


def make_sample(lines, kkma, word2id):
    samples = []
    label_list = ['0', '1', '2', '3', '4', '5', '6']
    for line in lines:
        sample = []
        sentence = line[:-2]
        label = line[-2]
        if label not in label_list:
            print()
        words = kkma.pos(sentence)
        idlist = []
        for word in words:
            vocab = word[0]
            if vocab in word2id.keys():
                idlist.append(word2id[vocab])
            else:
                idlist.append(word2id['<UNK>'])
        sample.append(idlist)
        sample.append(label)
        samples.append(sample)
    return samples


def get_max_length(samples):
    max_length = 0
    for sentence in samples:
        if len(sentence) > max_length:
            max_length = len(sentence)
    return max_length


def data_make(input_path, output_path, threshold):

    kkma = Kkma()

    with open(input_path, 'r', encoding='utf-8') as handle:

        lines = handle.readlines()

        train, valid, test = split_train_valid_test(lines)

        print('loading vocab from pre-trained word2vec files...')

        model = Word2Vec.load('./data/ko.word2vec.bin').wv
        word2vec_vocab = model.vocab

        print('-done!')

        up_vocab = {}
        down_vocab = {}

        sentence_set = set([])

        for line in train: # train data set

            sentence = line[:-2]
            if sentence in sentence_set:
                continue
            else:
                sentence_set.add(sentence)
            words = kkma.pos(sentence)
            for word in words:
                vocab = word[0]
                if vocab in word2vec_vocab.keys():
                    if vocab in up_vocab.keys():
                        pass
                    else:
                        up_vocab[vocab] = len(up_vocab)
                else:
                    if vocab in down_vocab.keys():
                        pass
                    else:
                        down_vocab[vocab] = len(down_vocab)

        print('hit : ', len(up_vocab), 'missed : ', len(down_vocab))

        id2word = {}
        word2id = {}

        id2word[0] = '<PAD>'
        id2word[1] = '<UNK>'
        word2id['<PAD>'] = 0
        word2id['<UNK>'] = 1

        # index = 2

        for word in down_vocab.keys():
            id2word[len(id2word)] = word
            word2id[word] = len(word2id)

        id2word[len(id2word)] = '<SPLIT>'
        word2id['<SPLIT>'] = len(word2id)

        for word in up_vocab.keys():
            id2word[len(id2word)] = word
            word2id[word] = len(word2id)

        make_initW(word2id, './data/initW_noHAN_kkma_190907.pkl')

        trainingSamples = make_sample(train, kkma, word2id)
        trainingSamples_list = []
        for _ in range(10):
            random.shuffle(trainingSamples)
            trainingSamples_ratiochange = []
            ratiochecker = [0, 0, 0, 0, 0, 0, 0]

            for i in trainingSamples:
                if ratiochecker[int(i[1])] < 20000:
                    trainingSamples_ratiochange.append(i)
                    ratiochecker[int(i[1])] += 1
            trainingSamples_list.append(trainingSamples_ratiochange[:int(len(trainingSamples)*1/1)])

        validSamples = make_sample(valid, kkma, word2id)
        testSamples = make_sample(test, kkma, word2id)

        max_len = 0
        for trainingSamples in trainingSamples_list:
            temp_max_len = max_length(trainingSamples)
            if max_len < temp_max_len:
                max_len = temp_max_len
        max_len = max(max_len, max_length(validSamples), max_length(testSamples))

        data_dict = {}
        data_dict['id2word'] = id2word
        data_dict['word2id'] = word2id
        data_dict['trainingSamples_list'] = trainingSamples_list
        data_dict['validSamples'] = validSamples
        data_dict['testSamples'] = testSamples
        data_dict['max_length'] = max_len

        with open(output_path, 'wb') as pkl:
            pickle.dump(data_dict, pkl)


def make_initW(word2id, path):
    index_split = word2id['<SPLIT>']
    model = Word2Vec.load('./data/ko.word2vec.bin').wv
    vocab = model.vocab
    initW = np.random.uniform(-0.25, 0.25, (len(word2id) - (index_split + 1), 200))

    for word in vocab.keys():
        if word in word2id.keys():
            initW[word2id[word] - (index_split + 1)] = model.word_vec(word)

    with open(path, 'wb') as handle:
        pickle.dump(initW, handle)


def load_data(path):
    with open(path,'rb') as handle:
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


def get_batches(data, batch_size, max_length):
    random.shuffle(data)
    batches = []
    data_len = len(data)

    def generate_next_samples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in generate_next_samples():
        batch = create_batch(samples, max_length)
        batches.append(batch)

    return batches


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


if __name__ == "__main__":
    data_make('./data/train.txt', './data/data_noHAN_kkma_190907_train.pkl', 1)
