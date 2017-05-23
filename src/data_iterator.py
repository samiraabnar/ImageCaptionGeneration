
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import logging
import pickle as pkl
import numpy

from io import open


rs = np.random.RandomState(1234)  # random state for shuffling data only

logger = logging.getLogger(__name__)

import tables
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# THIS IS AN INCOMPLETE PYTHON FILE. YOU NEED TO WRITE ALL THE WRAPPING AROUND IT!

def iterate_pkl_minibatches(path, fc7_data, w2i=None, shuffle=True, batch_size=10,
                        lowercase=False, max_unk_ratio=1.0):
    """
    Yield mini-batches of sentences and fc7 feature vectors

    w2i is a dictionary mapping words to indexes.
    :param data:
    :param shuffle:
    :param batch_size:
    :param lowercase: lowercase_words words
    :return minibatch:
    """
    sentences = []
    fc7_vectors = []

    fc7_file = tables.open_file(fc7_data, mode='r')
    fc7_vectors = fc7_file.root.feats[:]

    data = pkl.load(open(path, 'rb'))
    if shuffle:
        rs.shuffle(data)

    sentences = []
    fc7 = []

    for instance in data:
        if max_unk_ratio < 1.0:
            # calculate the unkyness of this sentence. If it's greater than 
            # the max_unk_ratio, bypass this example because it
            # will have too many unknown words
            words = instance[0].split()
            ids = [w2i[word] if word in w2i else 1 for word in words]
            unks = sum([x for x in ids if x == 1])
            if float(unks)/len(words) <= max_unk_ratio:
                sentences.append(instance[0])
                fc7.append(fc7_vectors[instance[1]])
            else:
                continue
        else:
            sentences.append(instance[0])
            fc7.append(fc7_vectors[instance[1]])

        if len(sentences) >= batch_size:
            yield [sentences, fc7]
            del sentences[:]
            del fc7[:]

    # last batch
    if len(sentences) > 0:
        yield [sentences, fc7]
        del sentences[:]
        del fc7[:]


def iterate_minibatches(data, fc7_data, w2i=None, shuffle=True, batch_size=10,
                        lowercase=False, max_unk_ratio=1.0,
                        sort_by_length=False):
    """
    Use for sentence files, not pickle files.
    Yield mini-batches of sentences and fc7 feature vectors
    :param data:
    :param shuffle:
    :param batch_size:
    :param lowercase: lowercase_words words
    :return minibatch:
    """
    sentences = []
    fc7_vectors = []

    fc7_file = tables.open_file(fc7_data, mode='r')
    fc7_vectors = fc7_file.root.feats[:]

    with open(data, mode='r', encoding='utf-8') as f:
        data = []
        i = 0
        for line in f:
            data.append((line.replace('\n',''), i))
            i += 1
        if sort_by_length:
            slen = numpy.array([len(s[0].split()) for s in data])
            sidx = slen.argsort()

            _data = [data[j] for j in sidx]
            data = _data

        #if shuffle:
        #    rs.shuffle(data)

        sentences = []
        fc7 = []

        for instance in data:
            if max_unk_ratio < 1.0:
                # calculate the unkyness of this sentence. If it's greater than 
                # the max_unk_ratio, bypass this example because it
                # will have too many unknown words
                words = instance[0].split()
                ids = [w2i[word] if word in w2i else 1 for word in words]
                unks = sum([x for x in ids if x == 1])
                if float(unks)/len(words) <= max_unk_ratio:
                    sentences.append(instance[0])
                    fc7.append(fc7_vectors[instance[1]])
                else:
                    continue
            else:
                sentences.append(instance[0])
                fc7.append(fc7_vectors[instance[1]])

            if len(sentences) >= batch_size:
                yield [sentences, fc7]
                del sentences[:]
                del fc7[:]

        # last batch
        if len(sentences) > 0:
            yield [sentences, fc7]
            del sentences[:]
            del fc7[:]

def load_vocabularies(config):
    """
    loads json-formatted vocabularies from disk
    :param config:
    :return:
    """
    w2i = load_json(config['word_vocabulary'])
    max_words = config['max_words']

    if max_words > 0:
        for word, idx in list(w2i.items()):
            if idx >= max_words:
                # print('deleting from dict: word={} idx={}'.format(word, idx))
                del w2i[word]

    w2i['</s>'] = 0
    w2i['UNK'] = 1

    return w2i


if __name__ == '__main__':
    for bool in iterate_minibatches("../Multi30k/train.en","../Multi30k/train-resnet50.hdf5"):
        print(bool)