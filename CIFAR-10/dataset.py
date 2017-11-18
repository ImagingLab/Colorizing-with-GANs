# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:17:10 2017

@author: 100446517
"""

import numpy as np
from utils import preproc


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_data(directory):
    names = unpickle('{}/batches.meta'.format(directory))[b'label_names']
    data, labels = [], []
    for i in range(1, 6):
        filename = '{}/data_batch_{}'.format(directory, i)
        batch_data = unpickle(filename)
        if len(data) > 0:
            data = np.vstack((data, batch_data[b'data']))
            labels = np.hstack((labels, batch_data[b'labels']))
        else:
            data = batch_data[b'data']
            labels = batch_data[b'labels']

    filename = '{}/test_batch'.format(directory)
    batch_data = unpickle(filename)
    data_test = batch_data[b'data']
    labels_test = batch_data[b'labels']

    return names, data, labels, data_test, labels_test


def load_data(normalize=False, shuffle=True, count=-1):
    names, data, labels, data_test, labels_test = read_data('../../../datasets/cfar10/')

    if shuffle:
        np.random.shuffle(data)

    if count != -1:
        data = data[:count]

    return preproc(data, normalize)


def load_test_data(normalize=False, count=-1):
    names, data, labels, data_test, labels_test = read_data('../../../datasets/cfar10/')

    if count != -1:
        data_test = data[:count]

    return preproc(data_test, normalize)
