# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:17:10 2017

@author: 100446517
"""

import numpy as np
from utils import rgb2yuv


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_data(directory):
    names = unpickle('{}/batches.meta'.format(directory))[b'label_names']
    print('names', names)

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

    return names, data, labels


def load_data():
    names, data, labels = read_data('../../../datasets/cfar10/')
    data_YUV = rgb2yuv(data)
    return data_YUV

