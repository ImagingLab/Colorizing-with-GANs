# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:17:10 2017

@author: 100446517
"""

import os
import pickle
import numpy as np
from utils import preproc


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar10_data(directory):
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


def load_cifar10_data(normalize=False, shuffle=False, flip=False, count=-1):
    names, data, labels, data_test, labels_test = read_cifar10_data('../../../datasets/cfar10/')

    if shuffle:
        np.random.shuffle(data)

    if count != -1:
        data = data[:count]

    return preproc(data, normalize=normalize, flip=flip)


def load_cfar10_test_data(normalize=False, count=-1):
    names, data, labels, data_test, labels_test = read_cifar10_data('../../../datasets/cfar10/')

    if count != -1:
        data_test = data[:count]

    return preproc(data_test, normalize=normalize)


def load_imagenet_data(idx, normalize=False, flip=False, count=-1):
    data_file = '../../../datasets/ImageNet/train_data_batch_'
    d = unpickle(data_file + str(idx))
    x = d['data']
    mean_image = d['mean']

    if count != -1:
        x = x[:count]

    return preproc(x, normalize=normalize, flip=flip, mean_image=mean_image)


def load_imagenet_test_data(normalize=False, count=-1):
    d = unpickle('../../../datasets/ImageNet/val_data')
    x = d['data']

    if count != -1:
        x = x[:count]

    return preproc(x, normalize=normalize)

# import pickle
#
# for batch in range(1, 11):
#     print('read batch ' + str(batch))
#     d = unpickle('../../../datasets/ImageNet/train_data_batch_' + str(batch))
#     print('read batch ' + str(batch) + ' complete')
#     x = d['data']
#     img_size = int(x.shape[1] / 3)
#     data_size = 32000
#
#     for sec in range(1, 5):
#         ix = str((batch - 1) * 4 + sec)
#         data = x[(sec - 1) * data_size:sec * data_size]
#         print('convert section ' + ix)
#         data_RGB = np.dstack((data[:, :img_size], data[:, img_size:2 * img_size], data[:, 2 * img_size:]))
#         data_RGB = data_RGB.reshape((data_size, int(np.sqrt(img_size)), int(np.sqrt(img_size)), 3))
#         data_RGB = data_RGB[0:data_size, :, :, :]
#         data_RGB_flip = data_RGB[:, :, :, ::-1]
#         data_RGB = np.concatenate((data_RGB, data_RGB_flip), axis=0)
#         print('convert section ' + ix + ' complete')
#         pickle.dump(data_RGB, open('../../../datasets/ImageNet/train_' + ix, 'wb'))
