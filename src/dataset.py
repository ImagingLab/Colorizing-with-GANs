import os
import glob
import pickle
import numpy as np
from scipy.misc import imread
from skimage import color
from utils import preproc

CIFAR10_PATH = '../../../datasets/cfar10'
IMAGENET_PATH = '../../../datasets/ImageNet'


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


def load_cifar10_data(normalize=False, shuffle=False, flip=False, count=-1, outType='YUV'):
    names = unpickle('{}/batches.meta'.format(CIFAR10_PATH))[b'label_names']
    data, labels = [], []
    for i in range(1, 6):
        filename = '{}/data_batch_{}'.format(CIFAR10_PATH, i)
        batch_data = unpickle(filename)
        if len(data) > 0:
            data = np.vstack((data, batch_data[b'data']))
            labels = np.hstack((labels, batch_data[b'labels']))
        else:
            data = batch_data[b'data']
            labels = batch_data[b'labels']

    if shuffle:
        np.random.shuffle(data)

    if count != -1:
        data = data[:count]

    return preproc(data, normalize=normalize, flip=flip, outType=outType)


def load_cifar10_test_data(normalize=False, count=-1, outType='YUV'):
    filename = '{}/test_batch'.format(CIFAR10_PATH)
    batch_data = unpickle(filename)
    data_test = batch_data[b'data']
    labels_test = batch_data[b'labels']

    if count != -1:
        data_test = data_test[:count]

    return preproc(data_test, normalize=normalize, outType=outType)


def load_extra_data(outType='YUV'):
    names = np.array(glob.glob('extra/*.jpg'))
    files = np.array([imread(f) for f in names])

    if outType == 'YUV':
        return color.rgb2yuv(files), files

    elif outType == 'LAB':
        return color.rgb2lab(files), color.rgb2gray(files)[:, :, :, None]


def load_imagenet_data(idx, normalize=False, flip=False, count=-1, outType='YUV'):
    data_file = IMAGENET_PATH + '/train_data_batch_'
    d = unpickle(data_file + str(idx))
    x = d['data']
    mean_image = d['mean']

    if count != -1:
        x = x[:count]

    return preproc(x, normalize=normalize, flip=flip, mean_image=mean_image, outType=outType)


def load_imagenet_test_data(normalize=False, count=-1, outType='YUV'):
    d = unpickle(IMAGENET_PATH + '/val_data')
    x = d['data']

    if count != -1:
        x = x[:count]

    return preproc(x, normalize=normalize, outType=outType)


def imagenet_data_generator(batch_size, normalize=False, flip=False, scale=1, outType='YUV'):
    while True:
        for idx in range(1, 11):
            data_file = IMAGENET_PATH + '/train_data_batch_'
            d = unpickle(data_file + str(idx))
            x = d['data']
            mean_image = d['mean']
            count = 0
            while count <= x.shape[0] - batch_size:
                data = x[count:count + batch_size]
                count = count + batch_size

                if outType == 'YUV':
                    data_yuv, data_rgb = preproc(data, normalize=normalize, flip=flip, mean_image=mean_image)
                    yield data_yuv[:, :, :, :1], data_yuv[:, :, :, 1:] * scale

                elif outType == 'LAB':
                    lab, grey = preproc(data, normalize=normalize, flip=flip, mean_image=mean_image, outType=outType)
                    yield lab, grey


def imagenet_test_data_generator(batch_size, normalize=False, scale=1, count=-1, outType='YUV'):
    d = unpickle(IMAGENET_PATH + '/val_data')
    x = d['data']

    if count != -1:
        x = x[:count]

    while True:
        count = 0
        while count <= x.shape[0] - batch_size:
            data = x[count:count + batch_size]
            count = count + batch_size

            if outType == 'YUV':
                data_yuv, data_rgb = preproc(data, normalize=normalize, outType=outType)
                yield data_yuv[:, :, :, :1], data_yuv[:, :, :, 1:] * scale

            elif outType == 'LAB':
                lab, grey = preproc(data, normalize=normalize, outType=outType)
                yield lab, grey


def dir_data_generator(dir, batch_size, data_range=(0, 0), outType='YUV'):
    names = np.array(glob.glob(dir + '/*.jpg'))

    if data_range != (0, 0):
        names = names[data_range[0]:data_range[1]]

    batch_count = len(names) // batch_size

    while True:
        for i in range(0, batch_count):
            files = np.array([imread(f) for f in names[i * batch_size:i * batch_size + batch_size]])

            if outType == 'YUV':
                yield color.rgb2yuv(files), files

            elif outType == 'LAB':
                yield color.rgb2lab(files), color.rgb2gray(files)[:, :, :, None]
