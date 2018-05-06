import glob
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from abc import abstractmethod
from .utils import unpickle

CIFAR10_DATASET = 'cifar10'
PLACES365_DATASET = 'places365'


class BaseDataset():
    def __init__(self, name, path, training=True, augment=True):
        self.name = name
        self.augment = augment
        self.training = training
        self.path = path
        self._data = []

    def __len__(self):
        return len(self.data) * (2 if self.augment else 1)

    def __iter__(self):
        total = len(self)
        start = 0

        while start < total:
            item = self[start]
            start += 1
            yield item

        raise StopIteration

    def __getitem__(self, index):
        ix = int(index / 2)
        val = self.data[ix]
        img = imread(val) if isinstance(val, str) else val

        if index % 2 != 0:
            img = img[:, ::-1, :]

        return img

    def generator(self, batch_size):
        total = len(self)
        start = 0

        while start < total:
            end = np.min([start + batch_size, total])
            items = np.array([self[item] for item in range(start, end)])
            start = end
            yield items

        raise StopIteration


    @property
    def data(self):
        if len(self._data) == 0:
            self._data = self.load()
            np.random.shuffle(self._data)

        return self._data

    @abstractmethod
    def load(self):
        return []


class Cifar10Dataset(BaseDataset):
    def __init__(self, path, training=True, augment=True):
        super(Cifar10Dataset, self).__init__(CIFAR10_DATASET, path, training, augment)

    def load(self):
        data = []
        if self.training:
            for i in range(1, 6):
                filename = '{}/data_batch_{}'.format(self.path, i)
                batch_data = unpickle(filename)
                if len(data) > 0:
                    data = np.vstack((data, batch_data[b'data']))
                else:
                    data = batch_data[b'data']

        else:
            filename = '{}/test_batch'.format(self.path)
            batch_data = unpickle(filename)
            data = batch_data[b'data']

        w = 32
        h = 32
        s = w * h
        data = np.array(data)
        data = np.dstack((data[:, :s], data[:, s:2 * s], data[:, 2 * s:]))
        data = data.reshape((-1, w, h, 3))
        return data


class Places365Dataset(BaseDataset):
    def __init__(self, path, training=True, augment=True):
        super(Places365Dataset, self).__init__(PLACES365_DATASET, path, training, augment)

    def load(self):
        if self.training:
            data = np.array(
                #glob.glob(self.path + '/data_256/**/*.jpg', recursive=True))
                glob.glob(self.path + '/val_256/**/*.jpg', recursive=True))

        else:
            data = np.array(glob.glob(self.path + '/val_256/*.jpg'))

        return data
