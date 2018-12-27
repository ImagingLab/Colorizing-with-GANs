import os
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
        self.augment = augment and training
        self.training = training
        self.path = path
        self._data = []

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        total = len(self)
        start = 0

        while start < total:
            item = self[start]
            start += 1
            yield item

        raise StopIteration

    def __getitem__(self, index):
        val = self.data[index]
        try:
            img = imread(val) if isinstance(val, str) else val

            # grayscale images
            if np.sum(img[:,:,0] - img[:,:,1]) == 0 and np.sum(img[:,:,0] - img[:,:,2]) == 0:
                return None

            if self.augment and np.random.binomial(1, 0.5) == 1:
                img = img[:, ::-1, :]

        except:
            img = None

        return img

    def generator(self, batch_size, recusrive=False):
        start = 0
        total = len(self)

        while True:
            while start < total:
                end = np.min([start + batch_size, total])
                items = []

                for ix in range(start, end):
                    item = self[ix]
                    if item is not None:
                        items.append(item)

                start = end
                yield items

            if recusrive:
                start = 0

            else:
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
            flist = os.path.join(self.path, 'train.flist')
            if os.path.exists(flist):
                data = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            else:
                data = glob.glob(self.path + '/data_256/**/*.jpg', recursive=True)
                np.savetxt(flist, data, fmt='%s')

        else:
            flist = os.path.join(self.path, 'test.flist')
            if os.path.exists(flist):
                data = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            else:
                data = np.array(glob.glob(self.path + '/val_256/*.jpg'))
                np.savetxt(flist, data, fmt='%s')

        return data


class TestDataset(BaseDataset):
    def __init__(self, path):
        super(TestDataset, self).__init__('TEST', path, training=False, augment=False)

    def __getitem__(self, index):
        path = self.data[index]
        img = imread(path)
        return path, img

    def load(self):

        if os.path.isfile(self.path):
            data = [self.path]

        elif os.path.isdir(self.path):
            data = list(glob.glob(self.path + '/*.jpg')) + list(glob.glob(self.path + '/*.png'))

        return data
