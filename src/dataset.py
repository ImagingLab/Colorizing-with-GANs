import os
import abc
import glob
import numpy as np
from scipy.misc import imread
from skimage import color
from utils import unpickle, preprocess


class BaseDataset():
    def __init__(self, path, training=True, colorspace='LAB', flip=True):
        self.flip = flip
        self.colorspace = colorspace
        self.current = 0
        self.training = training
        self.path = path
        self._data = []

    def __len__(self):
        """
        Retunrs the length of the dataset
        """
        return len(self.data) * (2 if self.flip else 1)

    def __iter__(self):
        """
        Iterates over dataset items
        """
        while True:
            item = self[self.current]
            self.current += 1

            if self.current == len(self):
                self.current = 0

            yield item

    def __getitem__(self, index):
        """
        Retrieves an item from dataset by its index
        """
        ix = int(index / 2)
        val = self.data[ix]
        img = imread(val) if isinstance(val, str) else val

        if index % 2 != 0:
            img = img[:, ::-1, :]

        if self.colorspace == 'LAB':
            img = color.rgb2lab(img)

        return preprocess(img, self.colorspace)

    @property
    def data(self):
        if len(self._data) == 0:
            self._data = self.load()
            np.random.shuffle(self._data)

        return self._data

    @abc.abstractmethod
    def load(self):
        return []


class Cifar10Dataset(BaseDataset):
    def __init__(self, path, training=True, colorspace='LAB', flip=True):
        super(Cifar10Dataset, self).__init__(path, training, colorspace, flip)

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


class PlacesDataset(BaseDataset):
    def __init__(self, path, training=True, colorspace='LAB', flip=True):
        super(PlacesDataset, self).__init__(path, training, colorspace, flip)

    def load(self):
        if self.training:
            data = np.array(glob.glob(self.path + '/data_256/**/*.jpg', recursive=True))

        else:
            data = np.array(glob.glob(self.path + '/val_256/*.jpg'))

        return data
