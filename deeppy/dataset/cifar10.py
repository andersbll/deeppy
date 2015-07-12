import os
import pickle
import numpy as np
from ..base import float_, int_
from .dataset import Dataset


_URLS = [
    'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
]

_SHA1S = [
    '874905e36347c8536514d0a26261acf3bff89bc7',
]


class CIFAR10(Dataset):
    '''
    The CIFAR-10 dataset [1]
    http://www.cs.toronto.edu/~kriz/cifar.html

    References:
    [1]: Learning Multiple Layers of Features from Tiny Images, Alex
         Krizhevsky, 2009.
    '''

    def __init__(self, data_root='datasets'):
        self.name = 'cifar10'
        self.n_classes = 10
        self.n_test = 10000
        self.n_train = 50000
        self.img_shape = (3, 32, 32)
        self.data_dir = os.path.join(data_root, self.name)
        self._install()
        self._data = self._load()

    def data(self, flat=False, dp_dtypes=False):
        x_train, y_train, x_test, y_test = self._data
        if dp_dtypes:
            x_train = x_train.astype(float_)
            y_train = y_train.astype(int_)
            x_test = x_test.astype(float_)
            y_test = y_test.astype(int_)
        if flat:
            x_train = np.reshape(x_train, (x_train.shape[0], -1))
            x_test = np.reshape(x_test, (x_test.shape[0], -1))
        return x_train, y_train, x_test, y_test

    def _install(self):
        self._download(_URLS, _SHA1S)
        self._unpack()

    def _load(self):
        dirpath = os.path.join(self.data_dir, 'cifar-10-batches-py')
        filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                     'data_batch_4', 'data_batch_5', 'test_batch']
        x = []
        y = []
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'rb') as f:
                dic = pickle.load(f)
                x.append(dic['data'])
                y.append(dic['labels'])
        x_train = np.vstack(x[:5])
        y_train = np.hstack(y[:5])
        x_test = np.array(x[5])
        y_test = np.array(y[5])
        x_train = np.reshape(x_train, (self.n_train,) + self.img_shape)
        x_test = np.reshape(x_test, (self.n_test,) + self.img_shape)
        return x_train, y_train, x_test, y_test
