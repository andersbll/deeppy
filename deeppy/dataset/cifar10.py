import os
import pickle
import numpy as np
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
        self.x, self.y = self._load()

    def data(self):
        return self.x, self.y

    def split(self, n_valid=0):
        train_idx = np.arange(self.n_train)
        test_idx = np.arange(self.n_train, self.n_train+self.n_test)
        return train_idx, test_idx

    def _install(self):
        self._download(_URLS, _SHA1S)
        self._unpack()

    def _load(self):
        dirpath = os.path.join(self.data_dir, 'cifar-10-batches-py')
        filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                     'data_batch_4', 'data_batch_5', 'test_batch']
        xs = []
        ys = []
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'rb') as f:
                dic = pickle.load(f)
                xs.append(dic['data'])
                ys.append(dic['labels'])
        x = np.vstack(xs)
        y = np.hstack(ys)
        x = np.reshape(x, (self.n_train + self.n_test,) + self.img_shape)
        if x.shape[0] != y.shape[0] != self.n_train + self.n_test:
            raise RuntimeError('dataset has invalid shape')
        return x, y
