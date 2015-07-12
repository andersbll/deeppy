import os
import numpy as np
import logging

from ..base import float_, int_
from .dataset import Dataset
from .util import touch, load_idx


log = logging.getLogger(__name__)

_URLS = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]

_SHA1S = [
    '6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d',
    '2a80914081dc54586dbdf242f9805a6b8d2a15fc',
    'c3a25af1f52dad7f726cce8cacb138654b760d48',
    '763e7fa3757d93b0cdec073cef058b2004252c17',
]


class MNIST(Dataset):
    '''
    THE MNIST DATABASE of handwritten digits [1]
    http://yann.lecun.com/exdb/mnist/

    References:
    [1]: Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
         learning applied to document recognition." Proceedings of the IEEE,
         86(11):2278-2324, November 1998
    '''

    def __init__(self, data_root='datasets'):
        self.name = 'mnist'
        self.data_dir = os.path.join(data_root, self.name)
        self._data_file = os.path.join(self.data_dir, 'mnist.npz')
        self.n_classes = 10
        self.n_test = 10000
        self.n_train = 60000
        self.img_shape = (28, 28)
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
            x_train = np.reshape(x_train, (self.n_train, -1))
            x_test = np.reshape(x_test, (self.n_test, -1))
        return x_train, y_train, x_test, y_test

    def _install(self):
        checkpoint = os.path.join(self.data_dir, self._install_checkpoint)
        if os.path.exists(checkpoint):
            return
        self._download(_URLS, _SHA1S)
        self._unpack()
        log.info('Converting MNIST data to Numpy arrays')
        filenames = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                     't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
        filenames = [os.path.join(self.data_dir, f) for f in filenames]
        x_train, y_train, x_test, y_test = map(load_idx, filenames)
        with open(self._data_file, 'wb') as f:
            np.savez(f, x_train=x_train, y_train=y_train, x_test=x_test,
                     y_test=y_test)
        touch(checkpoint)

    def _load(self):
        with open(self._data_file, 'rb') as f:
            dic = np.load(f)
            x_train = dic['x_train']
            y_train = dic['y_train']
            x_test = dic['x_test']
            y_test = dic['y_test']
        return x_train, y_train, x_test, y_test
