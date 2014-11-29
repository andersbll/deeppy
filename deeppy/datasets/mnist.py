import os
import struct
import numpy as np
import logging

from .dataset import Dataset
from .file_util import touch

logger = logging.getLogger(__name__)

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


def _read_int(buf):
    return struct.unpack('>i', buf.read(4))[0]


def _read_idx(filepath):
    with open(filepath, 'rb') as f:
        magic = _read_int(f)
        n = _read_int(f)
        if magic == 2051:
            height = _read_int(f)
            width = _read_int(f)
            shape = (n, height, width)
        elif magic == 2049:
            shape = n
        else:
            raise RuntimeError('could not parse header correctly')
        a = np.fromfile(f, dtype='B', count=np.prod(shape))
        a = np.reshape(a, shape)
    return a


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
        self.x, self.y = self._read()

    def data(self, flat=False):
        if flat:
            x = np.reshape(self.x, (self.n_train + self.n_test, -1))
        else:
            x = self.x
        return x, self.y

    def split(self):
        train_idx = np.arange(self.n_train)
        test_idx = np.arange(self.n_train, self.n_train+self.n_test)
        return train_idx, test_idx

    def _install(self):
        checkpoint = os.path.join(self.data_dir, self._install_checkpoint)
        if os.path.exists(checkpoint):
            return
        self._download(_URLS, _SHA1S)
        self._unpack()
        logger.info('Converting MNIST data to Numpy arrays')
        filenames = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                     't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
        filenames = [os.path.join(self.data_dir, f) for f in filenames]
        x_train, y_train, x_test, y_test = map(_read_idx, filenames)
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        if x.shape[0] != y.shape[0] != self.n_train + self.n_test:
            raise RuntimeError('dataset has invalid shape')
        with open(self._data_file, 'wb') as f:
            np.savez(f, x=x, y=y)
        touch(checkpoint)

    def _read(self):
        with open(self._data_file, 'rb') as f:
            dic = np.load(f)
            x = dic['x']
            y = dic['y']
        return x, y
