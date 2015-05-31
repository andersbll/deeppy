import os
import numpy as np
import logging
from subprocess import Popen

from .dataset import Dataset
from .util import touch, load_idx


logger = logging.getLogger(__name__)

_URLS = [
    'http://leon.bottou.org/_media/projects/infimnist.tar.gz',
]

_SHA1S = [
    '7044bf0e85e19dfdc422c3f5dba37348bf6a33ee',
]


class InfiMNIST(Dataset):
    '''
    The infinite MNIST dataset (formerly known as MNIST8M) [1, 2]
    http://leon.bottou.org/projects/infimnist

    References:
    [1]: Loosli, Gaelle, Stephane Canu, and Leon Bottou. "Training invariant
         support vector machines using selective sampling." Large scale kernel
         machines (2007): 301-320.
    [2]: Simard, Patrice, et al. "Tangent prop - a formalism for specifying
         selected invariances in an adaptive network." Advances in neural
         information processing systems. 1992.
    '''

    def __init__(self, data_root='datasets'):
        self.name = 'infimnist'
        self.data_dir = os.path.join(data_root, self.name)
        self._data_file = os.path.join(self.data_dir, 'infimnist.npz')
        self.n_classes = 10
        self.img_shape = (28, 28)
        self._install()
        self.x, self.y = self._load()
        self.n_samples = self.x.shape[0]

    def data(self, flat=False):
        if flat:
            x = np.reshape(self.x, (self.n_samples, -1))
        else:
            x = self.x
        return x, self.y

    def _install(self):
        checkpoint = os.path.join(self.data_dir, self._install_checkpoint)
        if os.path.exists(checkpoint):
            return
        self._download(_URLS, _SHA1S)
        self._unpack()

        logger.info('Building executable')
        cwd = os.path.join(self.data_dir, 'infimnist')
        if os.name == 'posix':
            Popen('make', cwd=cwd).wait()
        else:
            Popen(['nmake', '/f', 'NMakefile'], cwd=cwd).wait()

        logger.info('Generating InfiMNIST dataset')
        lab_file = os.path.join(cwd, 'mnist8m-labels-idx1-ubyte')
        pat_file = os.path.join(cwd, 'mnist8m-patterns-idx3-ubyte')
        with open(lab_file, 'wb') as out:
            Popen(['./infimnist', 'lab', '10000', '8109999'], stdout=out,
                  cwd=cwd).wait()
        with open(pat_file, 'wb') as out:
            Popen(['./infimnist', 'pat', '10000', '8109999'], stdout=out,
                  cwd=cwd).wait()

        logger.info('Converting InfiMNIST data to Numpy arrays')
        x, y = map(load_idx, [pat_file, lab_file])
        if x.shape[0] != y.shape[0]:
            raise RuntimeError('dataset has invalid shape')
        with open(self._data_file, 'wb') as f:
            np.savez(f, x=x, y=y)
        touch(checkpoint)

    def _load(self):
        with open(self._data_file, 'rb') as f:
            dic = np.load(f)
            x = dic['x']
            y = dic['y']
        return x, y
