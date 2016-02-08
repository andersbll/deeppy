import os
import numpy as np
import logging
from subprocess import Popen

from ..base import float_, int_
from .util import (dataset_home, download, checksum, archive_extract,
                   checkpoint, load_idx)


log = logging.getLogger(__name__)

_URL = 'http://leon.bottou.org/_media/projects/infimnist.tar.gz'
_SHA1 = '7044bf0e85e19dfdc422c3f5dba37348bf6a33ee'


class InfiMNIST(object):
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

    def __init__(self):
        self.name = 'infimnist'
        self.data_dir = os.path.join(dataset_home, self.name)
        self._npz_path = os.path.join(self.data_dir, 'infimnist.npz')
        self.n_classes = 10
        self.img_shape = (28, 28)
        self._infimnist_start = 10000
        self._infimnist_stop = 8109999
        self._install()
        self._arrays = self._load()

    def split(self, n_val=10000):
        infimnist_idxs = np.arange(self._infimnist_start, self._infimnist_stop)
        # Map InfiMNIST indices to MNIST indices according to infimnist.c
        testnum = 10000
        trainnum = 60000
        mnist_idxs = (infimnist_idxs - testnum) % trainnum
        # Use the last n_val digits for validation
        train_idx = mnist_idxs < (trainnum - n_val)
        val_idx = np.logical_not(train_idx)
        return train_idx, val_idx

    def arrays(self, flat=False, dp_dtypes=False):
        x, y = self._arrays
        if dp_dtypes:
            x = x.astype(float_)
            y = y.astype(int_)
        if flat:
            x = np.reshape(x, (x.shape[0], -1))
        return x, y

    def _install(self):
        checkpoint_file = os.path.join(self.data_dir, '__install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            log.info('Downloading %s', _URL)
            filepath = download(_URL, self.data_dir)
            if _SHA1 != checksum(filepath, method='sha1'):
                raise RuntimeError('Checksum mismatch for %s.' % _URL)

            log.info('Unpacking %s', filepath)
            archive_extract(filepath, self.data_dir)

            log.info('Building executable')
            cwd = os.path.join(self.data_dir, 'infimnist')
            if os.name == 'posix':
                Popen('make', cwd=cwd).wait()
            else:
                Popen(['nmake', '/f', 'NMakefile'], cwd=cwd).wait()

            log.info('Generating InfiMNIST dataset')
            lab_file = os.path.join(cwd, 'mnist8m-labels-idx1-ubyte')
            pat_file = os.path.join(cwd, 'mnist8m-patterns-idx3-ubyte')
            with open(lab_file, 'wb') as out:
                Popen(['./infimnist', 'lab', str(self._infimnist_start),
                       str(self._infimnist_stop)], stdout=out,
                      cwd=cwd).wait()
            with open(pat_file, 'wb') as out:
                Popen(['./infimnist', 'pat', str(self._infimnist_start),
                       str(self._infimnist_stop)], stdout=out,
                      cwd=cwd).wait()

            log.info('Converting InfiMNIST data to Numpy arrays')
            x, y = map(load_idx, [pat_file, lab_file])
            if x.shape[0] != y.shape[0]:
                raise RuntimeError('dataset has invalid shape')
            with open(self._npz_path, 'wb') as f:
                np.savez(f, x=x, y=y)

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            x = dic['x']
            y = dic['y']
        return x, y
