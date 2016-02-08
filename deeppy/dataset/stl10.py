import os
import numpy as np
import logging

from ..base import float_, int_
from .util import dataset_home, download, checksum, archive_extract, checkpoint


log = logging.getLogger(__name__)

_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
_SHA1 = 'b22ebbd7f3c4384ebc9ba3152939186d3750b902'


class STL10(object):
    '''
    The STL-10 dataset [1]
    http://cs.stanford.edu/~acoates/stl10

    References:
    [1]: An Analysis of Single Layer Networks in Unsupervised Feature Learning,
         Adam Coates, Honglak Lee, Andrew Y. Ng, AISTATS, 2011.
    '''

    def __init__(self):
        self.name = 'stl10'
        self.n_classes = 10
        self.n_train = 5000
        self.n_test = 8000
        self.n_unlabeled = 100000
        self.img_shape = (3, 96, 96)
        self.data_dir = os.path.join(dataset_home, self.name)
        self._npz_path = os.path.join(self.data_dir, 'stl10.npz')
        self._install()
        self._arrays, self.folds = self._load()

    def arrays(self, dp_dtypes=False):
        x_train, y_train, x_test, y_test, x_unlabeled = self._arrays
        if dp_dtypes:
            x_train = x_train.astype(float_)
            y_train = y_train.astype(int_)
            x_test = x_test.astype(float_)
            y_test = y_test.astype(int_)
            x_unlabeled = x_unlabeled.astype(float_)
        return x_train, y_train, x_test, y_test, x_unlabeled

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
            unpack_dir = os.path.join(self.data_dir, 'stl10_binary')
            log.info('Converting data to Numpy arrays')
            filenames = ['train_X.bin', 'train_y.bin', 'test_X.bin',
                         'test_y.bin', 'unlabeled_X.bin']

            def bin2numpy(filepath):
                with open(filepath, 'rb') as f:
                    arr = np.fromfile(f, dtype=np.uint8)
                    if '_X' in filepath:
                        arr = np.reshape(arr, (-1,) + self.img_shape)
                    return arr
            filepaths = [os.path.join(unpack_dir, f) for f in filenames]
            x_train, y_train, x_test, y_test, x_unlabeled = map(bin2numpy,
                                                                filepaths)
            folds = []
            with open(os.path.join(unpack_dir, 'fold_indices.txt'), 'r') as f:
                for line in f:
                    folds.append([int(s) for s in line.strip().split(' ')])
            folds = np.array(folds)
            with open(self._npz_path, 'wb') as f:
                np.savez(f, x_train=x_train, y_train=y_train, x_test=x_test,
                         y_test=y_test, x_unlabeled=x_unlabeled, folds=folds)

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            return ((dic['x_train'], dic['y_train'], dic['x_test'],
                     dic['y_test'], dic['x_unlabeled']), dic['folds'])
