import os
import logging
import numpy as np
from PIL import Image
from .util import download, checksum, archive_extract, checkpoint


log = logging.getLogger(__name__)

_URLS = {
    'original': (
        'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
        'a17d05bd522c52d84eca14327a23d494',
        # Checksum from website does not match.
        # 'ac79dc88658530a91423ebbba2b07bf3',
    ),
    'deepfunneled': (
        'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz',
        '68331da3eb755a505a502b5aacb3c201'
    ),
}


class LFW(object):
    '''
    The Labeled Faces in the Wild dataset [1].
    http://vis-www.cs.umass.edu/lfw/

    References:
    [1]: Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller.
         Labeled Faces in the Wild: A Database for Studying Face Recognition
         in Unconstrained Environments. University of Massachusetts, Amherst,
         Technical Report 07-49, October, 2007.
    '''

    def __init__(self, alignment='original', data_root='datasets'):
        if alignment not in ['original', 'deepfunneled']:
            raise ValueError('Invalid alignment: %s' % alignment)
        self.alignment = alignment
        self.name = 'lfw_' + alignment
        self.data_dir = os.path.join(data_root, self.name)
        self._npz_path = os.path.join(self.data_dir, self.name+'.npz')
        self._install()
        self.imgs, self.names_idx, self.names = self._load()

    def arrays(self):
        return self.imgs, self.names_idx, self.names

    def _install(self):
        checkpoint_file = os.path.join(self.data_dir, '__install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            url, md5 = _URLS[self.alignment]

            log.info('Downloading %s', url)
            filepath = download(url, self.data_dir)
            if md5 != checksum(filepath, method='md5'):
                raise RuntimeError('Checksum mismatch for %s.' % url)

            log.info('Unpacking %s', filepath)
            archive_extract(filepath, self.data_dir)

            log.info('Converting images to NumPy arrays')
            name_dict = {}
            imgs = []
            img_idx = 0
            for root, dirs, files in os.walk(self.data_dir):
                for filename in files:
                    _, ext = os.path.splitext(filename)
                    if ext.lower() != '.jpg':
                        continue
                    filepath = os.path.join(root, filename)
                    imgs.append(np.array(Image.open(filepath)))
                    _, name = os.path.split(root)
                    if name not in name_dict:
                        name_dict[name] = []
                    name_dict[name].append(img_idx)
                    img_idx += 1
            imgs = np.array(imgs)
            names = sorted(name_dict.keys())
            names_idx = np.empty(len(imgs))
            for name_idx, name in enumerate(names):
                for img_idx in name_dict[name]:
                    names_idx[img_idx] = name_idx
            with open(self._npz_path, 'wb') as f:
                np.savez(f, imgs=imgs, names_idx=names_idx, names=names)

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            return dic['imgs'], dic['names_idx'], dic['names']
