import os
import logging
import numpy as np
from PIL import Image
from .util import dataset_home, download, checksum, archive_extract, checkpoint


log = logging.getLogger(__name__)


_ALIGNED_IMGS_URL = (
    'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1',
    'b7e1990e1f046969bd4e49c6d804b93cd9be1646'
)

_PARTITIONS_URL = (
    'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADxLE5t6HqyD8sQCmzWJRcHa/Eval/list_eval_partition.txt?dl=1',
    'fb3d89825c49a2d389601eacb10d73815fd3c52d'
)

_ATTRIBUTES_URL = (
    'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAC7-uCaJkmPmvLX2_P5qy0ga/Anno/list_attr_celeba.txt?dl=1',
    'da6959c54754838f1a12cbb80ed9baba5618eddd'
)


class CelebA(object):
    '''
    Large-scale CelebFaces Attributes (CelebA) Dataset [1].
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    References:
    [1]: Ziwei Liu, Ping Luo, Xiaogang Wang and Xiaoou Tang.
         Deep Learning Face Attributes in the Wild. Proceedings of
         International Conference on Computer Vision (ICCV), December, 2015.
    '''

    def __init__(self):
        self.name = 'celeba'
        self.n_imgs = 202599
        self.data_dir = os.path.join(dataset_home, self.name)
        self._npz_path = os.path.join(self.data_dir, self.name+'.npz')
        self.img_dir = os.path.join(self.data_dir, 'img_align_celeba')
        self._install()
        (self.train_idxs, self.val_idxs, self.test_idxs, self.attribute_names,
         self.attributes) = self._load()

    def _download(self, url, sha1):
        log.info('Downloading %s', url)
        filepath = download(url, self.data_dir)
        if sha1 != checksum(filepath):
            raise RuntimeError('Checksum mismatch for %s.' % url)
        return filepath

    def img(self, idx):
        img_path = os.path.join(self.img_dir, '%.6d.jpg' % (idx+1))
        return np.array(Image.open(img_path))

    def imgs(self):
        for i in range(self.n_imgs):
            yield self.img(i)

    def _install(self):
        checkpoint_file = os.path.join(self.data_dir, '__install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            url, md5 = _ALIGNED_IMGS_URL
            filepath = self._download(url, md5)
            log.info('Unpacking %s', filepath)
            archive_extract(filepath, self.data_dir)

            url, md5 = _PARTITIONS_URL
            filepath = self._download(url, md5)
            partitions = [[], [], []]
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    img_name, partition = line.strip().split(' ')
                    if int(img_name[:6]) != i + 1:
                        raise ValueError('Parse error.')
                    partition = int(partition)
                    partitions[partition].append(i)
            train_idxs, val_idxs, test_idxs = map(np.array, partitions)

            url, md5 = _ATTRIBUTES_URL
            filepath = self._download(url, md5)
            attributes = []
            with open(filepath, 'r') as f:
                f.readline()
                attribute_names = f.readline().strip().split(' ')
                for i, line in enumerate(f):
                    fields = line.strip().replace('  ', ' ').split(' ')
                    img_name = fields[0]
                    if int(img_name[:6]) != i + 1:
                        raise ValueError('Parse error.')
                    attr_vec = np.array(map(int, fields[1:]))
                    attributes.append(attr_vec)
            attributes = np.array(attributes)

            with open(self._npz_path, 'wb') as f:
                np.savez(f, train_idxs=train_idxs, val_idxs=val_idxs,
                         test_idxs=test_idxs, attribute_names=attribute_names,
                         attributes=attributes)

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            return (dic['train_idxs'], dic['val_idxs'], dic['test_idxs'],
                    dic['attribute_names'][()], dic['attributes'])
