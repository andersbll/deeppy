import os
import shutil
import logging
import numpy as np
from PIL import Image
from .util import dataset_home, download, checksum, archive_extract, checkpoint


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

_PAIR_SPLIT_URLS = {
    'train': (
        'http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt',
        '4f27cbf15b2da4a85c1907eb4181ad21',
    ),
    'val': (
        'http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt',
        '5132f7440eb68cf58910c8a45a2ac10b',
    ),
    'test': (
        'http://vis-www.cs.umass.edu/lfw/pairs.txt',
        '9f1ba174e4e1c508ff7cdf10ac338a7d',
    ),
}

_PEOPLE_SPLIT_URLS = {
    'train': (
        'http://vis-www.cs.umass.edu/lfw/peopleDevTrain.txt',
        '54eaac34beb6d042ed3a7d883e247a21',
    ),
    'val': (
        'http://vis-www.cs.umass.edu/lfw/peopleDevTest.txt',
        'e4bf5be0a43b5dcd9dc5ccfcb8fb19c5',
    ),
    'test': (
        'http://vis-www.cs.umass.edu/lfw/people.txt',
        '450f0863dd89e85e73936a6d71a3474b',
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

    def __init__(self, alignment='original'):
        if alignment not in ['original', 'deepfunneled']:
            raise ValueError('Invalid alignment: %s' % alignment)
        self.alignment = alignment
        self.name = 'lfw_' + alignment
        self.data_dir = os.path.join(dataset_home, self.name)
        self._npz_path = os.path.join(self.data_dir, self.name+'.npz')
        self._install()
        self.people_splits = self._load()
        (self.imgs, self.index, self.people_splits,
         self.pair_splits) = self._load()

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
            extract_dir = os.path.splitext(os.path.split(filepath)[1])[0]
            extract_dir = os.path.join(self.data_dir, extract_dir)

            log.info('Converting images to NumPy arrays')
            index = {}
            imgs = []
            img_idx = 0
            for person_id in sorted(os.listdir(extract_dir)):
                person_dir = os.path.join(extract_dir, person_id)
                img_no = 0
                for filename in sorted(os.listdir(person_dir)):
                    if int(filename[-8:-4]) - 1 != img_no:
                        raise ValueError('Unexpected file: %s' % filename)
                    filepath = os.path.join(person_dir, filename)
                    imgs.append(np.array(Image.open(filepath)))
                    if person_id not in index:
                        index[person_id] = []
                    index[person_id].append(img_idx)
                    img_idx += 1
                    img_no += 1
            imgs = np.array(imgs)

            log.info('Downloading splits')
            people_splits = {}
            for split_name, (url, md5) in _PEOPLE_SPLIT_URLS.items():
                filepath = download(url, self.data_dir)
                if md5 != checksum(filepath, method='md5'):
                    raise RuntimeError('Checksum mismatch for %s.' % url)
                sets = []
                set_sizes = []
                current_set = -1
                with open(filepath, 'r') as f:
                    if 'test' in split_name:
                        f.readline()
                    for line in f:
                        fields = line.strip().split('\t')
                        if len(fields) == 2:
                            (name, n_imgs) = fields
                            sets[current_set].append(name)
                        else:
                            set_sizes.append(int(fields[0]))
                            sets.append([])
                            current_set += 1
                if set_sizes != [len(s) for s in sets]:
                    raise ValueError('Wrong # of images in sets.')
                if len(sets) == 1:
                    sets = sets[0]
                people_splits[split_name] = sets

            pair_splits = {}
            for split_name, (url, md5) in _PAIR_SPLIT_URLS.items():
                filepath = download(url, self.data_dir)
                if md5 != checksum(filepath, method='md5'):
                    raise RuntimeError('Checksum mismatch for %s.' % url)
                sets = []
                set_sizes = []
                current_set = -1
                with open(filepath, 'r') as f:
                    for line in f:
                        fields = line.strip().split('\t')
                        if len(fields) == 1:
                            set_sizes.append(int(fields[0])*2)
                            sets.append([])
                            continue
                        elif len(fields) == 2:
                            for _ in range(int(fields[0])):
                                set_sizes.append(int(fields[1])*2)
                                sets.append([])
                            current_set += 1
                            continue
                        elif len(fields) == 3:
                            (name0, img_idx0, img_idx1) = fields
                            name1 = name0
                        elif len(fields) == 4:
                            (name0, img_idx0, name1, img_idx1) = fields
                        sets[current_set].append((name0, int(img_idx0), name1,
                                                  int(img_idx1)))
                        if len(sets[current_set]) == set_sizes[current_set]:
                            current_set += 1
                if set_sizes != [len(s) for s in sets]:
                    raise ValueError('Wrong # of images in sets.')
                if len(sets) == 1:
                    sets = sets[0]
                pair_splits[split_name] = sets
            with open(self._npz_path, 'wb') as f:
                np.savez(f, imgs=imgs, index=index,
                         people_splits=people_splits, pair_splits=pair_splits)
            shutil.rmtree(extract_dir)

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            return (dic['imgs'], dic['index'][()], dic['people_splits'][()],
                    dic['pair_splits'][()])
