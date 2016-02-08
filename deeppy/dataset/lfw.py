import os
import shutil
import logging
import numpy as np
import scipy.io
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

_ATTRIBUTES_URL = (
    'http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt',
    'a68c039ac854fd966fe4bdc36674b98f'
)

_LANDMARKS_URL = (
    'http://personal.ie.cuhk.edu.hk/~zs014/denseLFW_v2.tar.gz',
    '44fea754b93a5e678147427b8b4e7ff1'
)


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
        (self.imgs, self.index, self.people_splits,
         self.pair_splits) = self._load()
        self._attributes_path = os.path.join(self.data_dir, 'attributes.npz')
        self._attributes = None
        self._attribute_names = None
        self._landmarks_path = os.path.join(self.data_dir, 'landmarks.npz')
        self._landmarks68 = None
        self._landmarks194 = None

    def _download(self, url, md5):
        log.info('Downloading %s', url)
        filepath = download(url, self.data_dir)
        if md5 != checksum(filepath, method='md5'):
            raise RuntimeError('Checksum mismatch for %s.' % url)
        return filepath

    @property
    def attribute_names(self):
        if self._attribute_names is None:
            self._load_attributes()
        return self._attribute_names

    @property
    def attributes(self):
        if self._attributes is None:
            self._load_attributes()
        return self._attributes

    def landmarks(self, kind='68'):
        if self._landmarks68 is None:
            self._load_landmarks()
        if kind == '68':
            return self._landmarks68
        if kind == '194':
            return self._landmarks194

    def _install(self):
        checkpoint_file = os.path.join(self.data_dir, '__install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            url, md5 = _URLS[self.alignment]
            filepath = self._download(url, md5)
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
                img_no = 1
                for filename in sorted(os.listdir(person_dir)):
                    if int(filename[-8:-4]) != img_no:
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
                filepath = self._download(url, md5)
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
                filepath = self._download(url, md5)
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

    def _install_attributes(self):
        checkpoint_file = os.path.join(self.data_dir, '__attr_install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            url, md5 = _ATTRIBUTES_URL
            filepath = self._download(url, md5)

            attributes = {}
            with open(filepath, 'r') as f:
                line = f.readline()
                line = f.readline()
                fields = [s.strip() for s in line.split('\t')]
                attribute_names = [s.strip() for s in line.split('\t')[3:]]
                i = 0
                for line in f:
                    i += 1
                    fields = [s.strip() for s in line.split('\t')]
                    person_id = fields[0].replace(' ', '_')
                    img_no = int(fields[1].replace(' ', '_'))
                    attr_vec = np.array([float(s) for s in fields[2:]])
                    if person_id not in attributes:
                        attributes[person_id] = {}
                    attributes[person_id][img_no] = attr_vec

            with open(self._attributes_path, 'wb') as f:
                np.savez(f, attributes=attributes,
                         attribute_names=attribute_names)

    def _load_attributes(self):
        self._install_attributes()
        with open(self._attributes_path, 'rb') as f:
            dic = np.load(f)
            self._attributes = dic['attributes'][()]
            self._attribute_names = dic['attribute_names'][()]

    def _install_landmarks(self):
        checkpoint_file = os.path.join(self.data_dir, '__landm_install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            url, md5 = _LANDMARKS_URL
            filepath = self._download(url, md5)
            log.info('Unpacking %s', filepath)
            archive_extract(filepath, self.data_dir)
            mat_path = os.path.join(self.data_dir, 'denseLFW_v2.mat')
            matdict = scipy.io.loadmat(mat_path)
            img_paths = [str(s[0][0]) for s in matdict['nameList']]
            landmarks68 = np.zeros_like(matdict['pose68'])
            landmarks194 = np.zeros_like(matdict['pose194'])
            for i, img_path in enumerate(img_paths):
                person_id = img_path.split('/')[0]
                if person_id not in self.index.keys():
                    raise ValueError('invalid person_id')
                img_no = int(img_path[-8:-4])
                img_idx = self.index[person_id][img_no-1]
                landmarks68[img_idx] = matdict['pose68'][i]
                landmarks194[img_idx] = matdict['pose194'][i]
            with open(self._landmarks_path, 'wb') as f:
                np.savez(f, landmarks68=landmarks68, landmarks194=landmarks194)

    def _load_landmarks(self):
        self._install_landmarks()
        with open(self._landmarks_path, 'rb') as f:
            dic = np.load(f)
            self._landmarks68 = dic['landmarks68']
            self._landmarks194 = dic['landmarks194']
