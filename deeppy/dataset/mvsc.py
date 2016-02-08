import os
import glob
import shutil
import numpy as np
from PIL import Image
import logging

from .util import dataset_home, download, checksum, archive_extract, checkpoint


log = logging.getLogger(__name__)

_URLS = {
    'liberty': 'http://www.cs.ubc.ca/~mbrown/patchdata/liberty.zip',
    'notredame': 'http://www.cs.ubc.ca/~mbrown/patchdata/notredame.zip',
    'yosemite': 'http://www.cs.ubc.ca/~mbrown/patchdata/yosemite.zip',
}

_SHA1S = {
    'liberty': '18bffdb818146f5cba7fc6123d76f37d92c7ac20',
    'notredame': '1db5f3e7de6a03896210fa1cabb704b00a6fb07a',
    'yosemite': 'efeff1c654652e9b6e69b230a9286025dd2f57f8',
}


class MVSC(object):
    '''
    Multi-view Stereo Correspondence Dataset [1]
    http://cs.ubc.ca/~mbrown/patchdata/patchdata.html

    References:
    [1]: Brown, M.; Gang Hua; Winder, S., "Discriminative Learning of Local
         Image Descriptors," Pattern Analysis and Machine Intelligence,
         IEEE Transactions on , vol.33, no.1, pp.43,57, Jan. 2011
    '''

    def __init__(self, scene='liberty'):
        self.scene = scene
        self.name = 'mvsc_' + scene
        self.data_dir = os.path.join(dataset_home, self.name)
        self._npz_path = os.path.join(self.data_dir, scene+'.npz')
        self.img_shape = (64, 64)
        self._install()
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            self.patches = dic['patches']
            self.match_ids = dic['match_ids']
            self.ipoints = dic['ipoints']
            self.ref_img_ids = dic['ref_img_ids']
            self.similarities = dic['similarities']
            self.correspondences = dic['correspondences'][()]

    def _install(self):
        checkpoint_file = os.path.join(self.data_dir, '__install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            url = _URLS[self.scene]
            log.info('Downloading %s', url)
            filepath = download(url, self.data_dir)
            if _SHA1S[self.scene] != checksum(filepath, method='sha1'):
                raise RuntimeError('Checksum mismatch for %s.' % url)

            log.info('Unpacking %s', filepath)
            dirname, _ = os.path.splitext(os.path.split(filepath)[1])
            extract_dir = os.path.join(self.data_dir, dirname)
            archive_extract(filepath, extract_dir)

            log.info('Converting MVSC data to Numpy arrays')
            scene_dir = os.path.join(self.data_dir, self.scene)
            info_file = os.path.join(scene_dir, 'info.txt')
            match_ids = np.loadtxt(info_file, dtype=int)[:, 0]

            ipoints_file = os.path.join(scene_dir, 'interest.txt')
            ipoints = np.loadtxt(ipoints_file, dtype=float)
            ref_img_ids = ipoints[:, 0].astype(int)
            ipoints = ipoints[:, 1:]

            sim_file = os.path.join(scene_dir, 'sim.txt')
            similarities = np.loadtxt(sim_file, dtype=float)

            correspondence_files = [
                'm50_1000_1000_0.txt',
                'm50_2000_2000_0.txt',
                'm50_5000_5000_0.txt',
                'm50_10000_10000_0.txt',
                'm50_20000_20000_0.txt',
                'm50_50000_50000_0.txt',
                'm50_100000_100000_0.txt',
                'm50_200000_200000_0.txt',
                'm50_500000_500000_0.txt',
            ]
            correspondences = {}
            for f in correspondence_files:
                filename = os.path.join(scene_dir, f)
                if os.path.exists(filename):
                    corresp = np.loadtxt(filename, dtype=int)[:, [0, 1, 3, 4]]
                    correspondences[f] = corresp

            n_patches = match_ids.size
            patches = np.empty((n_patches,) + self.img_shape, dtype=np.uint8)
            patch_h, patch_w = self.img_shape
            tile_width = 1024
            patches_per_img = (tile_width//64)**2
            files = glob.glob(os.path.join(scene_dir, '*.bmp'))
            patch_idx = 0
            for f in sorted(files):
                img = np.array(Image.open(f))
                for i in range(patches_per_img):
                    if patch_idx == n_patches:
                        break
                    y = (i // (tile_width//patch_h)) * patch_h
                    x = (i % (tile_width//patch_h)) * patch_w
                    patches[patch_idx] = img[y:y+patch_h, x:x+patch_w]
                    patch_idx += 1
            if patch_idx != n_patches:
                raise RuntimeError('mismatching number of patches')
            with open(self._npz_path, 'wb') as f:
                np.savez(
                    f, match_ids=match_ids, patches=patches, ipoints=ipoints,
                    ref_img_ids=ref_img_ids, similarities=similarities,
                    correspondences=correspondences
                )
            shutil.rmtree(extract_dir)
