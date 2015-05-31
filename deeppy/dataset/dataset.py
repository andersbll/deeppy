import os
import shutil
import logging

from .util import checksum, download, archive_extract, is_archive, touch


logger = logging.getLogger(__name__)


class Dataset(object):
    name = None
    data_dir = None
    _download_checkpoint = '__download_complete'
    _unpack_checkpoint = '__unpack_complete'
    _install_checkpoint = '__install_complete'

    def __init__(self, data_root='datasets'):
        raise NotImplementedError()

    def data(self):
        raise NotImplementedError()

    def split(self):
        raise NotImplementedError()

    def _download(self, urls, sha1s=None):
        ''' Dowload dataset files given by the urls. If sha1s is given, the
        downloaded files are checked for correctness. '''
        checkpoint = os.path.join(self.data_dir, self._download_checkpoint)
        if os.path.exists(checkpoint):
            return
        if os.path.exists(self.data_dir):
            logger.info('Incomplete %s exists - restarting download.'
                        % self.data_dir)
            shutil.rmtree(self.data_dir)
            os.mkdir(self.data_dir)
        else:
            os.makedirs(self.data_dir)
        for i, url in enumerate(urls):
            logger.info('Downloading %s' % url)
            filepath = download(url, self.data_dir)
            if sha1s is not None:
                if sha1s[i] != checksum(filepath):
                    raise RuntimeError('SHA-1 checksum mismatch for %s.'
                                       % url)
        touch(checkpoint)

    def _unpack(self, separate_dirs=False):
        ''' Unpack all archive files in data_dir. '''
        checkpoint = os.path.join(self.data_dir, self._unpack_checkpoint)
        if os.path.exists(checkpoint):
            return
        to_be_removed = []
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            if is_archive(filepath):
                logger.info('Unpacking %s' % filepath)
                target_dir = os.path.abspath(self.data_dir)
                if separate_dirs:
                    dirname, _ = os.path.splitext(filename)
                    target_dir = os.path.join(target_dir, dirname)
                archive_extract(filepath, target_dir)
                to_be_removed.append(filepath)
        for filepath in to_be_removed:
            os.remove(filepath)
        touch(checkpoint)

    def _install(self):
        pass
