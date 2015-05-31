import os
import sys
import urllib
import tarfile
import zipfile
import gzip
import hashlib
import numpy as np
import struct
from subprocess import Popen


def touch(filepath, times=None):
    with open(filepath, 'a'):
        os.utime(filepath, times)


def download(url, target_dir, sha1=None, filename=None):
    filename = url.split('/')[-1].split('#')[0].split('?')[0]
    filepath = os.path.join(target_dir, filename)
    if sys.version_info[0] > 2:
        urllib.request.urlretrieve(url, filepath)
    else:
        urllib.urlretrieve(url, filepath)
    return filepath


def is_archive(filepath):
    return (tarfile.is_tarfile(filepath) or
            zipfile.is_zipfile(filepath) or
            filepath[-3:].lower() == '.gz' or
            filepath[-2:].lower() == '.z')


def archive_extract(filepath, target_dir):
    if tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, 'r') as tarf:
            # Check that no files get extracted outside target_dir
            for n in tarf.getnames():
                abs_path = os.path.abspath(os.path.join(target_dir, n))
                if not abs_path.startswith(target_dir):
                    raise RuntimeError('Archive tries to extract files '
                                       'outside target_dir.')
            tarf.extractall(target_dir)
    elif zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as zipf:
            zipf.extractall(target_dir)
    elif filepath[-3:].lower() == '.gz':
        with gzip.open(filepath, 'rb') as gzipf:
            with open(filepath[:-3], 'wb') as outf:
                outf.write(gzipf.read())
    elif filepath[-2:].lower() == '.z':
        if os.name != 'posix':
            raise NotImplementedError('Only Linux and Mac OS X support .Z '
                                      'compression.')
        filepath_out, _ = os.path.splitext(os.path.abspath(filepath))
        cmd = 'gzip -d %s' % filepath
        retval = Popen(cmd, shell=True).wait()
        if retval != 0:
            raise RuntimeError('Archive file extraction failed for %s.'
                               % filepath)
    else:
        raise ValueError('% is not a supported archive file.' % filepath)


def checksum(filename):
    data = open(filename, 'rb').read()
    return hashlib.sha1(data).hexdigest()


def _read_int(buf):
    return struct.unpack('>i', buf.read(4))[0]


def load_idx(filepath):
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
