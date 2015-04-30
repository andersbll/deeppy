#!/usr/bin/env python

import os
import re
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f]


version = None
regex = re.compile(r'''^__version__ = ['"]([^'"]*)['"]''')
with open(os.path.join('deeppy', '__init__.py')) as f:
    for line in f:
        mo = regex.search(line)
        if mo is not None:
            version = mo.group(1)
            break
if version is None:
    raise RuntimeError('Could not find version number')


setup(
    name='deeppy',
    version=version,
    author='Anders Boesen Lindbo Larsen',
    author_email='abll@dtu.dk',
    description="Deep learning in Python",
    license='MIT',
    url='http://compute.dtu.dk/~abll',
    packages=find_packages(),
    install_requires=install_requires,
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
)
