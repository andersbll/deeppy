#!/usr/bin/env python
import os
from setuptools import setup, find_packages
import deeppy


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='deeppy',
    version=deeppy.__version__,
    author='Anders Boesen Lindbo Larsen',
    author_email='abll@dtu.dk',
    description="Deep learning in Python",
    license='MIT',
    url='http://compute.dtu.dk/~abll',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'cudarray'],
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
