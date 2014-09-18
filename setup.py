#!/usr/bin/env python
import os

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='deeppy',
    version='0.1',
    author='Anders Boesen Lindbo Larsen',
    author_email='abll@dtu.dk',
    description="Deep learning in Python",
    license='MIT',
    url='http://compute.dtu.dk/~abll',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'cython', 'cudarray'],
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
    ext_modules = cythonize(['deeppy/ndarray/numpy/nnet/conv_bc01.pyx',
                             'deeppy/ndarray/numpy/nnet/pool_bc01.pyx']),
)
