#!/usr/bin/env python

import os
import re
from setuptools import setup, find_packages, Command
from setuptools.command.test import test as TestCommand


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


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import subprocess
        subprocess.call(['py.test'] + self.pytest_args + ['test'])


class Coverage(Command):
    description = 'Generate a test coverage report.'
    user_options = [('report=', 'r', 'Report type (report/html)')]

    def initialize_options(self):
        self.report = 'report'

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        subprocess.call(['coverage', 'run', '--source=deeppy', '-m', 'py.test',
                         'test'])
        subprocess.call(['coverage', self.report])


setup(
    name='deeppy',
    version=version,
    author='Anders Boesen Lindbo Larsen',
    author_email='abll@dtu.dk',
    description='Deep learning in Python',
    license='MIT',
    url='http://compute.dtu.dk/~abll',
    packages=find_packages(exclude=['doc', 'examples', 'test']),
    install_requires=install_requires,
    long_description=read('README.md'),
    cmdclass={
        'test': PyTest,
        'coverage': Coverage,
    },
    extras_require={
        'test': ['pytest', 'sklearn'],
        'coverage': ['pytest', 'sklearn', 'coverage'],
    },
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
