#! /usr/bin/env python
"""Setup img_pipe."""

import os
from setuptools import setup, find_packages

# get the version
version = None
with open(os.path.join('img_pipe', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


setup(name='img_pipe',
      maintaier='Liberty Hamilton',
      maintainer_email='libertyhamilton@gmail.com',
      description='Image processing pipeline for localization and '
                  'identification of electrodes for electrocorticography',
      license='BSD (3-clause)',
      url='https://github.com/ChangLabUcsf/img_pipe',
      version=version,
      download_url='https://github.com/ChangLabUcsf/img_pipe.git',
      long_description=open('README.md').read(),
      python_requires='~=3.7',
      packages=find_packages(),
      platforms='any',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved',
          'Programming Language :: Python',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Scientific/Engineering :: Medical Science Apps'
      ]
      )
