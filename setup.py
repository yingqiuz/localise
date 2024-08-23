#!/usr/bin/env python


import os
from setuptools import setup, find_packages

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the requirements from the requirements file
with open(os.path.join(HERE, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

# Define package metadata
PACKAGE_NAME = 'localise'
DESCRIPTION = 'A Python package for localisation'
VERSION = '0.1.0'  # Consider using setuptools_scm for automatic versioning
AUTHOR = 'Ying-Qiu Zheng'
AUTHOR_EMAIL = 'ying-qiu.zheng@ndcn.ox.ac.uk'
URL = 'https://git.fmrib.ox.ac.uk/yqzheng1/python-localise'
LICENSE = 'MIT'
PYTHON_REQUIRES = '>=3.7'

# Define package contents and scripts
PACKAGES = find_packages(exclude=('tests',))
SCRIPTS = [
    'scripts/localise',
    'scripts/create_masks',
    'scripts/create_tracts',
    'scripts/connectivity_driven'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    scripts=SCRIPTS,
    install_requires=requirements,
    python_requires=PYTHON_REQUIRES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    project_urls={
        'Bug Reports': f'{URL}/issues',
        'Source': URL,
    },
)