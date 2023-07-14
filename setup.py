#!/usr/bin/env python

from setuptools import setup

with open('requirements.txt', 'rt') as f:
    install_requires = [line.strip() for line in f.readlines()]

with open('README.md', 'rt') as f:
    long_description = f.read()

setup(
    name='localise',
    version='0.1.0',  # Update the version number for new releases
    url='https://git.fmrib.ox.ac.uk/yqzheng1/python-localise',
    license='MIT',
    author='Ying-Qiu Zheng',
    author_email='ying-qiu.zheng@ndcn.ox.ac.uk',
    description='A Python package for localisation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['localise',],
    install_requires=install_requires,
    python_requires='>=3.7',
)
