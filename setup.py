#!/usrbin/env python

from __future__ import print_function
from setuptools import setup, find_packages

setup(
    name='python-localise',
    version='0.1.0',  # Update the version number for new releases
    packages=find_packages(),
    url='https://github.com/yingqiuz/python-localise',
    license='MIT',
    author='Ying-Qiu Zheng',
    author_email='yingqiuz12fdu@gmail.com',
    description='A Python package for localisation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy>=1.15.0,<2.0.0',
        'torch>=1.0.0,<2.0.0',
        'torchvision>=0.5.0,<1.0.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)