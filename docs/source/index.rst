.. Localise documentation master file, created by
   sphinx-quickstart on Sun Oct  8 17:56:04 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Localise's documentation!
====================================

**LOCALISE** is a Python library for segmentating prevalent surgical targets using multimodal MRI. 
This toolbox uses Image Quality Transfer techniques to transfer anatomical information from high-quality 
data to a wide range of connectivity features in low-quality data. 
The goal is to augment the inference on DBS targets localisation even with compromised data quality.

At present, the toolbox supports:

- Localising the Vim nucleus of the thalamus, given the T1 and the diffusion MRI data.

- Training a model for localising a new target, given the T1 and the diffusion MRI data.

Future releases will include:

- Localising the STN, ACC, and other targets...

.. note::

   This project is under active development, and the API might still change substantially at any time!

How to Get Started
------------------

1. If the toolbox is not installed, following the :ref:`Installation Instructions <installation>`.

2. If you are not sure what data you need and how to prepare them, 
   check out the :ref:`Prepare Your Data <prepare>`.

3. If you simply want to localise a target using pre-trained models, 
   check out the :ref:`Predict Mode <predict_mode>`. 
   Look through the tutorials.

4. If the toolbox doesn't include the target you want to localise, 
   and you want to train a custom model tailored for your own data, 
   check out the :ref:`Train Mode <train_mode>`.

5. For more information about the toolbox, 
   check out `our paper <https://link.springer.com/chapter/10.1007/978-3-031-43996-4_17>`_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   sections/installation
   sections/usage
   sections/api
   sections/ref

Contributors
============

The original toolbox was written by Ying-Qiu Zheng.

Other Contributors:

- Saad Jbabdi