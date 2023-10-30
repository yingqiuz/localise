.. _usage:

Usage
=====

Introduction
------------

Welcome to our toolbox usage guide! 
This document offers a roadmap on how to utilise the software, 
focusing on the primary tasks: data preparation, 
target localisation, and custom model training. 
Each step is detailed below.


.. _prepare:

Prepare Your Data
-----------------

To localise a surgical target, you need to generate connectivity features using tractography 
from your diffusion MRI data. 
This section offers a step-by-step guide to help you accomplish that. 
If you've already prepared your connectivity data and are poised to train or utilise a model, 
you can skip ahead to :ref:`Localise A Target <localise>`.

.. toctree::
   :maxdepth: 1

   prepare/prerequisites
   prepare/estimate
   prepare/create_masks
   prepare/create_tracts

.. _localise:

Localise A Target (Predict Mode)
--------------------------------

If you haven't generated tract-density maps (i.e., connectivity features) required to
run the localise model, please follow the steps in :ref:`Prepare Your Data <prepare>`, 
otherwise you can proceed with the following steps to either localise a surgical target
using our pre-trained model, or a custom model tailored for your own data.

**Steps to Localise a Target:**

- **Prerequisites**: Before starting the localisation, ensure you have the required setup and data. 
  This step outlines the essential preconditions.

- **Prediction**: Utilise the 'predict' mode for localisation. Depending on your needs, 
  you can use the toolbox's pre-trained model or a custom one you've trained.

.. toctree::
   :maxdepth: 1

   localise/prerequisites
   localise/predict

Train A Custom Model For Your Data (Train Mode)
-----------------------------------------------

Our toolbox comes with pre-trained models. 
However, there may be instances where these models aren't a perfect fit for your data 
or the specific structure you're aiming to localise. 
In such cases, our software allows you to train a custom model tailored to your needs.

**Steps to Train a Custom Model:**

- **Prerequisites**: Before diving into model training, understand the requirements and 
  ensure you have all the necessary data and configurations in place.

- **Training**: Dive deep into the training procedure, 
  setting parameters and training the model on your dataset. 
  This tailored model can then be used for precise target localisation.

.. toctree::
   :maxdepth: 1

   train/prerequisites
   train/train