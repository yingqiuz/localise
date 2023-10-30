.. _usage:

Usage
=====

.. _prepare:

Prepare Your Data
-----------------

To localise a surgical target, you need to generate connectivity features using tractography 
from your diffusion MRI data. 
If you haven't already done this, please refer to the following steps. 
In this section, we guide you through the process of generating connectivity features, 
which are essential for localising surgical targets using our toolbox.
If you have already got your connectivity data and are ready to train or apply a model, 
proceed to :ref:`Localise A Target <localise>` instead.

.. toctree::
   :maxdepth: 1

   prepare/prerequisites
   prepare/estimate
   prepare/create_masks
   prepare/create_tracts

Localise A Target (Predict Mode)
--------------------------------

If you haven't generated tract-density maps (i.e., connectivity features) required to
run the localise model, please follow the steps in :ref:`Prepare Your Data <prepare>`, 
otherwise you can proceed with the following steps to either localise a surgical target
using our pre-trained model, or a custom model tailored for your own data.

.. toctree::
   :maxdepth: 1

   localise/prerequisites
   localise/predict

Train A Custom Model For Your Data (Train Mode)
-----------------------------------------------

It happens that the pre-trained models released with the tool may not work well for your data,
or the structure you want to localise is not included in the pre-trained models.
In this case, you can train a custom model tailored for your own data.

.. toctree::
   :maxdepth: 1

   train/prerequisites
   train/train