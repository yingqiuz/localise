.. _usage:

Usage
=====

.. _installation:

Installation
------------

To use Localise, first install it using pip:

.. code-block:: console

   (.venv) $ git clone https://git.fmrib.ox.ac.uk/yqzheng1/python-localise.git
   (.venv) $ cd python-localise 
   (.venv) $ pip install .

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

   prepare/prerequisites.rst
   prepare/estimate.rst
   prepare/create_masks.rst
   prepare/create_tracts.rst
