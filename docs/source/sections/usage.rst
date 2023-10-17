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

Detailed Usage
-------------------
To localise a surgical target, you need to generate connectivity features using tractography 
from your diffusion MRI data. 
If you haven't already done this, please refer to the steps in :ref:`Prepare your data <prepare>`.
If you have already got your connectivity data and are ready to train or apply a model, 
proceed to :ref:`Localise a target <localise>`.

.. toctree::
   :maxdepth: 1

   usage_details/prepare.rst
   usage_details/localise.rst
