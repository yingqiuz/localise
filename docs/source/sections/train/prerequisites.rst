.. _prerequisites2:

Prerequisites
=============

To train a custom model, you will need the anatomical masks 
and tract-density maps as created in :ref:`Prepare Your Data <prepare>`, 
as well as the training labels.
For example, your files could be organised as follows:

.. code-block:: console
    
    subject001/
    ├── tracts
    │   ├── left
    │       ├── seeds_to_target1.nii.gz
    │       ├── seeds_to_target2.nii.gz
    │   ├── right
    │       ├── seeds_to_target1.nii.gz
    │       ├── seeds_to_target2.nii.gz
    ├── roi
    │   ├── left
    │       ├── tha.nii.gz
    │       ├── atlas.nii.gz
    │   ├── right
    │       ├── tha.nii.gz
    │       ├── atlas.nii.gz
    ├── training-labels
    │   ├── left
    │       ├── labels.nii.gz
    │   ├── right
    │       ├── labels.nii.gz
    └── otherfiles
    subject002/
    subject003/