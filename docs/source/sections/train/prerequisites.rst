.. _prerequisites-train:

Prerequisites for Custom Model Training
=======================================

Before you can train a custom model, 
ensure you have the necessary anatomical masks, tract-density maps. 
This data should have been prepared according to the :ref:`Prepare Your Data <prepare>` section.
Different from the :ref:`Predict Mode <prerequisites-predict>``, You will also need the the training labels.

Recommended Data Structure:
---------------------------

A logical and consistent organisation of your files is crucial for a smooth training process. 
Here's a suggested structure:

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

Details:
- ``tracts`` directory contains tract-density maps, categorised by hemisphere.
- ``roi`` directory stores anatomical masks, also sorted by hemisphere.
- ``training-labels`` directory holds the training labels for each hemisphere.

By following this structure, you'll ensure that the training process is straightforward and efficient.
