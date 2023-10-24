.. _prerequisites2:

Prerequisites
=============

You will need the tract-density maps within the seed mask to localise the target of your choice.
For example, you may want to localise the ventral intermediate of the thalamus (Vim) within the thalamic masks.
It is strongly recommended to organise the files for each subject consistently. 
For example, the anatomical masks were stored under ``roi/left/`` and ``roi/left/``
under the directory of each subject; the tract-density maps were stored under
``tracts/left`` and ``tracts/right``. 

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
    └── otherfiles
    subject002/
    subject003/

Orgaising the files consistently will simply the steps forward. 
If you have created your masks and tract-density maps by following 
:ref:`Prepare Your Data <prepare>`, then the necessary files should have already been
organised consistently.