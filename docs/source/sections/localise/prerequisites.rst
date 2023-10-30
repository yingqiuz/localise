.. _prerequisites-predict:

Prerequisites for Data Organisation
===================================

For effective target localisation, you'll need tract-density maps corresponding to your chosen seed mask. 
For instance, if you aim to localise the ventral intermediate nucleus of the thalamus (Vim) within thalamic masks, 
ensure you have the necessary tract-density maps.

Consistent Data Organization:
-----------------------------

It's crucial to maintain a consistent file organization across subjects. 
Adopting this consistency simplifies further steps in the analysis pipeline. 
Below is a suggested file structure:

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

For example:

- Anatomical masks can be stored in directories like ``roi/left/`` and ``roi/right/`` for each subject.
- Tract-density maps can be placed in ``tracts/left`` and ``tracts/right``.

If you've prepared your masks and tract-density maps following the :ref:`Prepare Your Data <prepare>` guide, 
your files should already be in this consistent format.
