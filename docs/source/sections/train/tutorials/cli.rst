
.. _cli:

Command Line Interface
======================

Use the training mode when you want to train a custom model with a specific target list. 
Ensure you possess the required labels that serve as ground truth during the training process.

To train a model using subjects listed in ``training_subjs.txt``, and targets in ``your_target_list.txt``, 
and subsequently save the trained model as ``your_trained_model.pth``, use the following:

.. code-block:: bash

    localise --train --subject=training_subjs.txt --mask=roi/left/tha_small.nii.gz --label=high-quality-labels/left/labels.nii.gz --target_path=streamlines/left --target_list=your_target_list.txt --out_model=your_trained_model.pth --spatial

Once trained, your custom model (``your_trained_model.pth``) can be applied to new, unseen subjects, 
as showcased in the :ref:`Example II: Localise A Target Using Your Own Pre-trained Models <prediction_mode>` section.

Options
-------

- ``--train``:
    Train mode. Raise this flag if you want to train a model for your data, 
    or a target not included in the pre-trained models.

- ``--label``:
    Path to the training labels of the structure. The file must be in the same space as the connectivity features.

- ``--out_model``:
    Output filename of the trained model.

- ``-s, --subject``:
    Path to the subject directory, or a txt file containing paths to subject folders.
    Example:
    
    .. code-block:: plaintext
    
       /path/to/subj001
       /path/to/subj002
       ...

- ``-m, --mask``:
    Path to the binary seed mask, relative to the subject folder. For instance, if the subject folder is `/path/to/subject001` and the path to the binary mask is `/path/to/subject001/roi/left/tha.nii.gz`, provide `--mask=roi/left/tha.nii.gz`.

- ``-p, --target_path``:
    Path to the folder containing connectivity features, relative to the subject folder.

- ``-l, --target_list``:
    A txt file containing streamline distribution files (doesn't need to include the path).
    Example:

    .. code-block:: plaintext

       seeds_to_target1.nii.gz
       seeds_to_target2.nii.gz
       ...

- ``-a, --atlas``:
    Path to the atlas (group-average) probability map of the structure to be localised. 
    This file should be in the same space as the connectivity features (e.g., individual T1 space).

- ``--spatial``:
    Use the conditional random field (recommended).

- ``-v, --verbose``:
    Increase output verbosity.

- ``-e, --epochs``:
    Number of epochs for training (default: 100).