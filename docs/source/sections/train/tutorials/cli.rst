
.. _cli-train:

Command Line Interface
======================

Use the training mode when you want to train a custom model with a specific target list. 
Ensure you possess the required labels that serve as ground truth during the training process.

For a comprehensive list of available commands and options, enter:

.. code-block:: bash

    localise --help

Options
-------

Required arguments:
^^^^^^^^^^^^^^^^^^

- ``--train``:
    Train mode. Raise this flag if you want to train a model for your data.

- ``--label``:
    Path to the training labels of the structure, relative to the subject folder. 
    The file must be in the same space as the connectivity features (tract-density maps).

- ``--out_model``:
    Output filename of the trained model.

- ``-c, --hemisphere``:
    Hemisphere. Options include `left` and `right`.

- ``-s, --subject``:
    Path to the subject directory, or a txt file containing paths to subject folders.
    Example:
    
    .. code-block:: plaintext
    
       /path/to/subj001
       /path/to/subj002
       ...

- ``-i, --mask_dir``:
    Path to the folder containing anatomical masks, relative to the subject folder.
    For instance, if the subject folder is `/path/to/subject001` and the path to the
    tract-density maps is `/path/to/subject001/masks`, you need to provide
    `--target_dir=masks`.

- ``-x, --seed``:
    The binary seed mask (doesn't need the path). 
    This script assumes this binary seed mask can be found under the directory specified by mask_dir (``-i, --mask_dir``).
    For instance, if the subject folder is ``/path/to/subject001``, and the masks folder is ``/path/to/subject001/masks``,
    and the seed masks are 
    ``/path/to/subject001/masks/left/thlamus.nii.gz`` or ``/path/to/subject001/masks/right/thalamus.nii.gz``,
    depending on the hemisphere. 
    You only need to provide ``--seed=thalamus.nii.gz``.

- ``-p, --target_dir``:
    Path to the folder containing connectivity features (tract-density maps), 
    relative to the subject folder.
    For instance, if the subject folder is ``/path/to/subject001`` and the path to the
    tract-density maps is ``/path/to/subject001/tracts``, you need to provide
    ``--target_dir=tracts``.

Optional arguments:
^^^^^^^^^^^^^^^^^^

- ``-l, --target_list``: 
    A txt file containing streamline distribution files (tract-density maps).
    Example:

    .. code-block:: plaintext

       seeds_to_target1.nii.gz
       seeds_to_target2.nii.gz
       ...

    The script assumes that these tract-density can be found under the directory specified by target_dir (``-p, --target_dir``).
    e.g., ``/path/to/subject001/tracts/left`` and ``/path/to/subject001/tracts/right``, depending on the hemisphere.
    Must be specified if ``--data`` is not provided.

- ``-d, --data``:
    Presaved data file containing tract-density maps (doesn't need to include the path).
    This script assumes this data file can be found under the directory specified by target_dir (``-p, --target_dir``).
    e.g., ``/path/to/subject001/tracts/left/data.npy`` or ``/path/to/subject001/tracts/right/data.npy``, 
    depending on the hemisphere. Then you only need to provide ``--data=data.npy``.
    Must be specified if ``--target_list`` is not provided.

- ``-a, --atlas``:
    Path to the atlas (group-average probability map) of the structure to be localised, or `default`.
    This file should be in the same space as the tract-density maps (e.g., individual T1 space).
    If unspecified, the script will not use group-average probability maps as prior features.

- ``--spatial``:
    Use the conditional random field (recommended).

- ``-v, --verbose``:
    Increase output verbosity.

- ``-e, --epochs``:
    Number of epochs for training (default: 100).

Examples: Training a model
----------

This section guides you to train a model for your own target, 
which can be subsequently used to localise the target in new subjects.

For example, you want to train a model to localise Vim using your own data.
Your training labels are stored as ``training_labels_left.nii.gz`` and 
``training_labels_right.nii.gz`` under each training subject's folder.
Assume the file structure is as follows:

.. code-block:: console
    
    subject001/
    ├── tracts
    │   ├── left
    │       ├── seeds_to_target1.nii.gz
    │       ├── seeds_to_target2.nii.gz
    │       ├── ...
    │   ├── right
    │       ├── seeds_to_target1.nii.gz
    │       ├── seeds_to_target2.nii.gz
    │       ├── ...
    ├── masks
    │   ├── left
    │       ├── thalamus.nii.gz
    │       ├── ...
    │   ├── right
    │       ├── thalamus.nii.gz
    │       ├── ...
    ├── atlas_left.nii.gz
    ├── atlas_right.nii.gz
    ├── training_labels_left.nii.gz
    ├── training_labels_right.nii.gz
    └── otherfiles
    subject002/
    subject003/
    ...

And suppose the training subjects are listed in a file called ``training_subjs.txt``:

.. code-block:: plaintext

    /path/to/subject001
    /path/to/subject002
    /path/to/subject003
    ...

and the targets are listed in a file called ``your_target_list.txt``:

.. code-block:: plaintext

    seeds_to_target1.nii.gz
    seeds_to_target2.nii.gz
    ...

1. To train a model using subjects listed in ``training_subjs.txt``, and targets in ``your_target_list.txt``, 
   and subsequently save the trained model as ``your_trained_model.pth``, use the following:

   .. code-block:: bash

       localise --train --subject=training_subjs.txt --seed=roi/left/thalamus.nii.gz --hemisphere=left \ 
           --label=training_labels_left.nii.gz --mask_dir=masks --target_dir=tracts \ 
           --target_list=your_target_list.txt --out_model=your_trained_model.pth --spatial

   Once trained, your custom model (``your_trained_model.pth``) can be applied to new, unseen subjects, 
   as showcased in the :ref:`Example II: Localise A Target Using Your Own Pre-trained Models <prediction_mode>` section.

2. If you want to use your group-average probability map of the structure as a prior feature,
   use the following:

   .. code-block:: bash

       localise --train --subject=training_subjs.txt --seed=roi/left/thalamus.nii.gz --hemisphere=left \ 
           --label=training_labels_left.nii.gz --mask_dir=masks --target_dir=tracts \ 
           --target_list=your_target_list.txt --out_model=your_trained_model.pth --spatial \ 
           --atlas=atlas_left.nii.gz