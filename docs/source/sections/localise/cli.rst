.. _cli:

Command Line Interface
----------------------

For a comprehensive list of available commands and options, enter:

.. code-block:: bash

    localise --help

Options
^^^^^^^

- ``--predict``:
    Prediction mode. Raise this flag if you want to localise a target using a pre-trained model.

- ``-r, --structure``:
    Structure to be localised (name of the structure when in prediction mode).

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

- ``-t, --data_type``:
    Data type or modality. Options include "single32", "resting-state" (to be inplemented), etc.

- ``-o, --out``:
    Output filename for the localised structure.

Example I: Localise A Target Using Default Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section guides you to localise a target using the pre-trained default model released with the package.
For example, you want to localise the ventral intermediate nucleus of thalamus (Vim) 
in the thalamus using the default model. 
Assume the filetree under ``subj001`` looks like this:

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
    │       ├── thalamus.nii.gz
    │       ├── atlas.nii.gz
    │       ├── vim.nii.gz
    │   ├── right
    │       ├── thalamus.nii.gz
    │       ├── atlas.nii.gz
    │       ├── vim.nii.gz
    └── otherfiles

To localise Vim in left hemisphere, you need to provide the following arguments:

.. code-block:: bash

    localise --predict --structure=vim --subject=subj1001 --mask=thalamus.nii.gz --target_path=tracts --data_type=single32 --spatial --out=predictions

This example assumes your tract-density maps follow the naming convention of the default target list the model was trained with. 
Default target lists are available under the package root directory ``resources/data``. 
For Vim models, refer to ``resources/data/vim_default_targets.txt``.

It will creates a folder ``predictions`` under ``subj001`` containing the localised structure 
``vim.nii.gz`` for left and right thalamus.

Example II: Localise A Target Using Your Own Pre-trained Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't want to use the pre-trained default model released with the package, 
and you've trained your own model, e.g., saved as ``/path/to/your_trained_model.pth`` 
using your own target list ``/path/to/your_target_list.txt`` 
(see :ref:`Train Mode <train_mode>` for how to train your own model), 
you can use the following:

.. code-block:: bash

    localise --predict --subject=subjs.txt --mask=thalamus.nii.gz --model=/path/to/your_trained_model.pth --target_path=tracts --target_list=/path/to/your_target_list.txt --spatial --out=predictions

This example will create a folder ``predictions`` for each subject in ``subjs.txt`` to store the localised structure，
using your pre-trained model ``/path/to/your_trained_model.pth`` and target list ``/path/to/your_target_list.txt``.