.. _cli-predict:

Command Line Interface
======================

For a comprehensive list of available commands and options, enter:

.. code-block:: bash

    localise --help

Options
-------

Required arguments:
^^^^^^^^^^^^^^^^^^^

- ``--predict``:
    Prediction mode. Raise this flag if you want to localise a target using a pre-trained model.

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

- ``-o, --out``:
    Output filename for the localised structure (relative to the subject folder).

- ``-c, --hemisphere``:
    Hemisphere. Options include `left` and `right`.

Optional arguments:
^^^^^^^^^^^^^^^^^^^

- ``-l, --target_list``: 
    A txt file containing streamline distribution files (tract-density maps).
    Example:

    .. code-block:: plaintext

       seeds_to_target1.nii.gz
       seeds_to_target2.nii.gz
       ...

    The script assumes that these tract-density can be found under the directory specified by target_dir (``-p, --target_dir``).
    e.g., ``/path/to/subject001/tracts/left`` and ``/path/to/subject001/tracts/right``, depending on the hemisphere.

- ``-d, --data``:
    Presaved data file containing tract-density maps (doesn't need to include the path).
    This script assumes this data file can be found under the directory specified by target_dir (``-p, --target_dir``).
    e.g., ``/path/to/subject001/tracts/left/data.npy`` or ``/path/to/subject001/tracts/right/data.npy``, 
    depending on the hemisphere. Then you only need to provide ``--data=data.npy``.

- ``-r, --structure``: 
    Structure to be localised (name of the structure when in prediction mode),
    using the default model.
    can be `vim`,...
    Must be specified if ``--model`` is not specified.

- ``-a, --atlas``:
    Path to the atlas (group-average probability map) of the structure to be localised, or `default`.
    This file should be in the same space as the tract-density maps (e.g., individual T1 space).
    If unspecified, the script will not use group-average probability maps as prior features.

- ``-t, --data_type``:
    Data type or modality. Options include "single32", "resting-state" (to be inplemented), etc.
    Must be specified if using default models (i.e., ``--model`` is not specified).

- ``--spatial``:
    Use the conditional random field (recommended).

- ``-v, --verbose``:
    Increase output verbosity.

Example I: Localise a target using default models
-------------------------------------------------

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
    │       ├── ...
    │   ├── right
    │       ├── seeds_to_target1.nii.gz
    │       ├── seeds_to_target2.nii.gz
    │       ├── ...
    ├── masks
    │   ├── left
    │       ├── thalamus.nii.gz
    │   ├── right
    │       ├── thalamus.nii.gz
    ├── atlas_left.nii.gz
    ├── atlas_right.nii.gz
    └── otherfiles
    ...

1. To localise Vim in left hemisphere without considering the group-average Vim probability map, 
    for single-shell diffusion data with 32 directions,
    you need to provide the following arguments:

    .. code-block:: bash

        localise --predict --structure=vim --subject=subj1001 --seed=thalamus.nii.gz \ 
            --mask_dir=masks --target_dir=tracts --out=left_predictions.nii.gz \
            --hemisphere=left --data_type=single32 --spatial 

    It will creates a folder ``left_predictions.nii.gz`` under ``subj001``,
    containing the predicted location of the structure.

2. To localise Vim in left hemisphere with `atlas_left.nii.gz` as a prior feature,
    and you don't want the spatial model,
    you need to provide the following arguments:

    .. code-block:: bash

        localise --predict --structure=vim --subject=subj1001 --seed=thalamus.nii.gz \ 
            --mask_dir=masks --target_dir=tracts --out=left_predictions.nii.gz \
            --hemisphere=left --data_type=single32 --atlas=atlas_left.nii.gz

    It will creates a folder ``left_predictions.nii.gz`` under ``subj001``,
    containing the predicted location of the structure.

3. If you have prepared your data by following the steps in :ref:`Prepare Your Data <prepare>`,
    You can also use the default prior feature:

    .. code-block:: bash

        localise --predict --structure=vim --subject=subj1001 --seed=thalamus.nii.gz \ 
            --mask_dir=masks --target_dir=tracts --out=right_predictions.nii.gz \
            --hemisphere=right --data_type=single32 --atlas=default

    It will creates a folder ``right_predictions.nii.gz`` under ``subj001``,
    containing the predicted location of the structure.

.. _prediction_mode:

Example II: Localise a target using your own pre-trained models
---------------------------------------------------------------

If you don't want to use the pre-trained default model released with the package, 
and you've trained your own model using your own target list
(see :ref:`Train Mode <train_mode>` for how to train your own model), 
check the following examples:

1. You've trained your own spatial model, e.g., saved as ``/path/to/your_trained_model.pth`` 
    using your own target list ``/path/to/your_target_list.txt``. 
    It doesn't use the atlas (group-average probability) map as a prior feature.
    You want to apply this model to new subjects in ``subjs.txt``, which contains the paths to subject folders.
    Then you can use the following:

    .. code-block:: bash

        localise --predict --subject=subjs.txt --seed=thalamus.nii.gz --mask_dir=masks \
            --target_dir=tracts --target_list=/path/to/your_target_list.txt --spatial --out=predictions.nii.gz \
            --hemisphere=left --model=/path/to/your_trained_model.pth

    This example will create a file ``predictions.nii.gz`` for each subject in ``subjs.txt`` to store the localised structure.

2. If your trained model used the group-average map as a prior feature (on training subjects),
    and for your test subjects, they have ``atlas_left.nii.gz`` as a prior feature,
    then you can use the following:

    .. code-block:: bash

        localise --predict --subject=subjs.txt --seed=thalamus.nii.gz --mask_dir=masks \
            --target_dir=tracts --target_list=/path/to/your_target_list.txt --atlas=atlas_left.nii.gz \
            --out=predictions.nii.gz --hemisphere=left --model=/path/to/your_trained_model.pth

    This example will create a file ``predictions.nii.gz`` for each subject in ``subjs.txt`` to store the localised structure.