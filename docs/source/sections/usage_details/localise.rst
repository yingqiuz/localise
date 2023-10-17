.. _localise:

Localise a target
=================

If you haven't generated tract-density maps (i.e., connectivity features) required to
run the localise model, please follow the steps in :ref:`Prepare your data <prepare>`, 
otherwise you can proceed with the following steps to either localise a surgical target
using our pre-trained model, or train a new model tailored for your own data.

- :ref:`prerequisites2`
- :ref:`cli`
- :ref:`python`

.. _prerequisites2:

Prerequisites
-------------

It is strongly recommended to organise the files for each subject consistently. 
For example, the anatomical masks were stored under ``roi/left/`` and ``roi/left/``
under the directory of each subject; the tract-density maps were stored under
``streamlines/left`` and ``streamlines/right``. 

.. code-block:: console
    
    subject001/
    ├── streamlines
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
    ├── high-quality-labels
    │   ├── left
    │       ├── labels.nii.gz
    │   ├── right
    │       ├── labels.nii.gz
    └── otherfiles
    subject002/
    subject003/

Orgaising the files consistently will simply the steps forward. 
If you have created your masks and tract-density maps by following 
:ref:`Prepare your data <prepare>`, then the files should have already been
organised consistently.

.. _cli:

Command Line Interface
----------------------

For a comprehensive list of available commands and options, enter:

.. code-block:: bash

    localise --help

You can find more information in the following subsections:

.. toctree::
    :maxdepth: 1

    prediction_mode
    train_mode

.. _python:

Python Interface
----------------

You can also train a model or localise a target using the python interface.

.. code-block:: python

    import numpy as np
    from localise.load import load_data, load_features, ShuffledDataloader
    from localise.train import train
    from localise.predict import apply_model

    with open('train_subjs.txt', 'r') as f:
        train_list = [line.strip() for line in f]

    with open('test_subjs.txt', 'r') as f:
        test_list = [line.strip() for line in f]

    mask_name = 'roi/left/tha_small.nii.gz'
    label_name = 'high-quality-labels/left/labels.nii.gz'
    target_path = 'streamlines/left'
    target_list = 'target_list.txt'
    atlas = 'roi/left/atlas.nii.gz'
    output_fname = 'streamlines/left/data.npy'

    # load training data
    train_data = load_data(subject=train_list, mask_name=mask_name, 
                        target_path=target_path, target_list=target_list, 
                        atlas=atlas, label_name=label_name, 
                        output_fname=output_fname)
    # ShuffedDataloader shuffles the order of training subjects in each epoch
    train_dataloarder = ShuffledDataloader(train_data)

    # load test data
    test_data = load_data(subject=test_list, mask_name=mask_name, 
                        target_path=target_path, target_list=target_list, 
                        atlas=atlas, label_name=label_name, 
                        output_fname=output_fname)

    test_dataloarder = ShuffledDataloader(test_data)
    #### training
    # the trained model is saved in out_model
    model_save_path = 'your_trained_model.pth'

    # train a model and store in m
    m = train(train_dataloarder, test_dataloader, model_save_path=model_save_path)

    # make predictions
    predictions = apply_model(test_data, m)

    # save to nii
    for prediction, subject in zip(predictions, test_list):
        save_nifti_4D(prediction, os.path.join(subject, mask_name), os.path.join(subject, 'predictions'))
