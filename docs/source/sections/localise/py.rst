.. _python:

Python Interface
================

The Python interface provides more flexibility and allows you to have more control 
over the training and prediction process. Here we provide examples of how to use the 
Python API to localise a target using pre-trained models.

Localise A Target Using Default models
--------------------------------------

The following examples demonstrates how to localise the Vim in **left** thalamus,
using the default model:

.. code-block:: python

    import numpy as np
    from localise.load import load_features
    from localise.predict import apply_model
    from localise.utils import save_nifti_4D

    # read the txt file that contains the paths to the subject folders
    with open('subjs.txt', 'r') as f:
        subj_list = [line.strip() for line in f]

    # path to the seed mask, relative to the subject directory
    mask_name = 'roi/left/tha.nii.gz'
    # path to the tract-density folder, relative to the subject directory
    target_path = 'tracts/left'
    # path to the group-average (atlas) map, as a prior feature
    atlas = 'roi/left/atlas.nii.gz'
    # whether to save the tract-density maps as a voxels x features matrix 
    # and stored as *.npy
    # Its path is relative to the subject directory
    output_fname = 'tracts/left/data.npy'

    # load tract-density features and save as 'tracts/left/data.npy' for each subject
    test_data = load_features(subject=subj_list, mask_name=mask_name, 
                              target_path=target_path, target_list=target_list, 
                              atlas=atlas, output_fname=output_fname)

    # make predictions
    predictions = apply_model(test_data, m)

    # save to nii
    for prediction, subject in zip(predictions, test_list):
        save_nifti_4D(prediction, os.path.join(subject, mask_name), os.path.join(subject, 'predictions'))


Localise A Target Using Custom models
-------------------------------------

.. code-block:: python

    import numpy as np
    from localise.load import load_features
    from localise.predict import apply_model

    # read the txt file that contains the paths to the subject folders
    with open('subjs.txt', 'r') as f:
        subj_list = [line.strip() for line in f]

    # path to the seed mask, relative to the subject directory
    mask_name = 'roi/left/tha.nii.gz'
    # path to the tract-density folder, relative to the subject directory
    target_path = 'tracts/left'
    # path to the group-average (atlas) map, as a prior feature
    atlas = 'roi/left/atlas.nii.gz'
    # whether to save the tract-density maps as a voxels x features matrix and stored as *.npy
    # Its path is relative to the subject directory
    output_fname = 'tracts/left/data.npy'

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
