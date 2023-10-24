.. _python:

Python Interface
================

The Python interface provides more flexibility in training a model.
In the following example, we show how to train a model for localising Vim
in the left thalamus, using users' own training labels 
(either from manual annotation or from high-quality data segmentation).

.. code-block:: python

    import numpy as np
    from localise.load import load_data, load_features, ShuffledDataloader
    from localise.train import train
    from localise.predict import apply_model

    # list of training subjects
    with open('train_subjs.txt', 'r') as f:
        train_list = [line.strip() for line in f]

    # list of test subjects
    with open('test_subjs.txt', 'r') as f:
        test_list = [line.strip() for line in f]

    # seed mask
    mask_name = 'roi/left/tha_small.nii.gz'
    # high-quality labels
    label_name = 'high-quality-labels/left/labels.nii.gz'
    # directory of tract-density maps (left hemisphere)
    target_path = 'tracts/left'
    # list of tract-density maps, chosen by the user
    target_list = '/path/to/target_list.txt'
    # group-average probability map of the target structure
    atlas = 'roi/left/atlas.nii.gz'
    # save the tract density maps as a matrix, stored as data.npy
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
