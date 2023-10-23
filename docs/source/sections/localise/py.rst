.. _python:

Python Interface
================

The Python interface provides more flexibility and allows you to have more control 
over the training and prediction process. Here we provide examples of how to use the 
Python API to localise a target using pre-trained models.

Localise A Target Using Default models
--------------------------------------

.. code-block:: python

    import numpy as np
    from localise.load import load_data, load_features, ShuffledDataloader
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


Localise A Target Using Custom models
-------------------------------------

