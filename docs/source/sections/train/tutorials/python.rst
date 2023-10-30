.. _python-train:

Python Interface
================

Overview
--------

The Python interface offers a higher degree of flexibility in model training. 
This section details the process of training a model to localise Vim in the left thalamus. 
It emphasises the ability to use custom training labels 
which can be sourced from manual annotations or high-quality data segmentations.

Train a Model Using Custom Labels
---------------------------------

1. **Setup & Import Required Modules**

.. code-block:: python

    import numpy as np
    from localise.load import load_data, load_features, ShuffledDataloader
    from localise.train import train
    from localise.predict import apply_model

2. **Gather Subject Lists**

.. code-block:: python

    # Obtain a list of training subjects
    with open('train_subjs.txt', 'r') as f:
        train_list = [line.strip() for line in f]

    # Obtain a list of test subjects
    with open('test_subjs.txt', 'r') as f:
        test_list = [line.strip() for line in f]

3. **Configuration of Data Paths**

.. code-block:: python

    # Paths and filenames for seed mask, labels, tract-density, and atlas
    mask_name = 'roi/left/tha_small.nii.gz'
    label_name = 'high-quality-labels/left/labels.nii.gz'
    target_path = 'tracts/left'
    target_list = '/path/to/target_list.txt'
    atlas = 'roi/left/atlas.nii.gz'
    output_fname = 'tracts/left/data.npy'

4. **Data Loading**

.. code-block:: python

    # Load the training data
    train_data = load_data(subject=train_list, mask_name=mask_name, 
                           target_path=target_path, target_list=target_list, 
                           atlas=atlas, label_name=label_name, 
                           output_fname=output_fname)

    # Use ShuffledDataloader to shuffle the order of training subjects for each epoch
    train_dataloader = ShuffledDataloader(train_data)

    # Load the test data
    test_data = load_data(subject=test_list, mask_name=mask_name, 
                          target_path=target_path, target_list=target_list, 
                          atlas=atlas, label_name=label_name, 
                          output_fname=output_fname)

    test_dataloader = ShuffledDataloader(test_data)

5. **Training the Model**

.. code-block:: python

    # Define the path to save the trained model
    model_save_path = 'your_trained_model.pth'

    # Train the model and store in variable 'm'
    m = train(train_dataloader, test_dataloader, model_save_path=model_save_path)

6. **Making Predictions and Saving Results**

.. code-block:: python

    # Make predictions using the trained model
    predictions = apply_model(test_data, m)

    # Save the predictions in Nifti format
    for prediction, subject in zip(predictions, test_list):
        save_nifti_4D(prediction, os.path.join(subject, mask_name), os.path.join(subject, 'predictions'))
