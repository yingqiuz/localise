.. _python-predict:

Python Interface
================

Overview
--------

The Python interface offers enhanced flexibility for the user, 
enabling more granular control over both the training and prediction processes. 
This documentation provides step-by-step examples of how to utilise the Python API to 
localise a target with pre-trained and custom models.

Localize A Target Using Default Models
--------------------------------------

In this section, we will demonstrate how to localize the Vim in the **left** thalamus using the default pre-trained model.

1. **Setup & Import Necessary Modules**

.. code-block:: python

    import numpy as np
    from localise.load import load_features
    from localise.predict import apply_model
    from localise.utils import save_nifti_4D

2. **Prepare Data Paths and Configuration**

.. code-block:: python

    # Reading a txt file to get the paths of subject folders
    with open('subjs.txt', 'r') as f:
        subj_list = [line.strip() for line in f]

    # Configuration paths relative to the subject directory
    mask_name = 'roi/left/tha.nii.gz'
    target_path = 'tracts/left'
    atlas = 'roi/left/atlas.nii.gz'
    output_fname = 'tracts/left/data.npy'

3. **Load Features and Make Predictions**

.. code-block:: python

    # Load tract-density features for each subject
    test_data = load_features(subject=subj_list, mask_name=mask_name, 
                              target_path=target_path, target_list=target_list, 
                              atlas=atlas, output_fname=output_fname)

    # Use the pre-trained model to make predictions
    predictions = apply_model(test_data, m)

4. **Save Predictions to Nifti Format**

.. code-block:: python

    for prediction, subject in zip(predictions, test_list):
        save_nifti_4D(prediction, os.path.join(subject, mask_name), os.path.join(subject, 'predictions'))

Localize A Target Using Custom Models
-------------------------------------

1. **Setup & Import Necessary Modules**

.. code-block:: python

    import numpy as np
    from localise.load import load_features
    from localise.predict import apply_model

2. **Prepare Data Paths and Configuration**

.. code-block:: python

    # Reading a txt file to get the paths of subject folders
    with open('subjs.txt', 'r') as f:
        subj_list = [line.strip() for line in f]

    # Configuration paths relative to the subject directory
    mask_name = 'roi/left/tha.nii.gz'
    target_path = 'tracts/left'
    atlas = 'roi/left/atlas.nii.gz'
    output_fname = 'tracts/left/data.npy'

3. **Load Data, Shuffle and Train Custom Model**

.. code-block:: python

    # Load training data
    train_data = load_data(subject=train_list, mask_name=mask_name, 
                           target_path=target_path, target_list=target_list, 
                           atlas=atlas, label_name=label_name, 
                           output_fname=output_fname)

    # Shuffle the order of training subjects in each epoch
    train_dataloader = ShuffledDataloader(train_data)

    # Load test data
    test_data = load_data(subject=test_list, mask_name=mask_name, 
                          target_path=target_path, target_list=target_list, 
                          atlas=atlas, label_name=label_name, 
                          output_fname=output_fname)

    test_dataloader = ShuffledDataloader(test_data)

    # Define the path to save the trained model
    model_save_path = 'your_trained_model.pth'

    # Train the custom model
    m = train(train_dataloader, test_dataloader, model_save_path=model_save_path)

4. **Make Predictions and Save Results**

.. code-block:: python

    # Use the trained model to make predictions
    predictions = apply_model(test_data, m)

    # Save the predictions to Nifti format
    for prediction, subject in zip(predictions, test_list):
        save_nifti_4D(prediction, os.path.join(subject, mask_name), os.path.join(subject, 'predictions'))
