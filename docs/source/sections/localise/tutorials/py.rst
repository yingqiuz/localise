.. _python-predict:

Python Interface
================

Overview
--------

The Python interface offers enhanced flexibility for the user, 
enabling more granular control over both the training and prediction processes. 
This documentation provides step-by-step examples of how to utilise the Python API to 
localise a target with pre-trained and custom models.

Localise A Target Using Default Models
--------------------------------------

In this section, we will demonstrate how to localise the Vim in the **left** thalamus using the default pre-trained model.
Again assume the file structure is organised as follows:

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
    subject002/
    subject003/
    ...

1. **Localising a target using default models.** For example, to localise Vim in the left thalamus
using the default pre-trained model, we can run the following command:

.. code-block:: python

    import numpy as np
    from localise.load import load_features
    from localise.predict import apply_model
    from localise.utils import save_nifti_4D
    from localise.mode import predict_mode

    # Configuration paths relative to the subject directory
    subjs = 'subjs.txt'
    mask_dir = 'masks'
    target_dir = 'tracts'
    seed = 'thalamus.nii.gz'
    hemisphere = 'left'
    # include the group-average probability map of vim, atlas_left.nii.gz, in the model
    atlas = 'atlas_left.nii.gz'
    out = 'vim_predictions_for_left.nii.gz'
    # for example, your data is single shell
    data_type = 'single32'

    predict_mode(subjs, seed, mask_dir=mask_dir, structure='vim', target_dir=target_dir,
                 atlas=atlas, out=out, spatial=True, 
                 data_type='single32', hemisphere=hemisphere, verbose=True)

2. **Localise a target using your own pre-trained model.** For example, you want to localise Vim
   in the left thalamus using your own pre-trained model ``/path/to/model.pth``, 
   you can run the following command:

.. code-block:: python

    # path to your pretrained model
    model = '/path/to/model.pth'
    # do not include the group-average probability map of vim in the model
    atlas = None

    predict_mode(subjs, seed, mask_dir=mask_dir, model=model, target_dir=target_dir,
                 atlas=atlas, out='vim_predictions_for_left.nii.gz', spatial=True, 
                 data_type='single32', hemisphere=hemisphere, verbose=True)

3. **Apply a model using python API.** If you want to have more control with the process
   instead of using the wrappers ``predict_mode`` or ``train_mode``, you can use the python API.
   Here we provide an example of how to apply a model using the python API.


.. code-block:: python

    import numpy as np
    from localise.load import load_data, load_features, ShuffledDataloader
    from localise.train import train, train_without_val
    from localise.predict import apply_model
    from localise.utils import save_nifti_4D

    # Reading a txt file to get the paths of subject folders
    # Obtain a list of training subjects
    with open('train_subjs.txt', 'r') as f:
        train_list = [line.strip() for line in f]

    # Obtain a list of test subjects
    with open('test_subjs.txt', 'r') as f:
        test_list = [line.strip() for line in f]

    # Configuration paths relative to the subject directory
    # path to the seed mask, relative to the subject folder
    mask_name = 'roi/left/tha.nii.gz'
    # the path to the tract-density maps, relative to the subject folder
    target_path = 'tracts/left'
    # include the group-average map as a prior feature, relative to the subject folder
    atlas = 'roi/left/atlas.nii.gz'
    # a txt file containing the names of targets, relative to the subject folder
    target_list = '/path/to/target_list.txt'
    # path to training labels, relative to subject folders
    label_name = 'high-quality-labels/left/labels.nii.gz'

    # The tract-density maps within the seed mask is saved as a matrix
    # and stored as output_fname
    output_fname = 'tracts/left/data.npy'

    # Load the training data
    train_data = load_data(subject=train_list, mask_name=mask_name,
                        target_path=target_path, target_list=target_list,
                        atlas=atlas, label_name=label_name,
                        output_fname=output_fname)

    # Use ShuffledDataloader to shuffle the order of training subjects for each epoch
    train_dataloader = ShuffledDataloader(train_data)

    # Train the model and store in variable 'm'
    m = train_without_val(train_dataloader, model_save_path='your_trained_model.pth')

    # Load tract-density features for each subject
    test_data = load_features(subject=subj_list, mask_name=mask_name, 
                              target_path=target_path, target_list=target_list, 
                              atlas=atlas, output_fname=output_fname)

    # Use the pre-trained model to make predictions
    predictions = apply_model(test_data, m)

    # Save Predictions to Nifti Format
    for prediction, subject in zip(predictions, test_list):
        save_nifti_4D(prediction, os.path.join(subject, mask_name), os.path.join(subject, 'predictions'))
