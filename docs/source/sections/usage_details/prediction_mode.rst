.. _prediction_mode:

Prediction Mode
===============

Use the prediction mode when you want to apply a pre-trained model to 
localise a particular target, such as the ventral intermediate nucleus of the thalamus (Vim). 
Ensure you've generated the required tract-density maps before starting.

Example with Default Naming
---------------------------

This example assumes your tract-density maps follow the naming convention of the default target list the model was trained with. 
Default target lists are available under the package root directory ``resources/data``. 
For Vim models, refer to ``resources/data/vim_default_targets.txt``.

.. code-block:: bash

    localise --predict --structure=vim --subject=subj1001 --mask=roi/left/tha_small.nii.gz --target_path=streamlines/left --data_type=single32 --spatial --out=predictions.nii.gz

Example with Custom Naming
--------------------------

If your tract-density maps use a different naming scheme, you can provide your own target list. Additionally, you can input a text file containing paths to multiple subjects. 
Make sure they are in the same order as the ``resources/data/vim_default_targets.txt``.

.. code-block:: bash

    localise --predict --structure=vim --subject=subjs.txt --mask=roi/left/tha_small.nii.gz --target_path=streamlines/left --target_list=your_target_list.txt --data_type=single32 --spatial --out=predictions.nii.gz

Using a Custom model
--------------------

If you don't want to use the pre-trained default model released with the package, 
and you've trained your own model ``your_trained_model.pth`` 
using a specific target list ``your_target_list.txt`` 
(see :ref:`Train Mode <train_mode>` for custom training), you can use the following:

.. code-block:: bash

    localise --predict --subject=subjs.txt --mask=roi/left/tha_small.nii.gz --model=your_trained_model.pth --target_path=streamlines/left --target_list=your_target_list.txt --spatial --out=predictions.nii.gz

