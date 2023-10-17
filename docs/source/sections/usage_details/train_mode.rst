.. _train_mode:

Train Mode
==========

Use the training mode when you want to train a custom model with a specific target list. 
Ensure you possess the required labels that serve as ground truth during the training process.

To train a model using subjects listed in ``training_subjs.txt``, and targets in ``your_target_list.txt``, 
and subsequently save the trained model as ``your_trained_model.pth``, use the following:

.. code-block:: bash

    localise --train --subject=training_subjs.txt --mask=roi/left/tha_small.nii.gz --label=high-quality-labels/left/labels.nii.gz --target_path=streamlines/left --target_list=your_target_list.txt --out_model=your_trained_model.pth --spatial

Once trained, your custom model (``your_trained_model.pth``) can be applied to new, unseen subjects, 
as showcased in the :ref:`Prediction Mode <prediction_mode>` section.

