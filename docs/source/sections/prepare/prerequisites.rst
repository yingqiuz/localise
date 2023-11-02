.. _prerequisites:

Prerequisites
=============

This section describes the prerequisites for running the toolbox.

- :ref:`required_data`
- :ref:`registration`
- :ref:`cortical` 
- :ref:`data_organisation`

.. _required_data:

Required Data 
-------------

You'll need both T1 and diffusion MRI, preferally bias-corrected.

Ensure that the diffusion and T1 images are properly aligned, 
producing transformation affine matrices that map between the two spaces. 
You can achieve this using tools from `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>`_. 
Specifically, you can use FSL's `flirt <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide>`_ 
or `epi_reg <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide#epi_reg>`_. 

This usually involves transforming B0 images into the structural (T1) space. If using ``epi_reg``, here's an example:

.. code-block:: console

    $ epi_reg --epi=<B0> --t1=<T1> --t1brain=<T1_brain> --out=<B0_to_T1>
  
where ``<B0>`` is the B0 image, ``<T1>`` is the T1 image, ``<T1_brain>`` is the brain-extracted T1 image,
and ``<B0_to_T1>`` is the output transformation matrix that maps diffusion space into structural space.

To obtain the inverse transformation matrix ``<T1_to_B0>``, simply use FSL's 
`convert_xfm <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide#convert_xfm>`_:

.. code-block:: console
    
    $ convert_xfm -omat <T1_to_B0> -inverse <B0_to_T1>
    
where ``<T1_to_B0>`` is the output transformation matrix that maps structural space back into diffusion space.

.. _registration:

Registration to Standard space
------------------------------

You'll also need to obtain the warp fields that aligns MNI 1mm standard space to the structural space.
This can be done via FSL's `fnirt <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FNIRT/UserGuide>`_:

.. code-block:: console

    $ fnirt --ref=<T1> --in=<MNI_1mm> --aff=<T1_to_MNI_1mm> --cout=<T1_to_MNI_1mm_warp>
    
where ``<T1>`` is the T1 image, ``<MNI_1mm>`` is the MNI 1mm standard space image, 
``<T1_to_MNI_1mm>`` is the affine transformation matrix that maps structural space to MNI 1mm space,
and ``<T1_to_MNI_1mm_warp>`` is the output warp field that maps MNI 1mm space to structural space.

.. _cortical:

Cortical and Subcortical Parcellations
--------------------------------------

You'll also need to generate cortical and subcortical parcellations from T1. 
This can be done via Freesurfer's `recon-all <https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all>`_:

.. code-block:: console

    $ recon-all -all -i <T1> -s <subject_id> -sd <output_dir> -parallel -openmp <nthreads>
    
where ``<T1>`` is the T1 image, ``<subject_id>`` is the subject ID, ``<output_dir>`` is the output directory,
and ``<nthreads>`` is the number of threads to use.

.. _data_organisation:

Data Organisation 
-----------------

If you're working with multiple subjects (where each subject has its own folder), 
ensure that you organise the files for each subject consistently. 
For example, the T1 and diffusion files
should have the same naming convention across subjects.

.. code-block:: console
    
    subject001/
    ├── structural
    │   ├── T1_brain.nii.gz
    ├── diffusion
    │   ├── bvals
    │   ├── bvecs
    │   ├── data.nii.gz
    └── otherfiles
    subject002/
    ├── structural
    │   ├── T1_brain.nii.gz
    ├── diffusion
    │   ├── bvals
    │   ├── bvecs
    │   ├── data.nii.gz
    └── otherfiles
    subject003/

This will simplify the process moving forward. 