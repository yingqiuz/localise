.. _create-anatomical-masks:

Create anatomical masks
=======================

In this section, you will create all necessary anatomical masks/templates required for tractography and subsequent analysis.
This can be done by simply running ``create_masks``. For example, for a single subject ``subj0001``, you can run:

.. code-block:: bash

    create_masks --ref=/path/to/subj0001/t1.nii.gz --warp=/path/to/subj0001/warp.nii.gz --out=/path/to/subj0001/roi --aparc=/path/to/subj0001/aparc.a2009s+aseg.nii.gz

Or, using short flags:

.. code-block:: bash

   create_masks -r /path/to/subj0001/t1.nii.gz -w /path/to/subj0001/warp.nii.gz -o /path/to/subj0001/roi -a /path/to/subj0001/aparc.a2009s+aseg.nii.gz
   
After running this command, anatomical masks required to run tractography and subsequent
analysis for both hemisphere will be created under the output directory you specified, 
for example, ``/path/to/subj0001/roi/left`` and ``/path/to/subj0001/roi/right``.

Parameters
----------

- **-r, --ref** : The reference image, typically in the individual T1 native space.
- **-w, --warp** : The warp field that maps the MNI standard space to the individual T1 space.
- **-a, --aparc** : The cortical segmentation file from Freesurfer. Acceptable formats include `*.mgz`, `*.nii.gz`, or `*.nii`.
- **-o, --out** : The directory to store the anatomical masks. 
- **-b, --brainmask** : (Optional) A binary brain mask in the reference space.
- **-q, --queue** : (Optional) The queue for job submissions. By default, jobs run locally.
- **-n, --njobs** : (Optional) Defines the number of jobs to run in parallel locally.

For additional details and instructions, run ``create_masks --help``.

More Examples
-------------

1. **Running locally  in parallel**: If you don't specify the queue to be submitted to, the script will run locally. 
   For example, to run the script locally in parallel with 4 jobs, you can run:

.. code-block:: bash

   create_masks.sh -r t1.nii.gz -w standard2str.nii.gz -o outputdir -n 4

2. **Submitting to a job queue**: If you have a system that supports job queues 
and `fsl_sub <https://git.fmrib.ox.ac.uk/fsl/fsl_sub>`_ installed,
you can specify the queue and potentially the number of jobs:

.. code-block:: bash

   create_masks.sh -r t1.nii.gz -w standard2str.nii.gz -o outputdir -q short
