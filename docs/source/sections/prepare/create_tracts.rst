.. _tract:

Generate tract-density maps
===========================

After you have generated anatomical masks (e.g., stored under ``subj001/roi``), 
now you can create tract-density maps (connectivity features).
This can be done by running the ``create_tracts`` command. For example, if you want to generate tract-density maps 
for a seed region ``seed.nii.gz``, you can run the following command:

.. code-block:: bash

   create_tracts.sh --inputdir=subj001/roi --samples=subj001/dMRI.bedpostX --seed=seed.nii.gz --out=subj001/tracts

or, using short flags:

.. code-block:: bash

   create_tracts.sh -i subj001/roi -m subj001/dMRI.bedpostX -s seed.nii.gz -o subj001/tracts

This script creates tracts for the specified seed ``seed.nii.gz`` in both hemispheres, 
using fiber samples ``subj001/dMRI.bedpostX`` estimated via BedpostX, 
and stores the results in ``subj001/tracts``.

Parameters
----------

- ``-m``, ``--samples``:
  BedpostX directory storing fibre orientation files. Example: ``data.bedpostX``.

- ``-i``, ``--inputdir``:
  Input directory storing left and right hemisphere anatomical masks in native T1 space.

- ``-s``, ``--seed``:
  Seed mask, usually in the individual T1 native space (path not needed if in the ``inputdir``).

- ``-o``, ``--out``:
  Output directory for tract density maps. Defaults to 'tracts_out' in the ``inputdir``.

- ``-x``, ``--xfm`` (Optional):
  Transformation matrix mapping the reference T1 space to the diffusion space. Necessary if diffusion is not aligned to the reference T1 space.

- ``-r``, ``--ref`` (Optional):
  Reference image, usually in the individual T1 native space. Necessary if ``xfm`` (``-x``) is specified.

- ``-b``, ``--brainmask`` (Optional):
  Binary brain mask in the diffusion space.

- ``-n``, ``--nsteps`` (Optional):
  Number of steps per sample. Default=2000.

- ``-c``, ``--cthr`` (Optional):
  Curvature threshold. Default=0.2.

- ``-f``, ``--fibthresh`` (Optional):
  Volume fraction threshold before considering subsidiary fibre orientations. Default=0.01.

- ``-j``, ``--nsamples`` (Optional):
  Number of samples. Default=10000.

- ``-l``, ``--steplength`` (Optional):
  Steplength in mm. Default=0.5.

- ``-p``, ``--sampvox`` (Optional):
  Sample random points within a sphere of radius 'x' mm from the seed voxel center.

- ``-d``, ``--distthresh`` (Optional):
  Discards samples shorter than this threshold (in mm). Default=0.

- ``-g``, ``--gpu`` (Optional):
  Use the GPU version of ``probtractx2``.

- ``-q``, ``--queue`` (Optional):
  The job queue to submit to. Runs locally if unspecified.

For additional details and instructions, run ``create_tracts --help``.

More Examples
-------------

1. **Using GPU to run tractography**: 
   If you have a GPU, you can raise the ``--gpu`` flag to use the GPU version of ``create_tracts``:

.. code-block:: bash

    create_tracts.sh -i subj001/roi -m subj001/dMRI.bedpostX -s seed.nii.gz -o subj001/tracts -g   

2. **Submitting to a job queue**: If you have a system that supports job queues 
and `fsl_sub <https://git.fmrib.ox.ac.uk/fsl/fsl_sub>`_ installed,
you can specify the queue and potentially the number of jobs:

.. code-block:: bash

   create_tracts.sh -i subj001/roi -m subj001/dMRI.bedpostX -s seed.nii.gz -o subj001/tracts -g -q gpu_short