.. _estimate-fiber:

Estimate fiber orientations
===========================

You'll need to estimate fiber orientations for tractography. 
You need to run FSL's `bedpostx <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#BEDPOSTX>`_
on your diffusion data. The input directory, e.g., `diffusion_data`, must contain the following files:

-  `data.nii.gz` - the 4D diffusion weighted data
-  `bvecs` - the b-vectors for the diffusion weighted data
-  `bvals` - the b-values for the diffusion weighted data
-  `nodif_brain_mask.nii.gz` - a brain mask for the diffusion weighted data

Tip: Run ``bedpostx_datacheck`` in command line to check if your input directory contains 
the correct files required for bedpostx. Then to run bedpostx, type in command line:

.. code-block:: bash

    bedpostx <diffusion_data> -n 3

This will create a directory called `diffusion_data.bedpostX` with the following files:

- `merged_th<i>samples` - 4D volume - Samples from the distribution on theta
- `merged_ph<i>samples` - 4D volume - Samples from the distribution on phi 
  (theta and phi together represent the principal diffusion direction)
- `merged_f<i>samples` - 4D volume - Samples from the distribution on anisotropic volume fraction
- `mean_th<i>samples` - 3D volume - Mean of the distribution on theta
- `mean_ph<i>samples` - 3D volume - Mean of the distribution on phi
- `mean_f<i>samples` - 3D volume - Mean of the distribution on anisotropic volume fraction
- `dyads<i>` - Mean of principal diffusion direction in vector form
- `mean_S0samples` - 3D volume - Mean of distribution on T2w baseline signal intensity S0
- `mean_dsamples` - 3D volume - Mean of distribution on diffusivity
- `mean_d_stdsamples` - 3D Volume - Mean of distribution on diffusivity variance parameter d_std (not produced if --model=1)
- `dyads<i>_dispersion`` - 3D Volume - Uncertainty on the estimated fibre orientation.
- `nodif_brain_mask` - binary mask created from nodif_brain - copied from input directory

For a detailed understanding and any troubleshooting, refer to the
`BEDPOSTX UserGuide <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#BEDPOSTX>`_.