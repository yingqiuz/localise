# LOCALISE

[![Build Status](https://git.fmrib.ox.ac.uk/yqzheng1/python-localise/badges/main/pipeline.svg)](https://git.fmrib.ox.ac.uk/yqzheng1/python-localise)
[![Documentation Status](https://img.shields.io/badge/docs-dev-blue.svg)](https://open.win.ox.ac.uk/pages/yqzheng1/python-localise/)
[![Codecov](https://git.fmrib.ox.ac.uk/yqzheng1/python-localise/badges/main/coverage.svg)](https://git.fmrib.ox.ac.uk/yqzheng1/python-localise)

This library implements LOCALISE, a python toolbox developed to address the challenges associated with accurately targeting DBS targets on low-quality clinical-like dataset. This toolbox uses Image Quality Transfer techniques to transfer anatomical information from high-quality data to a wide range of connectivity features in low-quality data. The goal is to augment the inference on DBS targets localisation even with compromised data quality. We also have a [Julia implementation](https://git.fmrib.ox.ac.uk/yqzheng1/hqaugmentation.jl). For more details, please check our [research paper](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_17).

## Installation

```bash
pip install git+https://github.com/yingqiuz/localise.git
```

Requirements: [FSL](https://fsl.fmrib.ox.ac.uk/fsl/) (with `FSLDIR` set) for the
`prepare-*` steps; PyTorch for training and prediction.

## Usage

The `localise` command covers the full pipeline. To localise a structure
(e.g., VIM) for a subject you need a reference image (usually the T1),
a FreeSurfer `aparc.a2009s+aseg` segmentation in that space, a warp field
from MNI standard space to the reference space, and a bedpostX folder:

```bash
# 1. create anatomical masks in reference space (both hemispheres)
localise prepare-masks --ref sub01/t1.nii.gz \
                       --aparc sub01/aparc.a2009s+aseg.nii.gz \
                       --warp sub01/std2native_warp.nii.gz \
                       --structure vim --out sub01

# 2. run probabilistic tractography (both hemispheres; add --gpu if available)
localise prepare-tracts --bpx sub01/dMRI.bedpostX --masks sub01/masks \
                        --structure vim --out sub01/streamlines

# 3. localise the structure with the pre-trained model (both hemispheres)
localise predict --masks sub01/masks --tracts sub01/streamlines \
                 --structure vim --spatial --out sub01
```

By default `predict` uses the model trained on 2mm single-shell (32-direction)
data. `--model` selects a different one: either the name of a shipped model
trained on another low-quality protocol (`--model 2mm`, `--model single32`),
or the path to your own trained model (`--model my_model.pth`).

The probability maps are saved as `sub01/left/probmap.nii.gz` and
`sub01/right/probmap.nii.gz`. Add `--hemisphere left` (or `right`) to any
command to process a single hemisphere.

You can also train your own model on subjects with high-quality labels and
apply it (a model is trained per hemisphere):

```bash
localise train --seed seeds.txt --labels labels.txt --tracts tracts.txt \
               --hemisphere left --spatial --out-model my_model.pth

localise predict --seed sub01/roi/left/tha.nii.gz \
                 --tracts sub01/streamlines --hemisphere left \
                 --model my_model.pth --spatial --out sub01
```

where the txt files list one path per subject. Run `localise <subcommand> --help`
for all options, including `localise connectivity-driven` for the classic
thresholded-overlap approach.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) for details.

## Contributing

This toolbox is under active development. We are integrating more structures greatly appreciate contributions from the community! If you're interested in improving localise, there are many ways you can contribute.

### Reporting Issues

If you're experiencing a problem with localise, please open an issue on the GitHub repository. When submitting an issue, try to include as much detail as you can. This should include the exact steps to reproduce the issue, your operating system, and version of localise.

### Fixing bugs or adding new features

If you're interested in contributing code to fix open issues or adding new features, please follow these steps:

 1. Fork the repository and clone it locally.
 2. Create a branch for your edits.
 3. Add, commit, and push your changes.
 4. Submit a pull request.

When submitting a pull request, please make sure to include a descriptive title and clear description of your changes.

### Documentation

If you spot a problem with the documentation or think it could be clearer, you can make amendments by editing the relevant files and submitting a pull request.

Remember, contributions aren't just about code - any way in which you can help us improve is much appreciated!

Thank you for your interest in contributing to localise!
