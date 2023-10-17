# localise

This library implements LOCALISE.

## Installation

```bash
git clone https://git.fmrib.ox.ac.uk/yqzheng1/python-localise.git
cd python-localise
pip install .
```

## Usage

### 1. **Preparation step**: generate connectivity features using tractography (either on high- or low-quality data)

If you haven't done so, you may want to create anatomical masks and run tractography to generate connectivity features. If you have your own tract density maps, please skip to [2. Localise a target](#2-localise-a-target).

#### 1.1 Create anatomical masks

To create anatomical masks, you can either run the shell scripts under `/scripts` by

```bash
create_masks --ref=/path/to/t1.nii.gz --warp=/path/to/warp.nii.gz --out=/path/to/roi --aparc=/path/to/aparc.a2009s+aseg.nii.gz
```

where `--ref=t1.nii.gz` and `--warp=warp.nii.gz` are required arguments, specifying the reference image and the warp field that maps MNI 1mm space to the reference space. It is recommended to have your own cortical parcellation file `--aparc=aparc.a2009s+aseg.nii.gz`, otherwise standard ROIs will be employed. If the output directory is unspecified, it will create a folder called `roi` under the same directory as the reference image. More instructions can be found by typing `/path/to/HQAugmentation.jl/scripts/create_masks.sh --help`.

#### 1.2 Create tract density maps (connectivity features)

To create tract density maps, you can either run the following in command line

```bash
create_tract --samples=/path/to/data.bedpostX --inputdir=/path/to/roi/left --seed=/path/to/seed.nii.gz --out=/path/to/my_output_dir
```

where `--samples` and `--inputdir` are required arguments. You must specify the folder containing `merged_*samples.nii.gz` samples, e.g., the output directory from FSL's BedpostX. `--inputdir` is the directory that stores anatomical masks. If you don't specify the seed, it will use `tha.nii.gz` under the `inputdir`. More info can be found by typing `/path/to/HQAugmentation.jl/scripts/create_tract.sh --help`.

You can also run this in julia by executing the following function:

#### 1.3 Create connectivity-driven Vim

If you have high-quality data, you may create "ground truth" location using the following connectivity-driven approach.

```shell
connectivity_driven --target1=seeds_to_m1.nii.gz --thr1=50 --target2=seeds_to_cerebellum.nii.gz --thr2=15 --out=label.nii.gz
```

will create connectivity-driven Vim.

Subjects must be organised in the same file structure. Suppose your data is organised in this way:

### 2. Localise a target

#### 2.1 Command-line interface

```bash
subject001/
├── streamlines
│   ├── left
│       ├── seeds_to_target1.nii.gz
│       ├── seeds_to_target2.nii.gz
│   ├── right
│       ├── seeds_to_target1.nii.gz
│       ├── seeds_to_target2.nii.gz
├── roi
│   ├── left
│       ├── tha.nii.gz
│       ├── atlas.nii.gz
│   ├── right
│       ├── tha.nii.gz
│       ├── atlas.nii.gz
├── high-quality-labels
│   ├── left
│       ├── labels.nii.gz
│   ├── right
│       ├── labels.nii.gz
└── otherfiles
subject002/
subject003/
```

For a comprehensive list of available commands and options, enter:

```bash
localise --help
```

##### Prediction mode

Use the prediction mode when you want to apply a pre-trained model to localise a particular target, such as the ventral intermediate nucleus of the thalamus (Vim). Ensure you've generated the required tract-density maps before starting.

###### Example with Default Naming

This example assumes your tract-density maps follow the naming convention of the default target list the model was trained with. Default target lists are available under `resources/data`. For Vim models, refer to `resources/data/vim_default_targets.txt`.

```bash
localise --predict --structure=vim --subject=subj1001 --mask=roi/left/tha_small.nii.gz --target_path=streamlines/left --data_type=single32 --spatial --out=predictions.nii.gz
```

###### Example with Custom Naming

If your tract-density maps use a different naming scheme, you can provide your own target list. Additionally, you can input a text file containing paths to multiple subjects. Make sure they are in the same order as the `resources/data/vim_default_targets.txt`.

```bash
localise --predict --structure=vim --subject=subjs.txt --mask=roi/left/tha_small.nii.gz --target_path=streamlines/left --target_list=your_target_list.txt --data_type=single32 --spatial --out=predictions.nii.gz
```

###### Using a Custom Model

If you don't want to use the pre-trained default model, and you've trained your own model `your_trained_model.pth` using a specific target list `your_target_list.txt` (see [Train mode](#train-mode) for custom training), you can use the following:

```bash
localise --predict --subject=subjs.txt --mask=roi/left/tha_small.nii.gz --model=your_trained_model.pth --target_path=streamlines/left --target_list=your_target_list.txt --spatial --out=predictions.nii.gz
```

##### Train mode

Use the training mode when you want to train a custom model with a specific target list. Ensure you possess the required labels that serve as ground truth during the training process.

To train a model using subjects listed in `training_subjs.txt`, and targets in `your_target_list.txt`, and subsequently save the trained model as `your_trained_model.pth`, use the following:

```bash
localise --train --subject=training_subjs.txt --mask=roi/left/tha_small.nii.gz --label=high-quality-labels/left/labels.nii.gz --target_path=streamlines/left --target_list=your_target_list.txt --out_model=your_trained_model.pth --spatial
```

Once trained, your custom model (your_trained_model.pth) can be applied to new, unseen subjects, as showcased in the [Prediction mode](#prediction-mode) section.

#### 2.2 Python interface

You can also train a model or localise for unseen subjects using the python interface.

```python
import numpy as np
from localise.load import load_data, load_features, ShuffledDataloader
from localise.train import train
from localise.predict import apply_model

with open('train_subjs.txt', 'r') as f:
    train_list = [line.strip() for line in f]

with open('test_subjs.txt', 'r') as f:
    test_list = [line.strip() for line in f]

mask_name = 'roi/left/tha_small.nii.gz'
label_name = 'high-quality-labels/left/labels.nii.gz'
target_path = 'streamlines/left'
target_list = 'target_list.txt'
atlas = 'roi/left/atlas.nii.gz'
output_fname = 'streamlines/left/data.npy'

# load training data
train_data = load_data(subject=train_list, mask_name=mask_name, 
                       target_path=target_path, target_list=target_list, 
                       atlas=atlas, label_name=label_name, 
                       output_fname=output_fname)
# ShuffedDataloader shuffles the order of training subjects in each epoch
train_dataloarder = ShuffledDataloader(train_data)

# load test data
test_data = load_data(subject=test_list, mask_name=mask_name, 
                      target_path=target_path, target_list=target_list, 
                      atlas=atlas, label_name=label_name, 
                      output_fname=output_fname)

test_dataloarder = ShuffledDataloader(test_data)
#### training
# the trained model is saved in out_model
model_save_path = 'your_trained_model.pth'

# train a model and store in m
m = train(train_dataloarder, test_dataloader, model_save_path=model_save_path)

# make predictions
predictions = apply_model(test_data, m)

# save to nii
for prediction, subject in zip(predictions, test_list):
    save_nifti_4D(prediction, os.path.join(subject, mask_name), os.path.join(subject, 'predictions'))
```
