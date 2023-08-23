# localise

This library implements LOCALISE.

## Installation

```bash
git clone https://git.fmrib.ox.ac.uk/yqzheng1/python-localise.git
cd python-localise
pip install .
```

## Usage

### Command-line interface

For a comprehensive list of available commands and options, enter:

```bash
localise --help
```

#### Prediction mode

Use the prediction mode when you want to apply a pre-trained model to localise a particular target, such as the ventral intermediate nucleus of the thalamus (Vim). Ensure you've generated the required tract-density maps before starting.

##### Example with Default Naming

This example assumes your tract-density maps follow the naming convention of the default target list the model was trained with. Default target lists are available under `resources/data`. For Vim models, refer to `resources/data/vim_default_targets.txt`.

```bash
localise --predict --structure=vim --subject=subj1001 --mask=roi/left/tha_small.nii.gz --target_path=streamlines/left --data_type=single32 --spatial --out=predictions.nii.gz
```

##### Example with Custom Naming

If your tract-density maps use a different naming scheme, you can provide your own target list. Additionally, you can input a text file containing paths to multiple subjects. Make sure they are in the same order as the `resources/data/vim_default_targets.txt`.

```bash
localise --predict --structure=vim --subject=subjs.txt --mask=roi/left/tha_small.nii.gz --target_path=streamlines/left --target_list=your_target_list.txt --data_type=single32 --spatial --out=predictions.nii.gz
```

##### Using a Custom Model

If you don't want to use the pre-trained default model, and you've trained your own model `your_trained_model.pth` using a specific target list `your_target_list.txt` (see [Train mode](#train-mode) for custom training), you can use the following:

```bash
localise --predict --subject=subjs.txt --mask=roi/left/tha_small.nii.gz --model=your_trained_model.pth --target_path=streamlines/left --target_list=your_target_list.txt --spatial --out=predictions.nii.gz
```

#### Train mode

Use the training mode when you want to train a custom model with a specific target list. Ensure you possess the required labels that serve as ground truth during the training process.

To train a model using subjects listed in `training_subjs.txt`, and targets in `your_target_list.txt`, and subsequently save the trained model as `your_trained_model.pth`, use the following:

```bash
localise --train --subject=training_subjs.txt --mask=roi/left/tha_small.nii.gz --label=high-quality-labels/left/labels.nii.gz --target_path=streamlines/left --target_list=your_target_list.txt --out_model=your_trained_model.pth --spatial
```

Once trained, your custom model (your_trained_model.pth) can be applied to new, unseen subjects, as showcased in the [Prediction mode](#prediction-mode) section.

### Python interface
