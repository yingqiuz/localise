import os, logging
import numpy as np
import nibabel as nib
from pathlib import Path
from localise.load import load_data, load_features, ShuffledDataLoader
from localise.train import train, train_with_val, train_without_val
from localise.predict import apply_pretrained_model


PKG_PATH = Path(__file__).parent.parent

def save_nifti(data, mask_file, output_file):
    """
    Function to save a 1D numpy array into a 3D NIfTI file using a specified binary NIfTI mask.

    Parameters:
    data (np.array): 1D numpy array to be saved.
    mask_file (str): Path to the binary NIfTI mask file.
    output_file (str): Output path where the NIfTI file will be saved.
    """

    # Load the mask NIfTI file
    mask_nifti = nib.load(mask_file)
    mask_data = mask_nifti.get_fdata()

    # Check if the number of non-zero entries in the mask matches the length of the data
    if np.count_nonzero(mask_data) != len(data):
        raise ValueError('The number of non-zero entries in the mask does not match the length of the data.')

    # Create a 3D numpy array from the mask
    output_data = np.zeros_like(mask_data)

    # Distribute the data into the 3D space defined by the mask
    output_data[mask_data > 0] = data

    # Create a NIfTI image from the output data
    output_nifti = nib.Nifti1Image(output_data, mask_nifti.affine, mask_nifti.header)

    # Save the NIfTI image
    nib.save(output_nifti, output_file)

# Usage example:
# data = np.random.rand(1000)  # For example, a 1D numpy array
# mask_file = 'mask.nii.gz'  # Binary NIfTI mask file
# output_file = 'output.nii.gz'  # Output NIfTI file
# save_nifti(data, mask_file, output_file)


def save_nifti_4D(data, mask_file, output_file):
    """
    Function to save a 2D numpy array into a 4D NIfTI file using a specified binary NIfTI mask.

    Parameters:
    data (np.array): 2D numpy array to be saved.
    mask_file (str): Path to the binary NIfTI mask file.
    output_file (str): Output path where the NIfTI file will be saved.
    """

    # Load the mask NIfTI file
    mask_nifti = nib.load(mask_file)
    mask_data = mask_nifti.get_fdata()

    # number of classes
    k = data.shape[1]

    # size in x, y, z dim
    x, y, z = mask_data.shape

    # Check if the number of non-zero entries in the mask matches the length of the data
    if np.count_nonzero(mask_data) != len(data):
        raise ValueError('The number of non-zero entries in the mask does not match the length of the data.')

    # Create a 3D numpy array from the mask
    output_data = np.zeros((x, y, z, k))

    # non-zero items
    indices = mask_data > 0

    # Distribute the data into the 4D space defined by the mask
    for kk in range(k):
        output_data[indices, kk] = data[:, kk]

    # Create a NIfTI image from the output data
    output_nifti = nib.Nifti1Image(output_data, mask_nifti.affine, mask_nifti.header)

    # Save the NIfTI image
    nib.save(output_nifti, output_file)


def get_subjects(subject_path):
    """Load subjects from file or directory."""
    if os.path.isfile(subject_path):
        with open(subject_path, 'r') as f:
            return [line.strip() for line in f]
    elif os.path.isdir(subject_path):
        return [subject_path]
    else:
        raise ValueError(f'Invalid subject path: {subject_path}. Please specify a correct subject dir or txt file.')


def predict_mode(subject, mask, structure=None, target_path=None, target_list=None, 
                 data=None, atlas=None, out=None, model=None, spatial=True, data_type=None):

    logging.info('Predict mode on.\n')
    subjects = get_subjects(subject)

    if model is None:
        # error checking
        if structure is None:
            raise ValueError('When using the default model, you must specify the structure.')
        if data_type is None:
            raise ValueError('When using the default model, you must specify the data_type.')

        logging.info(f'Using the default model for {structure} on {data_type}.')
        # load the default model.
        model_dir = os.path.join(PKG_PATH, 'resources', 'models', structure, data_type)
        model_name = f'{structure}_spatial_model.pth' if spatial else f'{structure}_model.pth'
        model = os.path.join(model_dir, model_name)

        if not os.path.exists(model):
            raise ValueError(f'We dont have a pretrained model for {structure} {data_type}.')

        target_list_fname = os.path.join(PKG_PATH, 'resources', 'data', 
                                         f'{structure}_default_target_list.txt')
        # checking whether or not to use default
        if data is None and target_list is None:
            # load default target list
            logging.info('Using default target list.')
            
            with open(target_list_fname, 'r') as f:
                target_list = [line.strip() for line in f]

        else:
            logging.info(f'Please make sure your data or target_list matches the order of the default target list {target_list_fname}.')

    else:
        logging.info(f'Using the model stored in {model}.')

        # check errors. either specify --data, or specify both --target_path and --target_list
        if data is None:
            if target_path is None:
                raise ValueError("Please specify --target_path if you didn't specify --data")
            if target_list is None:
                raise ValueError("Please specify --target_list if you didn't specify --data when you are not using the default model.")        

    # load connectivity features
    data = [
        load_features(
            subject=subject, 
            mask_name=mask, 
            target_path=target_path, 
            target_list=target_list, 
            data=data, 
            atlas=atlas
        ) 
        for subject in subjects
    ]

    predictions = apply_pretrained_model(data, model, spatial_model=spatial)

    # save to nii files
    for subject, prediction in zip(subjects, predictions):
        save_nifti(prediction.detach().numpy()[:, -1], os.path.join(subject, mask), os.path.join(subject, out))

    return predictions


def train_mode(subject, mask, label, target_path=None,
               target_list=None, data=None, atlas=None, out_model=None, 
               spatial=True, epochs=100):
    
    logging.info('Training mode on.\n')
    subjects = get_subjects(subject)
    
    if data is None and target_list is None:
        raise ValueError("Please specify --target_list if you didn't specify --data.")
    
    if data is None and target_path is None:
        raise ValueError("Please specify --target_path if you didn't specify --data.")
    
    data = [
        load_data(
            subject=subject, 
            mask_name=mask, 
            label_name=label,
            target_path=target_path, 
            target_list=target_list, 
            data=data, 
            atlas=atlas
        ) 
        for subject in subjects
    ]
    
    dataloader = ShuffledDataLoader(data)
    model = train_without_val(dataloader, n_epochs=epochs, 
                              spatial_model=spatial, 
                              model_save_path=out_model)
    
    return model

