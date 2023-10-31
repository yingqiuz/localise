import os, logging, subprocess
import numpy as np
import nibabel as nib
import subprocess
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

def return_hemisphere(hemisphere):
    ## a function to check hemisphere
    if isinstance(hemisphere, str):
        if hemisphere.lower() in ['left', 'l']:
            hemisphere = 'left'
        elif hemisphere.lower() in ['right', 'r']:
            hemisphere = 'right'
        else:
            raise ValueError(f'Invalid hemisphere: {hemisphere}. Please specify left or right.')
    else:
        raise ValueError(f'Invalid hemisphere: {hemisphere}. Please specify left or right.')

    return hemisphere

def check_params(mask, mask_dir, target_dir, data, hemisphere):
    # create mask name
    if mask_dir is None:
        raise ValueError('Please specify the directory for anatomical masks (relative to the subject folder).')

    mask = os.path.join(mask_dir, hemisphere, mask)

    # create target_path
    if target_dir is None:
        raise ValueError('Please specify the directory for target masks (relative to the subject folder).')

    target_path = os.path.join(target_dir, hemisphere)

    if data is not None:
        data = os.path.join(target_dir, hemisphere, data)
    
    return mask, mask_dir, target_path, data

def predict_mode(subject, mask, mask_dir=None, structure=None, target_dir=None, target_list=None, 
                 data=None, atlas='default', out=None, model=None, spatial=True, 
                 data_type=None, hemisphere=None):

    logging.info('Predict mode on.\n')
    subjects = get_subjects(subject)
    hemisphere = return_hemisphere(hemisphere)
    mask, mask_dir, target_path, data = check_params(mask, mask_dir, target_dir, data, hemisphere)

    # create output name
    if out is None: 
        raise ValueError('Please specify the output name.')

    if atlas == 'default':
        if structure is None:
            raise ValueError('Please specify the structure (--structure, -s) if using the default atlases.')
        # create default atlas path
        atlas = os.path.join(mask_dir, hemisphere, f'{structure}.nii.gz')

    if model is None:
        # error checking
        if structure is None:
            raise ValueError('When using the default model, you must specify the structure.')
        if data_type is None:
            raise ValueError('When using the default model, you must specify the data_type.')

        logging.info(f'Using the default model for {structure} on {data_type}.')
        
        # load the default model.
        model_dir = os.path.join(PKG_PATH, 'resources', 'models', structure, data_type, hemisphere)
        model_name = 'spatial_model.pth' if spatial else 'model.pth'
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
    if not spatial:
        atlas = None

    data = [
        load_features(
            subject=subject, 
            mask_name=mask, 
            target_path=target_path, 
            target_list=target_list, 
            data=data, 
            atlas=atlas,
            power=[2, 1, 0.5]
        ) 
        for subject in subjects
    ]

    predictions = apply_pretrained_model(data, model, spatial_model=spatial)

    # save to nii files
    for subject, prediction in zip(subjects, predictions):
        save_nifti(prediction.detach().numpy()[:, -1], os.path.join(subject, mask), os.path.join(subject, out))

    return predictions


def train_mode(subject, mask, label, mask_dir=None, target_dir=None,
               target_list=None, data=None, atlas=None, out_model=None, 
               spatial=True, hemisphere=None, epochs=100):
    
    logging.info('Training mode on.\n')
    subjects = get_subjects(subject)
    hemisphere = return_hemisphere(hemisphere)
    mask, mask_dir, target_path, data = check_params(mask, mask_dir, target_dir, data, hemisphere)

    if data is None and target_list is None:
        raise ValueError("Please specify --target_list if you didn't specify --data.")
    
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


def create_masks(ref, warp, out=None, aparc=None, brainmask=None):
    """
    Create masks based on provided parameters.
    
    Parameters:
    - ref (str): Reference string.
    - warp (str): Warp string.
    - out (Optional[str]): Output string. Defaults to None.
    - aparc (Optional[str]): Aparc string. Defaults to None.
    - brainmask (Optional[str]): Brainmask string. Defaults to None.
    
    Raises:
    - ValueError: If required environment variables or files are missing.
    """
    if "FSLDIR" not in os.environ:
        raise ValueError("FSLDIR environment variable does not exist.")
    if not os.path.isfile(ref):
        raise ValueError(f"{ref} does not exist.")
    if not os.path.isfile(warp):
        raise ValueError(f"{warp} does not exist.")
    
    cmd = os.path.join(PKG_PATH, "scripts", "create_masks")
    args = [cmd, f"--ref={ref}", f"--warp={warp}"]
    
    params = [("out", out), ("aparc", aparc), ("brainmask", brainmask)]
    for param, value in params:
        if value is not None:
            args.append(f"--{param}={value}")
    
    run_command(args)


def create_tracts(samples_dir, input_dir, seed=None, xfm=None, ref=None, 
                  out=None, brainmask=None, nsteps=None, cthr=None, 
                  fibthresh=None, nsamples=None, steplength=None, 
                  sampvox=None, distthresh=None, gpu=True):
    """
    Create tracts based on provided parameters.
    
    Parameters are described in the function signature.
    
    Raises:
    - ValueError: If required environment variables or directories are missing.
    """
    
    if "FSLDIR" not in os.environ:
        raise ValueError("FSLDIR environment variable is missing.")
    if not os.path.isdir(samples_dir):
        raise ValueError(f"Directory '{samples_dir}' is missing.")
    if not os.path.isdir(input_dir):
        raise ValueError(f"Directory '{input_dir}' is missing.")
    
    cmd = os.path.join(PKG_PATH, "scripts", "create_tracts")
    args = [cmd, f"--samples={samples_dir}", f"--inputdir={input_dir}"]
    
    params = [
        ("out", out), 
        ("seed", seed), 
        ("xfm", xfm), 
        ("ref", ref), 
        ("brainmask", brainmask),
        ("nsteps", nsteps),
        ("cthr", cthr),
        ("fibthresh", fibthresh),
        ("nsamples", nsamples),
        ("steplength", steplength),
        ("sampvox", sampvox),
        ("distthresh", distthresh)
    ]
    for param, value in params:
        if value is not None:
            args.append(f"--{param}={value}")
    
    if gpu:
        args.append("--gpu")
    
    run_command(args)


def connectivity_driven(target1, target2, out, target3=None, 
                        thr1=None, thr2=None, thr3=None, thr=None):
    """
    Perform a connectivity-driven analysis using two or three target images.

    Parameters are described in the function signature.
    
    Raises:
    - ValueError: If required environment variables or directories are missing.
    """
    if "FSLDIR" not in os.environ:
        raise ValueError("FSLDIR environment variable is missing.")
    if not os.path.isfile(target1):
        raise ValueError(f"{target1} does not exist.")
    if not os.path.isfile(target2):
        raise ValueError(f"{target2} does not exist.")
    
    cmd = os.path.join(PKG_PATH, "scripts", "connectivity_driven")
    args = [cmd, f"--target1={target1}", f"--target2={target2}", f"--out={out}"]
    
    params = [
        ("thr1", thr1),
        ("thr2", thr2),
        ("thr3", thr3),
        ("target3", target3),
        ("thr", thr)
    ]
    for param, value in params:
        if value is not None:
            args.append(f"--{param}={value}")
    
    run_command(args)
    

def run_command(cmd):
    try:
        subprocess.run(cmd, check=True)
        print(f"Command {' '.join(cmd)} executed successfully.")
    except subprocess.CalledProcessError:
        print(f"Error executing command: {' '.join(cmd)}")