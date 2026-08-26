import os, sys
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path


PKG_PATH = Path(__file__).parent.parent

def get_resources_path():
    """
    Get the absolute path to the resources directory.
    This function works whether the package is installed or in development mode.
    """
    if getattr(sys, 'frozen', False):
        # The application is frozen (packaged)
        return Path(sys._MEIPASS) / 'resources'
    else:
        # The application is not frozen
        # Get the directory of the current file (utils.py)
        current_file = Path(__file__).resolve()
        # Go up one level to the package root, then into 'resources'
        return current_file.parent.parent / 'resources'
    

def get_absolute_path(file_path):
    """Get absolute path of a file."""
    return str(Path(file_path).resolve())


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


def run_command(cmd):
    try:
        subprocess.run(cmd, check=True)
        print(f"Command {' '.join(cmd)} executed successfully.")
    except subprocess.CalledProcessError:
        print(f"Error executing command: {' '.join(cmd)}")
        
        
def check_fsl_environment():
    """Check if FSL is properly set up."""
    fsl_dir = os.environ.get('FSLDIR')
    if not fsl_dir:
        raise EnvironmentError("Environment variable FSLDIR does not exist. Please set up FSLDIR.")
    return fsl_dir


def fsl_bin(tool):
    """Absolute path of an FSL executable ($FSLDIR/bin/<tool>), so that FSL
    commands work even when FSLDIR is set but not on PATH. Falls back to the
    bare tool name if FSLDIR is not set."""
    fsl_dir = os.environ.get('FSLDIR')
    return str(Path(fsl_dir) / 'bin' / tool) if fsl_dir else tool


def run_fsl_command(command, check=True):
    """Run an FSL command and return the result."""
    try:
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise


def check_fsl_sub_queues():
    """Check if FSL has queue system available."""
    try:
        result = subprocess.run(['fsl_sub', '--has_queues'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip() == "Yes"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    

def find_mask_file(base_path, extensions=['.nii.gz', '.nii']):
    """Find a mask file with common extensions."""
    for ext in extensions:
        file_path = Path(str(base_path) + ext)
        if file_path.exists():
            return str(file_path)
    return None