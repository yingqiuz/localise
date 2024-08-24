import os, sys
import numpy as np
import subprocess
from localise.utils import get_resources_path
from localise.utils import save_nifti, save_nifti_4D
from localise.utils import run_command
from pathlib import Path
import nibabel as nib
from unittest.mock import patch


path_to_data = Path(__file__).parent / 'test_data'

def test_get_resources_path():
    # Test for non-frozen application
    with patch.object(sys, 'frozen', False, create=True):
        with patch('os.path.dirname') as mock_dirname:
            mock_dirname.return_value = '/fake/path/localise'
            result = get_resources_path()
            assert result == '/fake/path/resources'

    # Test for frozen application
    with patch.object(sys, 'frozen', True, create=True):
        with patch.object(sys, '_MEIPASS', '/fake/frozen/path', create=True):
            result = get_resources_path()
            assert result == '/fake/frozen/path/resources'

    # Test that the returned path exists (assuming you're running tests from the project root)
    actual_path = get_resources_path()
    assert os.path.exists(actual_path), f"The path {actual_path} does not exist"
    assert os.path.isdir(actual_path), f"The path {actual_path} is not a directory"

def test_save_nifti():
    subject = '100610'
    mask = os.path.join(path_to_data, subject, 'roi', 'left', 'tha.nii.gz')
    mask_data = nib.load(mask).get_fdata()
    output_fname = os.path.join(path_to_data, subject, 'saved_file.nii.gz')

    vectors = np.random.randn(np.count_nonzero(mask_data))
    save_nifti(vectors, mask, output_fname)
    
    saved_data = nib.load(output_fname).get_fdata()
    assert np.allclose(saved_data[mask_data != 0], vectors, atol=1e-6, rtol=1e-5)


def test_save_nifti_4D():
    subject = '100610'
    mask = os.path.join(path_to_data, subject, 'roi', 'left', 'tha.nii.gz')
    mask_data = nib.load(mask).get_fdata()
    output_fname = os.path.join(path_to_data, subject, 'saved_file.nii.gz')

    vectors = np.random.randn(np.count_nonzero(mask_data), 2)
    save_nifti_4D(vectors, mask, output_fname)
    
    saved_data = nib.load(output_fname).get_fdata()
    assert saved_data.shape[-1] == 2
    for k in range(2):
        saved_data_slice = saved_data[:, :, :, k]
        assert np.allclose(saved_data_slice[mask_data != 0], vectors[:, k], atol=1e-6, rtol=1e-5)
    
def test_run_command():
    # Test successful command execution
    with patch('subprocess.run') as mock_run:
        run_command(['echo', 'hello'])
        mock_run.assert_called_once_with(['echo', 'hello'], check=True)

    # Test unsuccessful command execution
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'bad_cmd')) as mock_run:
        run_command(['bad_cmd'])
        mock_run.assert_called_once_with(['bad_cmd'], check=True)
    