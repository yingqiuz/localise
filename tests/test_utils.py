import os, sys, pytest
import numpy as np
import subprocess
import tempfile
from unittest.mock import patch, Mock, MagicMock, call
from localise.utils import (
    get_resources_path,
    get_absolute_path,
    check_fsl_environment,
    run_fsl_command,
    check_fsl_sub_queues,
    save_nifti,
    save_nifti_4D,
    run_command,
    find_mask_file
)
from pathlib import Path
import nibabel as nib


path_to_data = Path(__file__).parent / 'test_data'

def test_get_resources_path_ends_with_resources():
    assert get_resources_path().name == 'resources'
    assert get_resources_path().exists()
    assert get_resources_path() == Path(__file__).parent.parent / 'resources'


def test_get_absolute_path():
    """Test getting absolute path."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        result = get_absolute_path(tmp_file.name)
        assert os.path.isabs(result)
        assert result == str(Path(tmp_file.name).resolve())
        
        
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
    
    
def test_check_fsl_environment_success():
    """Test FSL environment check when FSLDIR is set."""
    with patch.dict(os.environ, {'FSLDIR': '/usr/local/fsl'}):
        fsl_dir = check_fsl_environment()
        assert fsl_dir == '/usr/local/fsl'


def test_check_fsl_environment_missing():
    """Test FSL environment check when FSLDIR is not set."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError, match="FSLDIR does not exist"):
            check_fsl_environment()
            

def test_run_fsl_command_success():
    """Test successful FSL command execution."""
    with patch('subprocess.run') as mock_run:
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        result = run_fsl_command(['echo', 'test'])
        assert result == mock_result
        mock_run.assert_called_once_with(['echo', 'test'], check=True, capture_output=True, text=True)


def test_run_fsl_command_failure():
    """Test FSL command execution failure."""
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, 'command', stderr='error')
        
        with pytest.raises(subprocess.CalledProcessError):
            run_fsl_command(['false'])


def test_check_fsl_sub_queues_yes():
    """Test FSL queue detection when queues are available."""
    with patch('subprocess.run') as mock_run:
        mock_result = Mock()
        mock_result.stdout = "Yes\n"
        mock_run.return_value = mock_result
        
        result = check_fsl_sub_queues()
        assert result is True


def test_check_fsl_sub_queues_no():
    """Test FSL queue detection when queues are not available."""
    with patch('subprocess.run') as mock_run:
        mock_result = Mock()
        mock_result.stdout = "No\n"
        mock_run.return_value = mock_result
        
        result = check_fsl_sub_queues()
        assert result is False


def test_check_fsl_sub_queues_error():
    """Test FSL queue detection when command fails."""
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd')):
        result = check_fsl_sub_queues()
        assert result is False
    
    with patch('subprocess.run', side_effect=FileNotFoundError):
        result = check_fsl_sub_queues()
        assert result is False
        

def test_find_mask_file():
    """Test mask file finding."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test.nii.gz"
        test_file.touch()
        
        result = find_mask_file(Path(tmp_dir) / "test")
        assert result == str(test_file)
        
        # Test not found
        result = find_mask_file(Path(tmp_dir) / "nonexistent")
        assert result is None