import os, pytest
import numpy as np
import subprocess
from localise.utils import save_nifti, save_nifti_4D, get_subjects
from localise.utils import predict_mode, train_mode
from localise.utils import run_command
from pathlib import Path
import nibabel as nib
from unittest.mock import patch


path_to_data = Path(__file__).parent / 'test_data'

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


def test_get_subjects(tmp_path):
    # Test with file
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "subjects.txt"
    p.write_text("subject1\nsubject2\n")

    result = get_subjects(str(p))
    assert result == ["subject1", "subject2"]

    # Test with directory
    result = get_subjects(str(d))
    assert result == [str(d)]

    # Test with invalid path
    with pytest.raises(ValueError):
        get_subjects(str(tmp_path / "nonexistent"))


def test_predict_mode():
    subject = f'{path_to_data}/101915'
    mask = 'roi/left/mist_left_thalamus_mask_small_1.nii.gz'
    target_path = 'streamlines/left'
    target_list = f'{path_to_data}/models/vim_default_target_list160.txt'
    atlas = 'roi/left/atlas.nii.gz'
    out = 'roi/left/prediction.nii.gz'
    model = f'{path_to_data}/models/tmp_model160_2mm_single32.pth'

    predict_mode(subject=subject, mask=mask, target_path=target_path, 
                 target_list=target_list, atlas=atlas, out=out, model=model)
    assert os.path.isfile(os.path.join(subject, out))
    os.remove(os.path.join(subject, out))

    predict_mode(subject=subject, mask=mask, structure='vim', target_path=target_path, 
                 atlas=atlas, out=out, data_type='2mm_single32')
    assert os.path.isfile(os.path.join(subject, out))
    os.remove(os.path.join(subject, out))
    
    with pytest.raises(ValueError):
        predict_mode(subject=subject, mask=mask, target_path=target_path, 
                     atlas=atlas, out=out)
        
    with pytest.raises(ValueError):
        predict_mode(subject=subject, mask=mask, structure='vim', target_path=target_path, 
                     atlas=atlas, out=out)
        
    predict_mode(subject=subject, mask=mask, structure='vim', target_path=target_path, 
                 target_list=target_list, atlas=atlas, out=out, data_type='2mm_single32')

    assert os.path.isfile(os.path.join(subject, out))
    os.remove(os.path.join(subject, out))

    predict_mode(subject=subject, mask=mask, data='streamlines/left/X160_small_1mm.npy', 
                 structure='vim', spatial=False,
                 atlas=atlas, out=out, data_type='2mm_single32')

    assert os.path.isfile(os.path.join(subject, out))
    os.remove(os.path.join(subject, out))

def test_train_mode():
    subject = f'{path_to_data}/100206'
    mask = 'roi/left/tha_small.nii.gz'
    label = 'high-quality-labels/left/labels.nii.gz'
    target_path = 'streamlines/left'
    target_list = f'{path_to_data}/models/targets_list113_left.txt'
    atlas = 'roi/left/atlas.nii.gz'
    out_model = f'{path_to_data}/models/test_model.pth'

    train_mode(subject=subject, mask=mask, label=label, target_path=target_path, 
               target_list=target_list, atlas=atlas, out_model=out_model)
    
    assert os.path.isfile(out_model)
    os.remove(out_model)
    
def test_run_command():
    # Test successful command execution
    with patch('subprocess.run') as mock_run:
        run_command(['echo', 'hello'])
        mock_run.assert_called_once_with(['echo', 'hello'], check=True)

    # Test unsuccessful command execution
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'bad_cmd')) as mock_run:
        run_command(['bad_cmd'])
        mock_run.assert_called_once_with(['bad_cmd'], check=True)
    