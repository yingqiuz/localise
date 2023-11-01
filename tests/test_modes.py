import os, pytest
from localise.modes import get_subjects
from localise.modes import predict_mode, train_mode
from localise.modes import return_hemisphere, check_params
from pathlib import Path
from unittest.mock import patch


path_to_data = Path(__file__).parent / 'test_data'

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

def test_return_hemisphere():
    # Test valid inputs for left hemisphere
    assert return_hemisphere('left') == 'left'
    assert return_hemisphere('l') == 'left'
    assert return_hemisphere('L') == 'left'
    
    # Test valid inputs for right hemisphere
    assert return_hemisphere('right') == 'right'
    assert return_hemisphere('r') == 'right'
    assert return_hemisphere('R') == 'right'

    # Test invalid input
    with pytest.raises(ValueError, match=r"Invalid hemisphere: center. Please specify left or right."):
        return_hemisphere('center')
    
    # Test empty input
    with pytest.raises(ValueError, match=r"Invalid hemisphere: . Please specify left or right."):
        return_hemisphere('')

def test_check_params():
    mask_dir = "masks"
    target_dir = "targets"
    hemisphere = "left"

    # Test valid inputs
    mask, m_dir, target_path, data = check_params("mask1.nii.gz", mask_dir, target_dir, "data1.nii.gz", hemisphere)
    assert mask == os.path.join(mask_dir, hemisphere, "mask1.nii.gz")
    assert m_dir == mask_dir
    assert target_path == os.path.join(target_dir, hemisphere)
    assert data == os.path.join(target_dir, hemisphere, "data1.nii.gz")

    # Test missing mask_dir
    with pytest.raises(ValueError, match="Please specify the directory for anatomical masks"):
        check_params("mask1.nii.gz", None, target_dir, "data1.nii.gz", hemisphere)

    # Test missing target_dir
    with pytest.raises(ValueError, match="Please specify the directory for target masks"):
        check_params("mask1.nii.gz", mask_dir, None, "data1.nii.gz", hemisphere)

    # Test None data
    mask, m_dir, target_path, data = check_params("mask1.nii.gz", mask_dir, target_dir, None, hemisphere)
    assert data is None

def test_predict_mode():
    subject = f'{path_to_data}/101915'
    mask = 'mist_left_thalamus_mask_small_1.nii.gz'
    mask_dir = 'roi'
    hemisphere = 'left'
    target_dir = 'streamlines'
    target_list = f'{path_to_data}/models/vim_default_target_list160.txt'
    atlas = 'roi/left/vim.nii.gz'
    out = 'roi/left/prediction.nii.gz'
    model = f'{path_to_data}/models/tmp_model160_2mm_single32.pth'

    predict_mode(subject=subject, mask=mask, mask_dir=mask_dir, target_dir=target_dir, 
                 target_list=target_list, atlas=atlas, out=out, model=model, hemisphere=hemisphere)
    assert os.path.isfile(os.path.join(subject, out))
    os.remove(os.path.join(subject, out))

    predict_mode(subject=subject, mask=mask, mask_dir=mask_dir, structure='vim', target_dir=target_dir, 
                 atlas='default', out=out, data_type='2mm_single32', hemisphere=hemisphere)
    assert os.path.isfile(os.path.join(subject, out))
    os.remove(os.path.join(subject, out))
    
    with pytest.raises(ValueError):
        predict_mode(subject=subject, mask=mask, mask_dir=mask_dir, target_dir=target_dir, 
                     atlas=atlas, out=out, hemisphere=hemisphere)
        
    with pytest.raises(ValueError):
        predict_mode(subject=subject, mask=mask, mask_dir=mask_dir, structure='vim', target_dir=target_dir, 
                     atlas=atlas, out=out, hemisphere=hemisphere)
        
    with pytest.raises(ValueError):
        predict_mode(subject=subject, mask=mask, mask_dir=mask_dir, structure='vim', target_dir=target_dir, 
                     atlas=atlas, out=out, hemisphere=None)
        
    predict_mode(subject=subject, mask=mask, mask_dir=mask_dir, structure='vim', target_dir=target_dir, 
                 target_list=target_list, atlas=atlas, out=out, data_type='2mm_single32', hemisphere=hemisphere)

    assert os.path.isfile(os.path.join(subject, out))
    os.remove(os.path.join(subject, out))

    predict_mode(subject=subject, mask=mask, mask_dir=mask_dir, target_dir=target_dir, 
                 data='X160_small_1mm.npy', structure='vim', spatial=False,
                 atlas=atlas, out=out, data_type='2mm_single32', hemisphere=hemisphere)

    assert os.path.isfile(os.path.join(subject, out))
    os.remove(os.path.join(subject, out))

def test_train_mode():
    subject = f'{path_to_data}/100206'
    mask_dir = 'roi'
    mask = 'tha_small.nii.gz'
    label = 'high-quality-labels/left/labels.nii.gz'
    target_dir = 'streamlines'
    target_list = f'{path_to_data}/models/targets_list113_left.txt'
    atlas = 'roi/left/atlas.nii.gz'
    out_model = f'{path_to_data}/models/test_model.pth'

    train_mode(subject=subject, mask=mask, label=label, mask_dir=mask_dir, target_dir=target_dir,
               target_list=target_list, atlas=atlas, out_model=out_model, hemisphere='left')
    
    assert os.path.isfile(out_model)
    os.remove(out_model)
