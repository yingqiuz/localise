import os
import numpy as np
from localise.utils import save_nifti, get_subjects
from pathlib import Path
import nibabel as nib


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