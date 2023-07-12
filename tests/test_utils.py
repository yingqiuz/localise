import os
from localise.utils import save_nifti
from pathlib import Path
import nibabel as nib


path_to_data = Path(__file__).parent / 'test_data'

def test_save_nifti():
    subject = '100610'
    mask = os.path.join(subject, 'roi', 'left', 'tha.nii.gz')
    mask_data = nib.load(mask).get_fdata()
    output_fname = os.path.join(subject, 'saved_file.nii.gz')

    vectors = np.random.randn(np.count_nonzero(mask_data))
    save_nifti(vectors, mask, output_fname)
    
    saved_data = nib.load(output_fname).get_fdata()
    assert np.all(saved_data[mask_data != 0] == vectors)