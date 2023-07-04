import os
from glob import glob

from pathlib import Path
path_to_data = Path(__file__).parent / 'test_data'

from localise.load import load_features
from localise.flatten_batch import FlattenedCRFBatchTensor
from localise.flatten_forward import Affine, AffineCRF


def test_affine():
    subject='100610'
    target_list = [os.path.split(f)[-1] for f in sorted(glob(f'{path_to_data}/'+subject+'/streamlines/left/seeds_to_*'))]

    batch = load_features(subject=os.path.join(f'{path_to_data}',subject), 
                          mask_name='roi/left/tha_small.nii.gz', 
                          target_path='streamlines/left', 
                          target_list=target_list,power=[1])
    
    model = Affine(75, 2)
    output = model(batch)
    assert list(output.shape) == [batch.X.shape[1], 2]
    
def test_affine_crf():
    subject='100610'
    target_list = [os.path.split(f)[-1] for f in sorted(glob(f'{path_to_data}/'+subject+'/streamlines/left/seeds_to_*'))]

    batch = load_features(subject=os.path.join(f'{path_to_data}',subject), 
                          mask_name='roi/left/tha_small.nii.gz', 
                          target_path='streamlines/left', 
                          target_list=target_list,power=[1],gamma=[0, 0.01])
    
    model = AffineCRF(75, 2, 2)
    output = model(batch)
    assert list(output.shape) == [batch.X.shape[1], 2]