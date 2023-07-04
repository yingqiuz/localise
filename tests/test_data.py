#!/usr/bin/env python

# Test data-related functionality
import os
from glob import glob

from pathlib import Path
path_to_data = Path(__file__).parent / 'test_data'

from localise.load import load_features
from localise.flatten_batch import FlattenedCRFBatchTensor
def test_load_features():
    batch = load_features(subject=f'{path_to_data}/100610', 
              mask_name='roi/left/tha_small.nii.gz', 
              target_path='streamlines/left', 
              target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'])
    
    assert type(batch) == FlattenedCRFBatchTensor
    assert batch.f.shape[0] == 1
    assert batch.f.dim() == 3
    
    subject='100610'
    target_list = [os.path.split(f)[-1] for f in sorted(glob(f'{path_to_data}/'+subject+'/streamlines/left/seeds_to_*'))]

    power = [1, 2]
    gamma = [0, 0.1]
    batch = load_features(subject=os.path.join(f'{path_to_data}', subject), 
              mask_name='roi/left/tha_small.nii.gz', 
              target_path='streamlines/left', 
              target_list=target_list, power=power, gamma=gamma)

    assert batch.X.shape[1] == len(target_list) * len(power)
    assert batch.f.shape[0] == len(gamma)
    assert batch.f.shape[1] == batch.X.shape[0]
    
def test_load_labels():
    subject=os.path.join(f'{path_to_data}','100610')
    mask_name='roi/left/tha_small.nii.gz'
    label_name='high-quality-labels/left/labels.nii.gz'
    from localise.load import load_labels
    labels = load_labels(subject, mask_name, label_name)
    assert list(labels.shape) == [4142, 2]
        
        

