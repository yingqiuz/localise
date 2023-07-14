#!/usr/bin/env python

# Test data-related functionality
import os
from glob import glob

from pathlib import Path
path_to_data = Path(__file__).parent / 'test_data'

from localise.load import load_features, load_labels, load_data, ShuffledDataLoader
from localise.flatten_batch import FlattenedCRFBatchTensor
from torch.utils.data import DataLoader


def test_load_features():
    batch = load_features(subject=f'{path_to_data}/100610', 
              mask_name='roi/left/tha_small.nii.gz', 
              target_path='streamlines/left', 
              target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'])
    
    assert type(batch) == FlattenedCRFBatchTensor
    assert batch.f.shape[0] == 1
    assert batch.f.dim() == 3
    
    batch = load_features(subject=f'{path_to_data}/100610', 
              mask_name='roi/left/tha_small.nii.gz', 
              target_path='streamlines/left', 
              atlas='roi/left/atlas.nii.gz', power=[0.5, 1, 2],
              target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'])
    assert batch.X.shape[1] == 7
    
    subject='100610'
    target_list = [os.path.split(f)[-1] for f in sorted(glob(f'{path_to_data}/'+subject+'/streamlines/left/seeds_to_*'))]

    power = [1, 2]
    gamma = [0, 0.1]
    batch = load_features(subject=os.path.join(f'{path_to_data}', subject), 
              mask_name='roi/left/tha_small.nii.gz', 
              target_path='streamlines/left', 
              target_list=target_list, power=power, gamma=gamma, 
              output_fname='streamlines/left/features75.npy')

    assert batch.X.shape[1] == len(target_list) * len(power)
    assert batch.f.shape[0] == len(gamma)
    assert batch.f.shape[1] == batch.X.shape[0]
    
    batch = load_features(subject=os.path.join(f'{path_to_data}', subject), 
              mask_name='roi/left/tha_small.nii.gz', 
              power=power, gamma=gamma, 
              data='streamlines/left/features75.npy')

    assert batch.X.shape[1] == len(target_list) * len(power)
    assert batch.f.shape[0] == len(gamma)
    assert batch.f.shape[1] == batch.X.shape[0]
    
def test_load_labels():
    subject=os.path.join(f'{path_to_data}','100610')
    mask_name='roi/left/tha_small.nii.gz'
    label_name='high-quality-labels/left/labels.nii.gz'
    labels = load_labels(subject, mask_name, label_name)
    assert list(labels.shape) == [4142, 2]
        
def test_load_data():
    mask_name='roi/left/tha_small.nii.gz'
    label_name='high-quality-labels/left/labels.nii.gz'
    batch = load_data(subject=f'{path_to_data}/100408', 
                      mask_name=mask_name, 
                      label_name=label_name,
                      power=[1, 2],
                      atlas='roi/left/atlas.nii.gz',
                      target_path='streamlines/left', 
                      target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'])
    assert batch[0].X.shape[0] == batch[1].shape[0]
    assert batch[0].X.shape[1] == 5
    assert isinstance(batch, tuple)
    assert isinstance(batch[0], FlattenedCRFBatchTensor)
    assert len(batch) == 2

def test_shuffleddataloader():
    mask_name='roi/left/tha_small.nii.gz'
    label_name='high-quality-labels/left/labels.nii.gz'
    data = [load_data(subject=f'{path_to_data}/{subject}', 
                              mask_name=mask_name, 
                              label_name=label_name,
                              target_path='streamlines/left', 
                              power=[1, 2],
                              target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'])
                for subject in ['100610', '100307', '100408']]
    dataloader = ShuffledDataLoader(data)
    assert len(dataloader) == 3
    X, y = dataloader[0]
    assert X.X.shape[0] == y.shape[0]
    for batch in dataloader:
        features, labels = batch
        assert features.X.shape[1] == 4
        assert labels.shape[0] == features.X.shape[0]
    
    train_set, test_set = dataloader.split_data(0.67)
    assert len(test_set) == 1
    assert len(train_set) == 2
    for batch in train_set:
        features, labels = batch
        assert features.X.shape[1] == 4
        assert labels.shape[0] == features.X.shape[0]

    for batch in test_set:
        features, labels = batch
        assert features.X.shape[1] == 4
        assert labels.shape[0] == features.X.shape[0]
        

