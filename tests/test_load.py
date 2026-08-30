#!/usr/bin/env python

# Test data-related functionality
import os
from glob import glob

from pathlib import Path
path_to_data = Path(__file__).parent / 'test_data'

from localise.load import load_features, load_labels, load_data, ShuffledDataLoader
from localise.batch import FlattenedCRFBatchTensor


def subject_paths(subject):
    sub = path_to_data / subject
    return {
        'seed': sub / 'roi' / 'left' / 'tha_small.nii.gz',
        'tracts': sub / 'streamlines' / 'left',
        'labels': sub / 'high-quality-labels' / 'left' / 'labels.nii.gz',
        'atlas': sub / 'roi' / 'left' / 'atlas.nii.gz',
    }


def test_load_features():
    p = subject_paths('100610')
    batch = load_features(seed=p['seed'], tracts=p['tracts'],
              target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'])

    assert type(batch) == FlattenedCRFBatchTensor
    assert batch.f.shape[0] == 1
    assert batch.f.dim() == 3

    batch = load_features(seed=p['seed'], tracts=p['tracts'],
              atlas=p['atlas'], power=[0.5, 1, 2],
              target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'])
    assert batch.X.shape[1] == 7

    target_list = [os.path.split(f)[-1]
                   for f in sorted(glob(str(p['tracts'] / 'seeds_to_*')))]

    power = [1, 2]
    gamma = [0, 0.1]
    batch = load_features(seed=p['seed'], tracts=p['tracts'],
              target_list=target_list, power=power, gamma=gamma,
              output_fname=p['tracts'] / 'features75.npy')

    assert batch.X.shape[1] == len(target_list) * len(power)
    assert batch.f.shape[0] == len(gamma)
    assert batch.f.shape[1] == batch.X.shape[0]

    batch = load_features(seed=p['seed'], power=power, gamma=gamma,
              data=p['tracts'] / 'features75.npy')

    assert batch.X.shape[1] == len(target_list) * len(power)
    assert batch.f.shape[0] == len(gamma)
    assert batch.f.shape[1] == batch.X.shape[0]


def test_load_features_multiple_subjects():
    subjects = ['100610', '100307']
    batches = load_features(
        seed=[subject_paths(s)['seed'] for s in subjects],
        tracts=[subject_paths(s)['tracts'] for s in subjects],
        target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'],
        power=[1, 2])

    assert isinstance(batches, list)
    assert len(batches) == len(subjects)
    for batch in batches:
        assert type(batch) == FlattenedCRFBatchTensor
        assert batch.X.shape[1] == 4


def test_load_labels():
    p = subject_paths('100610')
    labels = load_labels(p['seed'], p['labels'])
    assert list(labels.shape) == [4142, 2]

    subjects = ['100610', '100408']
    labels = load_labels(seed=[subject_paths(s)['seed'] for s in subjects],
                         labels=[subject_paths(s)['labels'] for s in subjects])
    assert isinstance(labels, list)
    assert len(labels) == len(subjects)


def test_load_data():
    p = subject_paths('100408')
    batch = load_data(seed=p['seed'],
                      labels=p['labels'],
                      power=[1, 2],
                      atlas=p['atlas'],
                      tracts=p['tracts'],
                      target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'])
    assert batch[0].X.shape[0] == batch[1].shape[0]
    assert batch[0].X.shape[1] == 5
    assert isinstance(batch, tuple)
    assert isinstance(batch[0], FlattenedCRFBatchTensor)
    assert len(batch) == 2


def test_load_data_multiple_subjects():
    subjects = ['100610', '100307', '100408']
    batches = load_data(
        seed=[subject_paths(s)['seed'] for s in subjects],
        labels=[subject_paths(s)['labels'] for s in subjects],
        tracts=[subject_paths(s)['tracts'] for s in subjects],
        target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'],
        power=[1, 2])

    assert isinstance(batches, list)
    assert len(batches) == len(subjects)
    for features, labels in batches:
        assert features.X.shape[1] == 4
        assert labels.shape[0] == features.X.shape[0]


def test_shuffleddataloader():
    subjects = ['100610', '100307', '100408']
    data = load_data(
        seed=[subject_paths(s)['seed'] for s in subjects],
        labels=[subject_paths(s)['labels'] for s in subjects],
        tracts=[subject_paths(s)['tracts'] for s in subjects],
        target_list=['seeds_to_11101_1.nii.gz', 'seeds_to_11102_1.nii.gz'],
        power=[1, 2])
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
