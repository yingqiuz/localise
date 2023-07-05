import os
from glob import glob

from pathlib import Path
path_to_data = Path(__file__).parent / 'test_data'

import torch
from localise.load import load_data
from localise.train import train_loop, val_loop, train
from localise.flatten_forward import FlexibleClassifier, MLP

subjects=['100610', '100408', '100307']

target_list = [os.path.split(f)[-1] for f in sorted(glob(f'{path_to_data}/100610/streamlines/left/seeds_to_*'))]
batches = [load_data(subject=os.path.join(f'{path_to_data}',subject), 
                      mask_name='roi/left/tha_small.nii.gz', 
                      label_name='high-quality-labels/left/labels.nii.gz',
                      target_path='streamlines/left', 
                      target_list=target_list, power=[1, 2], gamma=[0, 0.1]) for subject in subjects]


def test_train_loop():
    model = FlexibleClassifier(torch.nn.Linear(150, 2), is_crf=True, n_kernels=2, n_classes=2)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    train_loop(batches, model, loss_fn, optimizer, 1e-3, 1e-3, print_freq=1)


def test_val_loop():
    model = FlexibleClassifier(MLP(150, 2, 2), is_crf=True, n_kernels=2, n_classes=2)
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loop(batches, model, loss_fn)
    
def test_train():
    m = train([batches[0], batches[1]], [batches[2]], n_epochs=5)
    assert isinstance(m, FlexibleClassifier)