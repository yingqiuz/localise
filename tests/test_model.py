import os
from glob import glob

from pathlib import Path
path_to_data = Path(__file__).parent / 'test_data'

from localise.load import load_features, load_data
from localise.flatten_batch import FlattenedCRFBatchTensor
from localise.flatten_forward import FlexibleClassifier, MLP
from localise.train import train_loop
import torch


def test_flexibleclassifier():
    subject='100610'
    target_list = [os.path.split(f)[-1] for f in sorted(glob(f'{path_to_data}/'+subject+'/streamlines/left/seeds_to_*'))]

    batch = load_features(subject=os.path.join(f'{path_to_data}',subject), 
                          mask_name='roi/left/tha_small.nii.gz', 
                          target_path='streamlines/left', 
                          target_list=target_list, power=[1, 2], gamma=[0, 0.1])
    
    model = FlexibleClassifier(torch.nn.Linear(150, 2), is_crf=True, n_kernels=2, n_classes=2)
    output = model(batch)
    assert list(output.shape) == [batch.X.shape[0], 2]
    
    model = FlexibleClassifier(torch.nn.Linear(150, 2), is_crf=False, n_classes=2)
    output = model(batch)
    assert list(output.shape) == [batch.X.shape[0], 2]


def test_MLP():
    # Define input_dim, hidden_dim, output_dim, and create a model instance
    input_dim = 10
    hidden_dim = 5
    output_dim = 2
    model = MLP(input_dim, hidden_dim, output_dim)

    # Create a mock input tensor with size (batch_size, input_dim)
    batch_size = 32
    mock_input = torch.randn(batch_size, input_dim)

    # Perform a forward pass
    output = model(mock_input)

    # Check that the output has the expected size
    assert output.size() == (batch_size, output_dim)

    # Check that the output is not NaN or infinite
    assert torch.all(torch.isfinite(output))
