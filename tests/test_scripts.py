import pytest
from unittest.mock import patch
from localise.args import parse_arguments


def test_parse_arguments():

    # Testing the training mode
    with patch('argparse._sys.argv', 
               ['localise', '--train', 
                '--subject', '/path/to/subject', 
                '--seed', 'seed.nii.gz',
                '--mask_dir', 'roi',
                '--label', 'roi/label/label.nii.gz',
                '--out_model', 'model',
                '--target_dir', 'tracts',
                '--hemisphere', 'left',
                '--data', 'data.npy',
                '--epochs', '100']):
        args = parse_arguments()

        assert args.train
        assert args.subject == '/path/to/subject'
        assert args.seed == 'seed.nii.gz'
        assert args.mask_dir == 'roi'
        assert args.label == 'roi/label/label.nii.gz'
        assert args.out_model == 'model'
        assert args.target_dir == 'tracts'
        assert args.data == 'data.npy'
        assert args.epochs == 100

    # Testing the prediction mode
    with patch('argparse._sys.argv', 
               ['localise', '--predict', 
                '--subject', '/path/to/subject', 
                '--seed', 'seed.nii.gz',
                '--mask_dir', 'roi',
                '--target_dir', 'tracts',
                '--data', 'data.npy',
                '--hemisphere', 'right',
                '--structure', 'vim',
                '--data_type', 'single32',
                '--out', 'output']):
        args = parse_arguments()

        assert args.predict
        assert args.subject == '/path/to/subject'
        assert args.mask_dir == 'roi'
        assert args.data == 'data.npy'
        assert args.out == 'output'
        assert args.seed == 'seed.nii.gz'
        assert args.target_dir == 'tracts'
        assert args.structure == 'vim'

    # Testing the prediction mode with structure and model missing
    with pytest.raises(SystemExit):
        with patch('argparse._sys.argv', 
                ['localise', '--predict', 
                    '--subject', '/path/to/subject', 
                    '--seed', 'seed.nii.gz',
                    '--mask_dir', 'roi',
                    '--target_dir', 'tracts',
                    '--data', 'data.npy',
                    '--hemisphere', 'right',
                    '--out', 'output']):
            args = parse_arguments()

    # Testing the prediction mode with default atlas
    with pytest.raises(SystemExit):
        with patch('argparse._sys.argv', 
                ['localise', '--predict', 
                    '--subject', '/path/to/subject', 
                    '--seed', 'seed.nii.gz',
                    '--mask_dir', 'roi',
                    '--target_dir', 'tracts',
                    '--data', 'data.npy',
                    '--hemisphere', 'right',
                    '--model', 'model.pth',
                    '--atlas', 'default',
                    '--out', 'output']):
            args = parse_arguments()

    # Testing the prediction mode with default atlas
    with patch('argparse._sys.argv', 
               ['localise', '--predict', 
                '--subject', '/path/to/subject', 
                '--seed', 'seed.nii.gz',
                '--mask_dir', 'roi',
                '--target_dir', 'tracts',
                '--data', 'data.npy',
                '--hemisphere', 'right',
                '--structure', 'vim',
                '--out', 'output']):
        with pytest.raises(SystemExit):
            args = parse_arguments()
        
    # Test the missing argument in training mode
    with pytest.raises(SystemExit):
        with patch('argparse._sys.argv', ['localise', '--train']):
            args = parse_arguments()
            
    # Test the missing argument in prediction mode
    with pytest.raises(SystemExit):
        with patch('argparse._sys.argv', ['localise', '--predict']):
            args = parse_arguments()
            
    # Test the case where both train and predict are provided
    with pytest.raises(SystemExit):
        with patch('argparse._sys.argv', ['localise', '--train', '--predict']):
            args = parse_arguments()
            
    # Test the case where neither train or predict is provided
    with pytest.raises(SystemExit):
        with patch('argparse._sys.argv', ['localise']):
            args = parse_arguments()
