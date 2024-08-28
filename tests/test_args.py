import pytest
import sys
from unittest.mock import patch
from localise.args import parse_arguments
from io import StringIO

def test_parse_arguments():

    # Train a custom model with pre-saved data
    with patch('argparse._sys.argv', 
               ['localise', '--train', 
                '--subject', '/path/to/subject', 
                '--seed', '/path/to/subject/seed.nii.gz',
                '--masks', '/path/to/subject/roi',
                '--labels', '/path/to/subject/roi/labels/label.nii.gz',
                '--out_model', 'model.pth',
                '--hemisphere', 'left',
                '--data', '/path/to/subject/data.npy',
                '--atlas', '/path/to/subject/atlas.nii.gz']):
        args = parse_arguments()

        assert args.train
        assert args.subject == '/path/to/subject'
        assert args.seed == '/path/to/subject/seed.nii.gz'
        assert args.masks == '/path/to/subject/roi'
        assert args.labels == '/path/to/subject/roi/labels/label.nii.gz'
        assert args.out_model == 'model.pth'
        assert args.data == '/path/to/subject/data.npy'
        assert args.atlas == '/path/to/subject/atlas.nii.gz'
        assert args.epochs == 100
        
    # train a custom model with a list tract-density maps
    with patch('argparse._sys.argv', 
               ['localise', '--train', 
                '--subject', '/path/to/subject', 
                '--seed', '/path/to/subject/seed.nii.gz',
                '--masks', '/path/to/subject/roi',
                '--labels', '/path/to/subject/roi/labels/label.nii.gz',
                '--out_model', 'model.pth',
                '--hemisphere', 'left',
                '--tracts', '/path/to/subject/tracts/',
                '--tracts_list', 'tracts_list.txt',
                '--epochs', '1000']):
        args = parse_arguments()

        assert args.train
        assert args.subject == '/path/to/subject'
        assert args.seed == '/path/to/subject/seed.nii.gz'
        assert args.masks == '/path/to/subject/roi'
        assert args.labels == '/path/to/subject/roi/labels/label.nii.gz'
        assert args.out_model == 'model.pth'
        assert args.tracts == '/path/to/subject/tracts/'
        assert args.tracts_list == 'tracts_list.txt'
        assert args.epochs == 1000
        assert args.atlas is None

    # Testing the prediction mode
    with patch('argparse._sys.argv', 
               ['localise', '--predict', 
                '--subject', '/path/to/subject', 
                '--seed', '/path/to/subject/seed.nii.gz',
                '--masks', '/path/to/subject/masks',
                '--tracts', '/path/to/subject/tracts',
                '--structure', 'vim',
                '--data_type', 'single32',
                '--out', 'output.nii.gz']):
        args = parse_arguments()

        assert args.predict
        assert args.subject == '/path/to/subject'
        assert args.masks == '/path/to/subject/masks'
        assert args.tracts == '/path/to/subject/tracts'
        assert args.out == 'output.nii.gz'
        assert args.seed == '/path/to/subject/seed.nii.gz'
        assert args.structure == 'vim'
        assert args.data_type == 'single32'
        assert args.hemisphere is None
        assert args.atlas == 'default'

    stderr = StringIO()
    sys.stderr = stderr
    # Testing the prediction mode with target_list missing
    with pytest.raises(SystemExit):
        with patch('argparse._sys.argv', 
                   ['localise', '--predict', 
                    '--subject', '/path/to/subject', 
                    '--atlas', '/path/to/subject/atlas.nii.gz',
                    '--seed', 'seed.nii.gz',
                    '--masks', 'roi',
                    '--model', 'model.pth',
                    '--tracts', 'tracts',
                    '--hemisphere', 'right',
                    '--out', 'output']):
            args = parse_arguments()
    error_message = stderr.getvalue()
    expected_message = "Please specify the list of tract-density maps when using a custom model."
    assert expected_message in error_message
    
    # Testing the prediction mode with default atlas
    with pytest.raises(SystemExit):
        with patch('argparse._sys.argv', 
                ['localise', '--predict', 
                    '--subject', '/path/to/subject', 
                    '--seed', 'seed.nii.gz',
                    '--masks', 'roi',
                    '--data', 'data.npy',
                    '--hemisphere', 'right',
                    '--model', 'model.pth',
                    '--out', 'output']):
            args = parse_arguments()
    error_message = stderr.getvalue()
    expected_message = "Please specify an atlas (group-average proability map) for prediction when using a custom model."
    assert expected_message in error_message

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
