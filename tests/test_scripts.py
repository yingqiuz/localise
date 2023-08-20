import pytest
from unittest.mock import patch
from localise.scripts.localise import parse_arguments

def test_parse_arguments():

    # Testing the training mode
    with patch('argparse._sys.argv', 
               ['localise', '--train', 
                '--subject', '/path/to/subject', 
                '--mask', 'roi/mask',
                '--label', 'roi/label',
                '--out_model', 'model',
                '--target_path', 'data/left',
                '--epochs', '100']):
        args = parse_arguments()

        assert args.train
        assert args.subject == '/path/to/subject'
        assert args.mask == 'roi/mask'
        assert args.label == 'roi/label'
        assert args.out_model == 'model'
        assert args.epochs == 100

    # Testing the prediction mode
    with patch('argparse._sys.argv', 
               ['localise', '--predict', 
                '--subject', '/path/to/subject', 
                '--mask', 'roi/mask',
                '--data', 'data/data',
                '--out', 'output']):
        args = parse_arguments()

        assert args.predict
        assert args.subject == '/path/to/subject'
        assert args.mask == 'roi/mask'
        assert args.data == 'data/data'
        assert args.out == 'output'
        
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
