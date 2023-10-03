import pytest
from unittest.mock import patch
from localise.args import parse_arguments
from localise.utils import create_masks, create_tracts, connectivity_driven

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


def test_create_masks():
    # Mocking the environment variable
    with patch.dict('os.environ', {'FSLDIR': 'some_value'}):
        # Mocking the os.path.isfile function to always return True
        with patch('os.path.isfile', return_value=True):
            # Mocking the subprocess.run function to do nothing
            with patch('subprocess.run') as mock_subprocess:
                ref = "some_ref_path"
                warp = "some_warp_path"
                create_masks(ref, warp)
                mock_subprocess.assert_called_once()

    # Testing for missing FSLDIR environment variable
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="FSLDIR environment variable does not exist."):
            create_masks("some_ref_path", "some_warp_path")

    # Testing for missing ref file
    with patch.dict('os.environ', {'FSLDIR': 'some_value'}):
        with patch('os.path.isfile', side_effect=[False, True]):
            with pytest.raises(ValueError, match=f"{ref} does not exist."):
                create_masks(ref, warp)

    # Testing for missing warp file
    with patch.dict('os.environ', {'FSLDIR': 'some_value'}):
        with patch('os.path.isfile', side_effect=[True, False]):
            with pytest.raises(ValueError, match=f"{warp} does not exist."):
                create_masks(ref, warp)
                
            
def test_create_tracts():
    # Mocking the environment variable
    with patch.dict('os.environ', {'FSLDIR': 'some_value'}):
        # Mocking the os.path.isdir function to always return True
        with patch('os.path.isdir', return_value=True):
            # Mocking the subprocess.run function to do nothing
            with patch('subprocess.run') as mock_subprocess:
                samples_dir = "some_samples_dir_path"
                input_dir = "some_input_dir_path"
                create_tracts(samples_dir, input_dir)
                mock_subprocess.assert_called_once()

    # Testing for missing FSLDIR environment variable
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="FSLDIR environment variable is missing."):
            create_tracts(samples_dir, input_dir)

    # Testing for missing samples_dir
    with patch.dict('os.environ', {'FSLDIR': 'some_value'}):
        with patch('os.path.isdir', side_effect=[False, True]):
            with pytest.raises(ValueError, match=f"Directory '{samples_dir}' is missing."):
                create_tracts(samples_dir, input_dir)

    # Testing for missing input_dir
    with patch.dict('os.environ', {'FSLDIR': 'some_value'}):
        with patch('os.path.isdir', side_effect=[True, False]):
            with pytest.raises(ValueError, match=f"Directory '{input_dir}' is missing."):
                create_tracts(samples_dir, input_dir)
                

def test_connectivity_driven():
    # Mocking the environment variable
    with patch.dict('os.environ', {'FSLDIR': 'some_value'}):
        # Mocking the os.path.isfile function to always return True
        with patch('os.path.isfile', return_value=True):
            # Mocking the subprocess.run function to do nothing
            with patch('subprocess.run') as mock_subprocess:
                target1 = "some_target1_path"
                target2 = "some_target2_path"
                out = "some_out_path"
                connectivity_driven(target1, target2, out)
                mock_subprocess.assert_called_once()

    # Testing for missing FSLDIR environment variable
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="FSLDIR environment variable is missing."):
            connectivity_driven(target1, target2, out)

    # Testing for missing target1 file
    with patch.dict('os.environ', {'FSLDIR': 'some_value'}):
        with patch('os.path.isfile', side_effect=[False, True]):
            with pytest.raises(ValueError, match=f"{target1} does not exist."):
                connectivity_driven(target1, target2, out)

    # Testing for missing target2 file
    with patch.dict('os.environ', {'FSLDIR': 'some_value'}):
        with patch('os.path.isfile', side_effect=[True, False]):
            with pytest.raises(ValueError, match=f"{target2} does not exist."):
                connectivity_driven(target1, target2, out)