import pytest
import sys
from unittest.mock import patch
from localise.args import parse_arguments
from io import StringIO


class TestParseArguments:
    """Test suite for parse_arguments function."""
    
    def test_train_mode_with_data(self):
        """Test training mode with pre-saved data."""
        with patch('sys.argv', [
            'localise', '--train', 
            '--subject', '/path/to/subject', 
            '--seed', '/path/to/subject/seed.nii.gz',
            '--masks', '/path/to/subject/roi',
            '--labels', '/path/to/subject/roi/labels/label.nii.gz',
            '--out-model', 'model.pth',  # Fixed: was out_model
            '--hemisphere', 'left',
            '--data', '/path/to/subject/data.npy',
            '--atlas', '/path/to/subject/atlas.nii.gz'
        ]):
            args = parse_arguments()

            assert args.train is True
            assert args.predict is False
            assert args.subject == '/path/to/subject'
            assert args.seed == '/path/to/subject/seed.nii.gz'
            assert args.masks == '/path/to/subject/roi'
            assert args.labels == '/path/to/subject/roi/labels/label.nii.gz'
            assert args.out_model == 'model.pth'
            assert args.data == '/path/to/subject/data.npy'
            assert args.atlas == '/path/to/subject/atlas.nii.gz'
            assert args.epochs == 100  # default value
            assert args.hemisphere == 'left'
            assert args.spatial is False  # default value
            assert args.verbose is False  # default value

    def test_train_mode_with_tracts_list(self):
        """Test training mode with tract-density maps list."""
        with patch('sys.argv', [
            'localise', '--train', 
            '--subject', '/path/to/subject', 
            '--seed', '/path/to/subject/seed.nii.gz',
            '--masks', '/path/to/subject/roi',
            '--labels', '/path/to/subject/roi/labels/label.nii.gz',
            '--out-model', 'model.pth',
            '--hemisphere', 'left',
            '--tracts', '/path/to/subject/tracts/',
            '--tracts-list', 'tracts_list.txt',  # Fixed: was tracts_list
            '--epochs', '1000'
        ]):
            args = parse_arguments()

            assert args.train is True
            assert args.subject == '/path/to/subject'
            assert args.seed == '/path/to/subject/seed.nii.gz'
            assert args.masks == '/path/to/subject/roi'
            assert args.labels == '/path/to/subject/roi/labels/label.nii.gz'
            assert args.out_model == 'model.pth'
            assert args.tracts == '/path/to/subject/tracts/'
            assert args.tracts_list == 'tracts_list.txt'
            assert args.epochs == 1000
            assert args.atlas is None  # gets set to structure (None) in validation

    def test_train_mode_with_spatial_and_verbose(self):
        """Test training mode with spatial and verbose flags."""
        with patch('sys.argv', [
            'localise', '--train', 
            '--subject', '/path/to/subject', 
            '--seed', '/path/to/subject/seed.nii.gz',
            '--masks', '/path/to/subject/roi',
            '--labels', '/path/to/subject/labels.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/path/to/data.npy',
            '--spatial',
            '--verbose'
        ]):
            args = parse_arguments()

            assert args.spatial is True
            assert args.verbose is True

    def test_predict_mode_default_model(self):
        """Test prediction mode with default model."""
        with patch('sys.argv', [
            'localise', '--predict', 
            '--subject', '/path/to/subject', 
            '--masks', '/path/to/subject/masks',
            '--tracts', '/path/to/subject/tracts',
            '--structure', 'vim',
            '--data-type', 'singleshell',  # Fixed: was data_type
            '--out', 'output.nii.gz'
        ]):
            args = parse_arguments()

            assert args.predict is True
            assert args.train is False
            assert args.subject == '/path/to/subject'
            assert args.masks == '/path/to/subject/masks'
            assert args.tracts == '/path/to/subject/tracts'
            assert args.out == 'output.nii.gz'
            assert args.structure == 'vim'
            assert args.data_type == 'singleshell'
            assert args.hemisphere is None
            assert args.atlas == 'default'
            assert args.model is None

    def test_predict_mode_custom_model(self):
        """Test prediction mode with custom model."""
        with patch('sys.argv', [
            'localise', '--predict', 
            '--subject', '/path/to/subject', 
            '--seed', '/path/to/subject/seed.nii.gz',
            '--masks', '/path/to/subject/masks',
            '--model', 'custom_model.pth',
            '--atlas', '/path/to/atlas.nii.gz',
            '--data', '/path/to/data.npy',
            '--out', 'output.nii.gz',
            '--hemisphere', 'right'
        ]):
            args = parse_arguments()

            assert args.predict is True
            assert args.model == 'custom_model.pth'
            assert args.atlas == '/path/to/atlas.nii.gz'
            assert args.seed == '/path/to/subject/seed.nii.gz'
            assert args.hemisphere == 'right'

    def test_hemisphere_choices(self):
        """Test hemisphere argument validation."""
        # Test valid hemisphere
        with patch('sys.argv', [
            'localise', '--train', 
            '--subject', '/path/to/subject', 
            '--seed', 'seed.nii.gz',
            '--masks', 'masks',
            '--labels', 'labels.nii.gz',
            '--out-model', 'model.pth',
            '--data', 'data.npy',
            '--hemisphere', 'right'
        ]):
            args = parse_arguments()
            assert args.hemisphere == 'right'

        # Test invalid hemisphere
        with pytest.raises(SystemExit):
            with patch('sys.argv', [
                'localise', '--train', 
                '--subject', '/path/to/subject', 
                '--hemisphere', 'invalid'
            ]):
                parse_arguments()

    # Error case tests
    def test_missing_required_arguments(self):
        """Test missing required arguments."""
        # Missing mode
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['localise']):
                parse_arguments()
        
        # Missing subject
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['localise', '--train']):
                parse_arguments()

        # Missing masks
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['localise', '--train', '--subject', '/path']):
                parse_arguments()

    def test_mutually_exclusive_modes(self):
        """Test that train and predict are mutually exclusive."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['localise', '--train', '--predict']):
                parse_arguments()

    def test_train_validation_errors(self):
        """Test training mode validation errors."""
        # Missing labels
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', [
                'localise', '--train',
                '--subject', '/path',
                '--masks', '/masks',
                '--data', '/data.npy'
            ]):
                parse_arguments()

        # Missing seed
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', [
                'localise', '--train',
                '--subject', '/path',
                '--masks', '/masks',
                '--labels', '/labels.nii.gz',
                '--data', '/data.npy'
            ]):
                parse_arguments()

        # Missing out-model
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', [
                'localise', '--train',
                '--subject', '/path',
                '--masks', '/masks',
                '--labels', '/labels.nii.gz',
                '--seed', '/seed.nii.gz',
                '--data', '/data.npy'
            ]):
                parse_arguments()

    def test_predict_validation_errors(self):
        """Test prediction mode validation errors."""
        # Missing out directory
        with pytest.raises(SystemExit):
            with patch('sys.argv', [
                'localise', '--predict',
                '--subject', '/path',
                '--masks', '/masks',
                '--structure', 'vim',
                '--data-type', 'singleshell',
                '--data', '/data.npy'
            ]):
                parse_arguments()

        # Default model: missing structure
        with pytest.raises(SystemExit):
            with patch('sys.argv', [
                'localise', '--predict',
                '--subject', '/path',
                '--masks', '/masks',
                '--out', '/output',
                '--data-type', 'singleshell',
                '--data', '/data.npy'
            ]):
                parse_arguments()

        # Default model: missing data-type
        with pytest.raises(SystemExit):
            with patch('sys.argv', [
                'localise', '--predict',
                '--subject', '/path',
                '--masks', '/masks',
                '--structure', 'vim',
                '--out', '/output',
                '--data', '/data.npy'
            ]):
                parse_arguments()

        # Custom model: missing seed
        with pytest.raises(SystemExit):
            with patch('sys.argv', [
                'localise', '--predict',
                '--subject', '/path',
                '--masks', '/masks',
                '--model', '/model.pth',
                '--atlas', '/atlas.nii.gz',
                '--out', '/output',
                '--data', '/data.npy'
            ]):
                parse_arguments()

        # Custom model: missing atlas (default not allowed)
        with pytest.raises(SystemExit):
            with patch('sys.argv', [
                'localise', '--predict',
                '--subject', '/path',
                '--masks', '/masks',
                '--model', '/model.pth',
                '--seed', '/seed.nii.gz',
                '--out', '/output',
                '--data', '/data.npy'
            ]):
                parse_arguments()

        # Custom model: missing tract data
        with pytest.raises(SystemExit):
            with patch('sys.argv', [
                'localise', '--predict',
                '--subject', '/path',
                '--masks', '/masks',
                '--model', '/model.pth',
                '--seed', '/seed.nii.gz',
                '--atlas', '/atlas.nii.gz',
                '--out', '/output'
            ]):
                parse_arguments()

    def test_connectivity_features_validation(self):
        """Test that either tracts or data must be specified."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', [
                'localise', '--train',
                '--subject', '/path',
                '--masks', '/masks',
                '--labels', '/labels.nii.gz',
                '--seed', '/seed.nii.gz',
                '--out-model', 'model.pth'
            ]):
                parse_arguments()

    def test_atlas_default_handling_in_training(self):
        """Test atlas default handling in training mode."""
        with patch('sys.argv', [
            'localise', '--train',
            '--subject', '/path',
            '--masks', '/masks',
            '--labels', '/labels.nii.gz',
            '--seed', '/seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/data.npy',
            '--structure', 'vim'
        ]):
            args = parse_arguments()
            # In training mode, if atlas is 'default', it gets set to structure
            assert args.atlas == 'vim'

    def test_atlas_explicit_in_training(self):
        """Test explicit atlas in training mode (doesn't get overridden)."""
        with patch('sys.argv', [
            'localise', '--train',
            '--subject', '/path',
            '--masks', '/masks',
            '--labels', '/labels.nii.gz',
            '--seed', '/seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/data.npy',
            '--atlas', '/custom/atlas.nii.gz'
        ]):
            args = parse_arguments()
            # Explicit atlas should not be overridden
            assert args.atlas == '/custom/atlas.nii.gz'

    def test_short_argument_forms(self):
        """Test short argument forms."""
        with patch('sys.argv', [
            'localise', '--train',
            '--subject', '/path',
            '--masks', '/masks',
            '--labels', '/labels.nii.gz',
            '--seed', '/seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/data.npy',
            '-v'  # short form for verbose
        ]):
            args = parse_arguments()
            assert args.verbose is True

    def test_type_conversions(self):
        """Test type conversions work correctly."""
        with patch('sys.argv', [
            'localise', '--train',
            '--subject', '/path',
            '--masks', '/masks',
            '--labels', '/labels.nii.gz',
            '--seed', '/seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/data.npy',
            '--epochs', '500'  # string that should convert to int
        ]):
            args = parse_arguments()
            assert isinstance(args.epochs, int)
            assert args.epochs == 500

    def test_both_hemispheres_default(self):
        """Test that when hemisphere is not specified, it's None (both hemispheres)."""
        with patch('sys.argv', [
            'localise', '--train',
            '--subject', '/path',
            '--masks', '/masks',
            '--labels', '/labels.nii.gz',
            '--seed', '/seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/data.npy'
        ]):
            args = parse_arguments()
            assert args.hemisphere is None


class TestValidateArgs:
    """Test the validation function separately."""
    
    def test_validate_args_direct_call(self):
        """Test calling _validate_args directly with mock args."""
        from unittest.mock import Mock
        from localise.args import _validate_args
        
        # Mock parser
        parser = Mock()
        
        # Test valid training args
        args = Mock()
        args.train = True
        args.predict = False
        args.tracts = '/path/tracts'
        args.data = None
        args.labels = '/labels.nii.gz'
        args.seed = '/seed.nii.gz'
        args.out_model = 'model.pth'
        args.atlas = 'default'
        args.structure = 'vim'
        
        # Should not raise any errors
        _validate_args(parser, args)
        
        # Check that atlas was modified
        assert args.atlas == 'vim'

    def test_error_messages_content(self):
        """Test that error messages contain expected text."""
        import contextlib
        
        # Capture stderr to check error messages
        f = StringIO()
        
        with contextlib.redirect_stderr(f):
            with pytest.raises(SystemExit):
                with patch('sys.argv', [
                    'localise', '--predict',
                    '--subject', '/path',
                    '--masks', '/masks',
                    '--model', '/model.pth',
                    '--seed', '/seed.nii.gz',
                    '--out', '/output',
                    '--data', '/data.npy'
                ]):
                    parse_arguments()
        
        error_output = f.getvalue()
        assert 'Custom models require explicit --atlas' in error_output
