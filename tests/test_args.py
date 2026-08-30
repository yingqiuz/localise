import pytest
from io import StringIO
from unittest.mock import Mock

from localise.cli import parse_args, build_parser


class TestTrainParsing:
    """Test parsing of the train subcommand."""

    def test_train_with_data(self):
        """Test training mode with pre-saved data."""
        args = parse_args([
            'train',
            '--seed', '/path/to/subject/seed.nii.gz',
            '--masks', '/path/to/subject/roi',
            '--labels', '/path/to/subject/roi/labels/label.nii.gz',
            '--out-model', 'model.pth',
            '--hemisphere', 'left',
            '--data', '/path/to/subject/data.npy',
            '--atlas', '/path/to/subject/atlas.nii.gz'
        ])

        assert args.command == 'train'
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

    def test_train_with_tracts_list(self):
        """Test training mode with tract-density maps list."""
        args = parse_args([
            'train',
            '--seed', '/path/to/subject/seed.nii.gz',
            '--masks', '/path/to/subject/roi',
            '--labels', '/path/to/subject/roi/labels/label.nii.gz',
            '--out-model', 'model.pth',
            '--hemisphere', 'left',
            '--tracts', '/path/to/subject/tracts/',
            '--tracts-list', 'tracts_list.txt',
            '--epochs', '1000'
        ])

        assert args.command == 'train'
        assert args.seed == '/path/to/subject/seed.nii.gz'
        assert args.masks == '/path/to/subject/roi'
        assert args.labels == '/path/to/subject/roi/labels/label.nii.gz'
        assert args.out_model == 'model.pth'
        assert args.tracts == '/path/to/subject/tracts/'
        assert args.tracts_list == 'tracts_list.txt'
        assert args.epochs == 1000
        assert args.atlas is None

    def test_train_with_spatial_and_verbose(self):
        """Test training mode with spatial and verbose flags."""
        args = parse_args([
            'train',
            '--seed', '/path/to/subject/seed.nii.gz',
            '--masks', '/path/to/subject/roi',
            '--labels', '/path/to/subject/labels.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/path/to/data.npy',
            '--spatial',
            '--verbose'
        ])

        assert args.spatial is True
        assert args.verbose is True

    def test_short_argument_forms(self):
        """Test short argument forms."""
        args = parse_args([
            'train',
            '--masks', '/masks',
            '--labels', '/labels.nii.gz',
            '--seed', '/seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/data.npy',
            '-v'  # short form for verbose
        ])
        assert args.verbose is True

    def test_type_conversions(self):
        """Test type conversions work correctly."""
        args = parse_args([
            'train',
            '--masks', '/masks',
            '--labels', '/labels.nii.gz',
            '--seed', '/seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/data.npy',
            '--epochs', '500'  # string that should convert to int
        ])
        assert isinstance(args.epochs, int)
        assert args.epochs == 500

    def test_atlas_default_handling_in_training(self):
        """Test atlas default handling in training mode."""
        args = parse_args([
            'train',
            '--masks', '/masks',
            '--labels', '/labels.nii.gz',
            '--seed', '/seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/data.npy',
            '--structure', 'vim',
            '--atlas'
        ])
        # atlas stays 'default'; modes resolves it via the masks folder
        assert args.atlas == 'default'

    def test_atlas_explicit_in_training(self):
        """Test explicit atlas in training mode (doesn't get overridden)."""
        args = parse_args([
            'train',
            '--masks', '/masks',
            '--labels', '/labels.nii.gz',
            '--seed', '/seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', '/data.npy',
            '--atlas', '/custom/atlas.nii.gz'
        ])
        assert args.atlas == '/custom/atlas.nii.gz'


class TestPredictParsing:
    """Test parsing of the predict subcommand."""

    def test_predict_default_model(self):
        """Test prediction mode with default model."""
        args = parse_args([
            'predict',
            '--masks', '/path/to/subject/masks',
            '--tracts', '/path/to/subject/tracts',
            '--structure', 'vim',
            '--model', 'single32',
            '--out', 'output',
            '--atlas'
        ])

        assert args.command == 'predict'
        assert args.masks == '/path/to/subject/masks'
        assert args.tracts == '/path/to/subject/tracts'
        assert args.out == 'output'
        assert args.structure == 'vim'
        assert args.hemisphere is None
        assert args.atlas == 'default'
        assert args.model == 'single32'

    def test_predict_custom_model(self):
        """Test prediction mode with custom model."""
        args = parse_args([
            'predict',
            '--seed', '/path/to/subject/seed.nii.gz',
            '--masks', '/path/to/subject/masks',
            '--model', 'custom_model.pth',
            '--atlas', '/path/to/atlas.nii.gz',
            '--data', '/path/to/data.npy',
            '--out', 'output',
            '--hemisphere', 'right'
        ])

        assert args.command == 'predict'
        assert args.model == 'custom_model.pth'
        assert args.atlas == '/path/to/atlas.nii.gz'
        assert args.seed == '/path/to/subject/seed.nii.gz'
        assert args.hemisphere == 'right'

    def test_both_hemispheres_default(self):
        """When hemisphere is not specified, it is None (both hemispheres)."""
        args = parse_args([
            'predict',
            '--masks', '/masks',
            '--tracts', '/tracts',
            '--structure', 'vim',
            '--out', 'output'
        ])
        assert args.hemisphere is None

    def test_hemisphere_choices(self):
        """Test hemisphere argument validation."""
        args = parse_args([
            'train',
            '--masks', 'masks',
            '--labels', 'labels.nii.gz',
            '--seed', 'seed.nii.gz',
            '--out-model', 'model.pth',
            '--data', 'data.npy',
            '--hemisphere', 'right'
        ])
        assert args.hemisphere == 'right'

        with pytest.raises(SystemExit):
            parse_args(['train', '--hemisphere', 'invalid'])


class TestValidationErrors:
    """Test validation of argument combinations."""

    def test_missing_subcommand(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_train_validation_errors(self):
        # Missing labels
        with pytest.raises(SystemExit):
            parse_args(['train', '--masks', '/masks', '--data', '/data.npy'])

        # Missing seed
        with pytest.raises(SystemExit):
            parse_args(['train', '--masks', '/masks',
                        '--labels', '/labels.nii.gz', '--data', '/data.npy'])

        # Missing out-model
        with pytest.raises(SystemExit):
            parse_args(['train', '--masks', '/masks',
                        '--labels', '/labels.nii.gz', '--seed', '/seed.nii.gz',
                        '--data', '/data.npy'])

    def test_predict_validation_errors(self):
        # Missing out directory
        with pytest.raises(SystemExit):
            parse_args(['predict', '--masks', '/masks', '--structure', 'vim',
                        '--data', '/data.npy'])

        # Default model: missing structure
        with pytest.raises(SystemExit):
            parse_args(['predict', '--masks', '/masks', '--out', '/output',
                        '--data', '/data.npy'])

        # No --model: the shipped default model is used at run time
        args = parse_args(['predict', '--masks', '/masks', '--structure', 'vim',
                           '--out', '/output', '--data', '/data.npy'])
        assert args.model is None

        # A shipped-model name is accepted without --seed; a path is custom
        args = parse_args(['predict', '--masks', '/masks', '--structure', 'vim',
                           '--model', '2mm', '--out', '/output',
                           '--data', '/data.npy'])
        assert args.model == '2mm'

        # Custom model: missing seed
        with pytest.raises(SystemExit):
            parse_args(['predict', '--masks', '/masks', '--model', '/model.pth',
                        '--atlas', '/atlas.nii.gz', '--out', '/output',
                        '--data', '/data.npy'])

        # Custom model: missing atlas (default not allowed)
        with pytest.raises(SystemExit):
            parse_args(['predict', '--masks', '/masks', '--model', '/model.pth',
                        '--seed', '/seed.nii.gz', '--out', '/output',
                        '--data', '/data.npy', '--atlas'])

        # Missing masks and seed
        with pytest.raises(SystemExit):
            parse_args(['predict', '--structure', 'vim',
                        '--out', '/output',
                        '--data', '/data.npy'])

    def test_connectivity_features_validation(self):
        """Either tracts or data must be specified."""
        with pytest.raises(SystemExit):
            parse_args(['train', '--masks', '/masks',
                        '--labels', '/labels.nii.gz', '--seed', '/seed.nii.gz',
                        '--out-model', 'model.pth'])

    def test_error_messages_content(self, capsys):
        """Test that error messages contain expected text."""
        with pytest.raises(SystemExit):
            parse_args(['predict', '--masks', '/masks', '--model', '/model.pth',
                        '--seed', '/seed.nii.gz', '--out', '/output',
                        '--data', '/data.npy', '--atlas'])
        assert 'Custom models require --atlas path.' in capsys.readouterr().err


class TestOtherSubcommands:
    """Smoke tests for the prepare and connectivity-driven subcommands."""

    def test_prepare_masks_parsing(self):
        args = parse_args([
            'prepare-masks', '--ref', 't1.nii.gz', '--warp', 'warp.nii.gz',
            '--out', 'sub01', '--aparc', 'aparc.nii.gz', '--structure', 'vim'
        ])
        assert args.command == 'prepare-masks'
        assert args.structure == 'vim'
        assert args.brainmask is None

    def test_prepare_tracts_parsing(self):
        args = parse_args([
            'prepare-tracts', '--bpx', 'sub01/dMRI.bedpostX',
            '--masks', 'sub01/masks', '--out', 'sub01/tracts',
            '--structure', 'vim'
        ])
        assert args.command == 'prepare-tracts'
        assert args.hemisphere is None  # both hemispheres by default
        assert args.gpu is False

    def test_connectivity_driven_parsing(self):
        args = parse_args([
            'connectivity-driven', '--target1', 'm1.nii.gz',
            '--target2', 'cerebellum.nii.gz', '--out', 'vim_cd.nii.gz'
        ])
        assert args.command == 'connectivity-driven'
        assert args.thr1 == 50
        assert args.thr == 70
        assert args.target3 is None


class TestValidateArgs:
    """Test the validation functions directly."""

    def test_validate_train_direct_call(self):
        from localise.args import validate_train

        parser = Mock()

        args = Mock()
        args.tracts = '/path/tracts'
        args.data = None
        args.labels = '/labels.nii.gz'
        args.seed = '/seed.nii.gz'
        args.out_model = 'model.pth'
        args.atlas = 'default'
        args.structure = 'vim'
        args.masks = '/masks'

        # Should not raise any errors
        validate_train(parser, args)

        # atlas stays 'default'; modes resolves it via the masks folder
        assert args.atlas == 'default'


class TestCliEndToEnd:
    """Run the predict subcommand through the real CLI entry point."""

    def test_cli_predict_with_shipped_model(self, tmp_path):
        import numpy as np
        import nibabel as nib
        from pathlib import Path
        from localise.cli import main

        seed = (Path(__file__).parent / 'test_data' / '100610' / 'roi' /
                'left' / 'tha_small.nii.gz')
        n_voxels = int((nib.load(str(seed)).get_fdata() > 0).sum())
        data = tmp_path / 'features.npy'
        np.save(data, np.random.default_rng(0)
                .random((160, n_voxels)).astype('float32'))

        out = tmp_path / 'out'
        main(['predict', '--seed', str(seed), '--data', str(data),
              '--structure', 'vim', '--spatial', '--hemisphere', 'left',
              '--out', str(out)])

        assert (out / 'left' / 'probmap.nii.gz').exists()

    def test_cli_predict_with_missing_model_file(self, tmp_path, capsys):
        from localise.cli import main

        with pytest.raises(SystemExit):
            main(['predict', '--seed', 'seed.nii.gz', '--data', 'd.npy',
                  '--model', str(tmp_path / 'nope.pth'),
                  '--hemisphere', 'left', '--out', str(tmp_path)])
        assert 'does not exist' in capsys.readouterr().err
