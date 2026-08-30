import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from localise.prepare_tracts import (
    create_target_lists,
    resolve_seed_path,
    parse_arguments,
    build_probtrackx_command,
    create_tracts,
    main
)


class TestBasicFunctions:
    """Test core functions."""
    
    def test_create_target_lists(self):
        """Test target list creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = create_target_lists(tmp_dir)
            
            targets_file = Path(tmp_dir) / "targets.txt"
            assert targets_file.exists()
            
            with open(targets_file, 'r') as f:
                content = f.read()
            assert "ar_mask1.nii.gz" in content
            assert "cortical" in content or "1.nii.gz" in content
    
    def test_resolve_seed_path(self):
        """Test seed path resolution."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            hemi_dir = Path(tmp_dir) / "left"
            hemi_dir.mkdir()
            seed_file = hemi_dir / "tha.nii.gz"
            seed_file.touch()
            
            result = resolve_seed_path("tha.nii.gz", tmp_dir, "left")
            assert result == str(seed_file)
            
            # Test not found
            with pytest.raises(FileNotFoundError):
                resolve_seed_path("missing.nii.gz", tmp_dir, "left")


class TestArgumentParsing:
    """Test argument parsing."""
    
    def test_parse_arguments_vim(self):
        """Test VIM structure parsing."""
        with patch('sys.argv', [
            'create_tracts', '--bpx', '/bpx', '--masks', '/masks', 
            '--out', '/out', '--structure', 'vim'
        ]):
            args = parse_arguments()
            assert args.structure == 'vim'
            assert args.seed == 'tha.nii.gz'
    
    def test_parse_arguments_lgn(self):
        """Test LGN structure parsing."""
        with patch('sys.argv', [
            'create_tracts', '--bpx', '/bpx', '--masks', '/masks',
            '--out', '/out', '--structure', 'lgn'
        ]):
            args = parse_arguments()
            assert args.structure == 'lgn'
            assert args.seed == 'lgn_bin.nii.gz'


class TestMainFunction:
    """Test main function."""
    
    def test_main_both_hemispheres(self):
        """Test processing both hemispheres."""
        with patch('localise.prepare_tracts.parse_arguments') as mock_parse:
            args = Mock()
            args.hemisphere = None
            args.bpx = '/bpx'
            args.masks = '/masks'
            args.out = '/out'
            args.structure = 'vim'
            args.seed = 'tha.nii.gz'
            args.warp = None
            args.ref = None
            args.ptx_opts = None
            args.gpu = False
            mock_parse.return_value = args
            
            with patch('localise.prepare_tracts.create_tracts') as mock_create:
                main()
                
                # Should be called twice (left and right)
                assert mock_create.call_count == 2
                calls = mock_create.call_args_list
                assert calls[0][1]['hemisphere'] == 'left'
                assert calls[1][1]['hemisphere'] == 'right'


# Performance and edge case tests
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_masks_directory(self):
        """Test behavior with empty masks directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bpx_dir = Path(tmp_dir) / "bpx"
            bpx_dir.mkdir()
            masks_dir = Path(tmp_dir) / "masks"
            masks_dir.mkdir()
            left_dir = masks_dir / "left"
            left_dir.mkdir()  # Empty hemisphere directory
            
            with patch('localise.prepare_tracts.check_fsl_environment', return_value='/usr/local/fsl'):
                with pytest.raises(FileNotFoundError):
                    create_tracts(
                        bpx=str(bpx_dir),
                        masks=str(masks_dir),
                        out=str(tmp_dir),
                        hemisphere="left",
                        seed="tha.nii.gz"
                    )
    
    def test_malformed_ptx_opts_file(self):
        """Test behavior with malformed probtrackx options file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("")  # Empty file
            opts_file = f.name
        
        try:
            cmd = build_probtrackx_command(
                fsl_dir="/usr/local/fsl",
                prog="probtrackx2",
                seed="/path/to/seed.nii.gz",
                samples="/path/to/samples",
                brain_mask="/path/to/brain_mask.nii.gz",
                out_dir="/path/to/output",
                target_masks="/path/to/targets.txt",
                avoid_mask="/path/to/avoid.nii.gz",
                stop_mask="/path/to/stop.nii.gz",
                ptx_opts=opts_file
            )
            
            # Should handle empty opts file gracefully
            assert len(cmd) > 0
        finally:
            Path(opts_file).unlink()

class TestWriteTractsList:
    """Test the canonical tract list written for train/predict."""

    def test_write_tracts_list_vim(self, tmp_path):
        from localise.prepare_tracts import write_tracts_list
        from localise.utils import get_resources_path

        path = write_tracts_list(tmp_path, structure='vim')
        lines = Path(path).read_text().splitlines()

        assert Path(path).name == 'tracts_list.txt'
        assert len(lines) == 160
        assert lines[0] == 'seeds_to_1.nii.gz'
        assert lines[74] == 'seeds_to_75.nii.gz'
        assert lines[75] == 'seeds_to_ar_mask1.nii.gz'
        assert lines[-1] == 'seeds_to_scpct_mask15.nii.gz'

        # the non-cortical entries must match the shipped default target list
        # (the cortical ones differ only in naming: local 1..75 vs HCP 11101..)
        default = (get_resources_path() / 'data' /
                   'vim_default_target_list.txt').read_text().splitlines()
        assert len(default) == len(lines)
        assert lines[75:] == default[75:]

    def test_write_tracts_list_non_vim(self, tmp_path):
        from localise.prepare_tracts import write_tracts_list

        path = write_tracts_list(tmp_path, structure='lgn')
        lines = Path(path).read_text().splitlines()
        # no SCPCT targets outside vim
        assert len(lines) == 145
        assert not any('scpct' in line for line in lines)
