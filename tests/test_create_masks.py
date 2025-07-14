import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from localise.create_masks import (
    create_cortical_masks,
    create_warp_masks,
    create_scpct_masks,
    process_aparc_file,
    execute_commands,
    create_masks,
    parse_arguments,
    main
)


class TestMaskCreationFunctions:
    """Test mask creation functions."""
    
    def test_create_cortical_masks(self):
        """Test cortical mask creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            aparc_path = "/path/to/aparc.nii.gz"
            commands_file = tmp_path / "commands.txt"
            
            # Create directory structure
            (tmp_path / "masks" / "left").mkdir(parents=True)
            (tmp_path / "masks" / "right").mkdir(parents=True)
            
            create_cortical_masks(aparc_path, tmp_path, str(commands_file))
            
            # Check that commands file was created
            assert commands_file.exists()
            
            # Read and verify some commands
            with open(commands_file, 'r') as f:
                content = f.read()
            
            # Check for expected commands
            assert "fslmaths" in content
            assert aparc_path in content
            assert "cerebellum" in content
            assert "cortex" in content
            # Check for hemisphere-specific masks
            assert "masks/left/" in content
            assert "masks/right/" in content
    
    def test_create_warp_masks(self):
        """Test warp mask creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            commands_file = tmp_path / "commands.txt"
            
            # Create initial commands file
            commands_file.touch()
            
            create_warp_masks(
                ref_path="/path/to/ref.nii.gz",
                warp_path="/path/to/warp.nii.gz",
                out_dir=tmp_path,
                structure="vim",
                resources_dir="/path/to/resources",
                commands_file=str(commands_file)
            )
            
            # Check that commands were appended
            with open(commands_file, 'r') as f:
                content = f.read()
            
            # Check for expected commands
            assert "applywarp" in content
            assert "vim" in content
            assert "tha" in content  # VIM-specific
            assert "scpct_mask" in content
            assert "str_mask" in content
            assert "exclusion" in content
    
    def test_create_warp_masks_lgn(self):
        """Test warp mask creation for LGN structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            commands_file = tmp_path / "commands.txt"
            commands_file.touch()
            
            create_warp_masks(
                ref_path="/path/to/ref.nii.gz",
                warp_path="/path/to/warp.nii.gz",
                out_dir=tmp_path,
                structure="lgn",
                resources_dir="/path/to/resources",
                commands_file=str(commands_file)
            )
            
            with open(commands_file, 'r') as f:
                content = f.read()
            
            # Check for LGN-specific commands
            assert "lgn" in content
            assert "lgn_bin" in content
            # Should not contain VIM-specific commands
            assert "vim" not in content
            assert "tha" not in content
    
    def test_create_scpct_masks_vim(self):
        """Test SCPCT mask creation for VIM."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "vim").mkdir()
            
            result = create_scpct_masks(
                ref_path="/path/to/ref.nii.gz",
                warp_path="/path/to/warp.nii.gz",
                out_dir=tmp_path,
                brainmask_path="/path/to/brain.nii.gz",
                resources_dir="/path/to/resources",
                structure="vim",
                fsl_dir="/usr/local/fsl"
            )
            
            # Check that commands file was created
            commands_file = tmp_path / "vim" / "create_masks_scpct_cmds"
            assert commands_file.exists()
            
            with open(commands_file, 'r') as f:
                content = f.read()
            
            assert "SCPCT" in content
            assert "stop_for_scpct" in content
            assert "/path/to/brain.nii.gz" in content
    
    def test_create_scpct_masks_non_vim(self):
        """Test SCPCT mask creation for non-VIM structure."""
        result = create_scpct_masks(
            ref_path="/path/to/ref.nii.gz",
            warp_path="/path/to/warp.nii.gz",
            out_dir=Path("/tmp"),
            brainmask_path=None,
            resources_dir="/path/to/resources",
            structure="lgn",
            fsl_dir="/usr/local/fsl"
        )
        
        # Should return None for non-VIM structures
        assert result is None


class TestAparcProcessing:
    """Test aparc file processing."""
    
    def test_process_aparc_mgz_file(self):
        """Test processing MGZ aparc file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "masks").mkdir()
            
            aparc_path = "/path/to/aparc.mgz"
            ref_path = "/path/to/ref.nii.gz"
            
            with patch.dict(os.environ, {'FREESURFER_HOME': '/usr/local/freesurfer'}):
                with patch('localise.create_masks.run_fsl_command') as mock_run:
                    result = process_aparc_file(aparc_path, ref_path, tmp_path)
                    
                    # Should have called mri_convert and flirt
                    assert mock_run.call_count == 2
                    
                    # Check the calls
                    calls = mock_run.call_args_list
                    assert '/usr/local/freesurfer/bin/mri_convert' in calls[0][0][0]
                    assert 'flirt' in calls[1][0][0]
                    
                    # Result should be the converted file
                    assert result.endswith('aparc.nii.gz')
    
    def test_process_aparc_mgz_no_freesurfer(self):
        """Test processing MGZ file without FreeSurfer."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="FREESURFER_HOME"):
                process_aparc_file("/path/to/aparc.mgz", "/path/to/ref.nii.gz", Path("/tmp"))
    
    def test_process_aparc_nii_file(self):
        """Test processing NII aparc file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "masks").mkdir()
            
            aparc_path = "/path/to/aparc.nii.gz"
            ref_path = "/path/to/ref.nii.gz"
            
            with patch('localise.create_masks.run_fsl_command') as mock_run:
                result = process_aparc_file(aparc_path, ref_path, tmp_path)
                
                # Should have called flirt only
                assert mock_run.call_count == 1
                assert 'flirt' in mock_run.call_args[0][0]
                
                # Result should be the processed file
                assert result.endswith('aparc.nii.gz')
    
    def test_process_aparc_invalid_file(self):
        """Test processing invalid aparc file."""
        with pytest.raises(ValueError, match="Invalid aparc"):
            process_aparc_file("/path/to/aparc.txt", "/path/to/ref.nii.gz", Path("/tmp"))


class TestCommandExecution:
    """Test command execution functions."""
    
    def test_execute_commands_with_queue(self):
        """Test command execution with queue system."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            commands_file = "/path/to/commands.txt"
            
            with patch('localise.create_masks.check_fsl_sub_queues', return_value=True):
                with patch('localise.create_masks.run_fsl_command') as mock_run:
                    mock_run.return_value.stdout = "job_12345\n"
                    
                    result = execute_commands(commands_file, "test_job", log_dir)
                    
                    assert result == "job_12345"
                    mock_run.assert_called_once()
                    call_args = mock_run.call_args[0][0]
                    assert 'fsl_sub' in call_args
                    assert 'test_job' in call_args
    
    def test_execute_commands_without_queue(self):
        """Test command execution without queue system."""
        commands_file = "/path/to/commands.txt"
        
        with patch('localise.create_masks.check_fsl_sub_queues', return_value=False):
            with patch('subprocess.run') as mock_run:
                result = execute_commands(commands_file, "test_job", Path("/tmp"))
                
                assert result is None
                mock_run.assert_called_once_with(['bash', commands_file], check=True)
    
    def test_execute_commands_with_dependency(self):
        """Test command execution with job dependency."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            commands_file = "/path/to/commands.txt"
            
            with patch('localise.create_masks.check_fsl_sub_queues', return_value=True):
                with patch('localise.create_masks.run_fsl_command') as mock_run:
                    mock_run.return_value.stdout = "job_12346\n"
                    
                    result = execute_commands(commands_file, "test_job", log_dir, "job_12345")
                    
                    assert result == "job_12346"
                    call_args = mock_run.call_args[0][0]
                    assert '-j' in call_args
                    assert 'job_12345' in call_args


class TestMainFunction:
    """Test the main create_masks function."""
    
    def test_create_masks_success(self):
        """Test successful mask creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create temporary files
            ref_file = Path(tmp_dir) / "ref.nii.gz"
            warp_file = Path(tmp_dir) / "warp.nii.gz"
            aparc_file = Path(tmp_dir) / "aparc.nii.gz"
            
            ref_file.touch()
            warp_file.touch()
            aparc_file.touch()
            
            with patch('localise.create_masks.check_fsl_environment', return_value='/usr/local/fsl'):
                with patch('localise.create_masks.get_resources_path', return_value='/path/to/resources'):
                    with patch('localise.create_masks.process_aparc_file', return_value=str(aparc_file)):
                        with patch('localise.create_masks.execute_commands', return_value="job_123"):
                            with patch('localise.create_masks.create_scpct_masks', return_value=None):
                                
                                # Should not raise any exceptions
                                create_masks(
                                    ref=str(ref_file),
                                    warp=str(warp_file),
                                    out=tmp_dir,
                                    aparc=str(aparc_file),
                                    structure="vim"
                                )
                                
                                # Check that directories were created
                                assert (Path(tmp_dir) / "masks" / "left").exists()
                                assert (Path(tmp_dir) / "masks" / "right").exists()
                                assert (Path(tmp_dir) / "vim" / "logs").exists()
    
    def test_create_masks_missing_files(self):
        """Test mask creation with missing input files."""
        with pytest.raises(FileNotFoundError):
            create_masks(
                ref="/nonexistent/ref.nii.gz",
                warp="/nonexistent/warp.nii.gz",
                out="/tmp",
                aparc="/nonexistent/aparc.nii.gz",
                structure="vim"
            )
    
    def test_create_masks_unsupported_structure(self):
        """Test mask creation with unsupported structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_file = Path(tmp_dir) / "ref.nii.gz"
            warp_file = Path(tmp_dir) / "warp.nii.gz"
            aparc_file = Path(tmp_dir) / "aparc.nii.gz"
            
            ref_file.touch()
            warp_file.touch()
            aparc_file.touch()
            
            with patch('localise.create_masks.check_fsl_environment'):
                with patch('localise.create_masks.get_resources_path'):
                    with pytest.raises(ValueError, match="unsupported_structure currently not supported"):
                        create_masks(
                            ref=str(ref_file),
                            warp=str(warp_file),
                            out=tmp_dir,
                            aparc=str(aparc_file),
                            structure="unsupported_structure"
                        )


class TestArgumentParsing:
    """Test argument parsing."""
    
    def test_parse_arguments_required(self):
        """Test parsing with all required arguments."""
        with patch('sys.argv', [
            'create_masks', 
            '--ref', '/path/to/ref.nii.gz',
            '--warp', '/path/to/warp.nii.gz',
            '--out', '/path/to/output',
            '--aparc', '/path/to/aparc.nii.gz',
            '--structure', 'vim'
        ]):
            args = parse_arguments()
            assert args.ref == '/path/to/ref.nii.gz'
            assert args.warp == '/path/to/warp.nii.gz'
            assert args.out == '/path/to/output'
            assert args.aparc == '/path/to/aparc.nii.gz'
            assert args.structure == 'vim'
            assert args.brainmask is None
    
    def test_parse_arguments_with_brainmask(self):
        """Test parsing with optional brainmask argument."""
        with patch('sys.argv', [
            'create_masks',
            '--ref', '/path/to/ref.nii.gz',
            '--warp', '/path/to/warp.nii.gz',
            '--out', '/path/to/output',
            '--aparc', '/path/to/aparc.nii.gz',
            '--structure', 'vim',
            '--brainmask', '/path/to/brain.nii.gz'
        ]):
            args = parse_arguments()
            assert args.brainmask == '/path/to/brain.nii.gz'
    
    def test_parse_arguments_missing_required(self):
        """Test parsing with missing required arguments."""
        with patch('sys.argv', ['create_masks', '-ref', '/path/to/ref.nii.gz']):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestMainEntryPoint:
    """Test the main entry point."""
    
    def test_main_success(self):
        """Test successful main execution."""
        with patch('localise.create_masks.parse_arguments') as mock_parse:
            mock_args = Mock()
            mock_args.ref = '/path/to/ref.nii.gz'
            mock_args.warp = '/path/to/warp.nii.gz'
            mock_args.out = '/path/to/output'
            mock_args.aparc = '/path/to/aparc.nii.gz'
            mock_args.structure = 'vim'
            mock_args.brainmask = None
            mock_parse.return_value = mock_args
            
            with patch('localise.create_masks.create_masks') as mock_create:
                main()
                
                mock_create.assert_called_once_with(
                    ref='/path/to/ref.nii.gz',
                    warp='/path/to/warp.nii.gz',
                    out='/path/to/output',
                    aparc='/path/to/aparc.nii.gz',
                    structure='vim',
                    brainmask=None
                )
    
    def test_main_error_handling(self):
        """Test main error handling."""
        with patch('localise.create_masks.parse_arguments') as mock_parse:
            mock_parse.side_effect = Exception("Test error")
            
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print') as mock_print:
                    main()
                    
                    mock_print.assert_called()
                    mock_exit.assert_called_with(1)


# Integration test class
class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow_mock(self):
        """Test the full workflow with mocked FSL commands."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup test files
            ref_file = Path(tmp_dir) / "ref.nii.gz"
            warp_file = Path(tmp_dir) / "warp.nii.gz"
            aparc_file = Path(tmp_dir) / "aparc.nii.gz"
            
            ref_file.touch()
            warp_file.touch()
            aparc_file.touch()
            
            # Mock all external dependencies
            with patch('localise.create_masks.check_fsl_environment', return_value='/usr/local/fsl'):
                with patch('localise.create_masks.get_resources_path', return_value='/path/to/resources'):
                    with patch('localise.create_masks.run_fsl_command'):
                        with patch('localise.create_masks.check_fsl_sub_queues', return_value=False):
                            with patch('subprocess.run'):
                                
                                # This should complete without errors
                                create_masks(
                                    ref=str(ref_file),
                                    warp=str(warp_file),
                                    out=str(tmp_dir),
                                    aparc=str(aparc_file),
                                    structure="vim"
                                )
                                
                                # Verify directory structure was created
                                assert (Path(tmp_dir) / "masks" / "left").exists()
                                assert (Path(tmp_dir) / "masks" / "right").exists()
                                assert (Path(tmp_dir) / "vim" / "logs").exists()
                                
                                # Verify command files were created
                                assert (Path(tmp_dir) / "vim" / "create_masks_cmds").exists()