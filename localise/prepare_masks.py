#!/usr/bin/env python3
"""
Create anatomical masks for tractography analysis.

This module creates anatomical masks in reference space for tractography
and stores them under the output folder.

Copyright (C) 2019 University of Oxford
Written by Ying-Qiu Zheng (Python rewrite)
"""

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from localise.utils import (
    get_resources_path, 
    get_absolute_path, 
    run_fsl_command, 
    check_fsl_sub_queues, 
    check_fsl_environment
)


def create_cortical_masks(aparc_path, out_dir, commands_file):
    """Create cortical masks from aparc segmentation."""
    commands = []
    
    # Cerebellum masks
    commands.extend([
        f"fslmaths {aparc_path} -thr 6 -uthr 8 -bin {out_dir}/masks/left/cerebellum",
        f"fslmaths {aparc_path} -thr 45 -uthr 47 -bin {out_dir}/masks/right/cerebellum",
        f"fslmaths {aparc_path} -thr 7 -uthr 7 -bin {out_dir}/masks/right/cerebellum_target",
        f"fslmaths {aparc_path} -thr 46 -uthr 46 -bin {out_dir}/masks/left/cerebellum_target",
        f"fslmaths {aparc_path} -thr 24 -uthr 24 -bin {out_dir}/masks/csf"
    ])
    
    # Left hemisphere cortical masks (from 11101 to 11175)
    for k in range(11101, 11176):
        kk = k - 11100
        commands.append(f"fslmaths {aparc_path} -thr {k} -uthr {k} -bin {out_dir}/masks/left/{kk}")
    
    # Right hemisphere cortical masks (from 12101 to 12175)
    for k in range(12101, 12176):
        kk = k - 12100
        commands.append(f"fslmaths {aparc_path} -thr {k} -uthr {k} -bin {out_dir}/masks/right/{kk}")
    
    # Overall cortex mask
    commands.append(f"fslmaths {aparc_path} -thr 11100 -uthr 12175 -bin {out_dir}/masks/cortex")
    
    # Write commands to file
    with open(commands_file, 'w') as f:
        for cmd in commands:
            f.write(f"$FSLDIR/bin/{cmd}\n")


def create_warp_masks(ref_path, warp_path, out_dir, structure, resources_dir, commands_file):
    """Create masks by warping standard space masks to reference space."""
    commands = []
    
    for hemi in ['left', 'right']:
        # Create white matter masks - SCPCT tract
        for i in range(1, 16):
            tract = f"scpct_mask{i}"
            commands.append(
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/{tract} "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/{tract} --interp=nn"
            )
        
        # STR masks
        for i in range(1, 11):
            tract = f"str_mask{i}"
            commands.append(
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/{tract} "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/{tract} --interp=nn"
            )
        
        # ATR masks
        for i in range(1, 9):
            tract = f"atr_mask{i}"
            commands.append(
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/{tract} "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/{tract} --interp=nn"
            )
        
        # AR masks
        for i in range(1, 9):
            tract = f"ar_mask{i}"
            commands.append(
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/{tract} "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/{tract} --interp=nn"
            )
        
        # OR masks
        for i in range(1, 11):
            tract = f"or_mask{i}"
            commands.append(
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/{tract} "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/{tract} --interp=nn"
            )
        
        # Fornix masks
        for i in range(1, 11):
            tract = f"fx_mask{i}"
            commands.append(
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/{tract} "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/{tract} --interp=nn"
            )
        
        # M1 masks
        for i in range(1, 13):
            tract = f"to_precentral_mask{i}"
            commands.append(
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/{tract} "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/{tract} --interp=nn"
            )
        
        # S1 masks
        for i in range(1, 13):
            tract = f"to_postcentral_mask{i}"
            commands.append(
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/{tract} "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/{tract} --interp=nn"
            )
        
        # Structure-specific masks
        if structure == 'vim':
            commands.extend([
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/tha "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/tha --interp=nn",
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/vim "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/vim"
            ])
        elif structure == 'lgn':
            commands.extend([
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/lgn "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/lgn",
                f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/lgn_bin "
                f"-w {warp_path} -o {out_dir}/masks/{hemi}/lgn_bin --interp=nn"
            ])
        
        # Exclusion mask (common for all structures)
        commands.append(
            f"applywarp -r {ref_path} -i {resources_dir}/MNI_roi/{hemi}/exclusion "
            f"-w {warp_path} -o {out_dir}/masks/{hemi}/exclusion --interp=nn"
        )
    
    # Write commands to file
    with open(commands_file, 'a') as f:  # Append to existing file
        for cmd in commands:
            f.write(f"$FSLDIR/bin/{cmd}\n")


def create_scpct_masks(ref_path, warp_path, out_dir, brainmask_path, resources_dir, structure, fsl_dir):
    """Create SCPCT-related masks for VIM structure."""
    if structure != 'vim':
        return
    
    commands_file = out_dir / structure / "create_masks_scpct_cmds"
    commands = ['echo "Create SCPCT-related ROIs..."']
    
    # Handle brain mask
    if not brainmask_path:
        brain_mask_out = out_dir / "masks" / "brain_mask"
        commands.append(
            f"$FSLDIR/bin/applywarp -r {ref_path} "
            f"-i {fsl_dir}/data/standard/MNI152_T1_1mm_brain_mask "
            f"-w {warp_path} -o {brain_mask_out} --interp=nn"
        )
        brainmask_path = str(brain_mask_out)
    
    # Create SCPCT masks for both hemispheres
    for hemi in ['left', 'right']:
        commands.extend([
            f"$FSLDIR/bin/applywarp -r {ref_path} "
            f"-i {resources_dir}/MNI_roi/{hemi}/SCPCT "
            f"-w {warp_path} -o {out_dir}/masks/{hemi}/SCPCT",
            f"$FSLDIR/bin/fslmaths {out_dir}/masks/{hemi}/SCPCT "
            f"-thr 0.0001 -bin {out_dir}/masks/{hemi}/SCPCT_bin",
            f"$FSLDIR/bin/fslmaths {brainmask_path} "
            f"-rem {out_dir}/masks/{hemi}/SCPCT_bin -bin "
            f"-rem {out_dir}/masks/{hemi}/tha -bin "
            f"{out_dir}/masks/{hemi}/stop_for_scpct"
        ])
    
    # Write commands to file
    with open(commands_file, 'w') as f:
        for cmd in commands:
            f.write(f"{cmd}\n")
    
    return str(commands_file)


def process_aparc_file(aparc_path, ref_path, out_dir):
    """Process aparc file and return the processed file path."""
    aparc_file = Path(aparc_path)
    aparc_basename = aparc_file.name
    
    if aparc_file.suffix == '.mgz':
        # Check for FreeSurfer
        freesurfer_home = os.environ.get('FREESURFER_HOME')
        if not freesurfer_home:
            raise EnvironmentError(
                "If you feed in the aparc.a2009s+aseg in *.mgz format, "
                "please set up FREESURFER_HOME."
            )
        
        # Convert MGZ to NII.GZ
        output_path = out_dir / "masks" / f"{aparc_basename.replace('.mgz', '.nii.gz')}"
        run_fsl_command([
            f"{freesurfer_home}/bin/mri_convert", 
            str(aparc_path), 
            str(output_path)
        ])
        
        # Register to reference space
        run_fsl_command([
            "flirt", "-in", str(output_path), "-ref", ref_path,
            "-out", str(output_path), "-applyxfm", "-usesqform",
            "-interp", "nearestneighbour"
        ])
        
        return str(output_path)
    
    elif aparc_file.suffix in ['.nii', '.gz']:
        # Register NII file to reference space
        output_path = out_dir / "masks" / aparc_basename
        run_fsl_command([
            "flirt", "-in", str(aparc_path), "-ref", ref_path,
            "-out", str(output_path), "-applyxfm", "-usesqform",
            "-interp", "nearestneighbour"
        ])
        
        return str(output_path)
    
    else:
        raise ValueError(
            "Invalid aparc.a2009s+aseg file. Please use *.mgz or *.nii.gz/*.nii file."
        )


def execute_commands(commands_file, job_name, log_dir, dependency_job=None):
    """Execute FSL commands either through queue system or directly."""
    has_queues = check_fsl_sub_queues()
    
    if has_queues:
        cmd = ['fsl_sub', '-N', job_name, '-n', '-l', str(log_dir), '-t', commands_file]
        if dependency_job:
            cmd.extend(['-j', dependency_job])
        
        result = run_fsl_command(cmd)
        return result.stdout.strip()  # Return job ID
    else:
        # Execute directly
        subprocess.run(['bash', commands_file], check=True)
        return None


def create_masks(ref, warp, out, aparc, structure, brainmask=None):
    """
    Main function to create anatomical masks.
    
    Args:
        ref: Path to reference image
        warp: Path to warp field mapping MNI to reference space
        out: Path to output folder
        aparc: Path to cortical segmentation file from FreeSurfer
        structure: Name of structure (e.g., vim, lgn)
        brainmask: Optional path to binary brain mask
    """
    # Check environment
    fsl_dir = check_fsl_environment()
    resources_dir = get_resources_path()
    
    # Validate inputs
    for file_path in [ref, warp, aparc]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{file_path} does not exist.")
    
    if brainmask and not Path(brainmask).exists():
        raise FileNotFoundError(f"Brain mask {brainmask} does not exist.")
    
    # Supported structures
    supported_structures = ['vim', 'lgn']
    if structure not in supported_structures:
        raise ValueError(f"{structure} currently not supported. "
                        f"Supported: {', '.join(supported_structures)}")
    
    # Setup output directories
    out_dir = Path(out)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "masks" / "left").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks" / "right").mkdir(parents=True, exist_ok=True)
    (out_dir / structure / "logs").mkdir(parents=True, exist_ok=True)
    
    print(f"Creating masks for localising {structure} in {ref} space...")
    
    # Get absolute path for reference
    ref_abs = get_absolute_path(ref)
    
    # Process aparc file
    processed_aparc = process_aparc_file(aparc, ref_abs, out_dir)
    
    # Create main commands file
    commands_file = out_dir / structure / "create_masks_cmds"
    
    # Create cortical masks from aparc
    create_cortical_masks(processed_aparc, out_dir, str(commands_file))
    
    # Create masks by warping standard space ROIs
    create_warp_masks(ref_abs, warp, out_dir, structure, resources_dir, str(commands_file))
    
    # Execute main commands
    job_id = execute_commands(
        str(commands_file), 
        f"create_masks_for_{structure}",
        out_dir / structure / "logs"
    )
    
    # Create SCPCT masks if needed
    scpct_commands_file = create_scpct_masks(
        ref_abs, warp, out_dir, brainmask, resources_dir, structure, fsl_dir
    )
    
    if scpct_commands_file:
        execute_commands(
            scpct_commands_file,
            "create_masks_scpct",
            out_dir / structure / "logs",
            dependency_job=job_id
        )
    
    print(f"Mask creation completed. Output directory: {out_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create anatomical masks in reference space for tractography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Example usage:
                create_masks --ref subj001/t1.nii.gz --out subj001
                             --aparc subj001/aparc.a2009s+aseg.nii.gz
                             --warp subj001/std2str_warp.nii.gz
                             --structure vim
            
            This will create necessary masks for localising vim in the reference space (t1.nii.gz).
        """)
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--ref', required=True,
        help='Path to reference image (the space you want to run localise in)'
    )
    required.add_argument(
        '--warp', required=True,
        help='Path to warp field mapping the MNI standard space to the reference space'
    )
    required.add_argument(
        '--out', required=True,
        help='Path to output folder'
    )
    required.add_argument(
        '--aparc', required=True,
        help='Path to cortical segmentation file generated by Freesurfer (must be in the reference space)'
    )
    required.add_argument(
        '--structure', required=True,
        help='Name of structure, e.g., vim, lgn, etc.'
    )
    
    # Optional arguments
    parser.add_argument(
        '--brainmask',
        help='Path to binary brain mask in the reference space'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""    
    try:
        args = parse_arguments()
        create_masks(
            ref=args.ref,
            warp=args.warp,
            out=args.out,
            aparc=args.aparc,
            structure=args.structure,
            brainmask=args.brainmask
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()