#!/usr/bin/env python3
"""
Create tracts using probabilistic tractography.

This module runs FSL's probtrackx2 to generate tract density maps
for brain structure localization.

Copyright (C) 2025 University of Oxford
Written by Ying-Qiu Zheng (Python rewrite)
"""

import argparse
import sys
import textwrap
import shlex
from pathlib import Path
from localise.utils import (
    get_absolute_path,
    run_fsl_command,
    check_fsl_environment,
    check_fsl_sub_queues,
    find_mask_file,
    fsl_bin
)


# canonical target order: this is the feature order the localise models expect,
# matching resources/data/*_default_target_list.txt (cortical parcels first,
# then white-matter tracts, then - for vim - the SCPCT targets)
CANONICAL_TARGET_ORDER = (
    [str(k) for k in range(1, 76)]
    + [f"ar_mask{i}" for i in range(1, 9)]
    + [f"or_mask{i}" for i in range(1, 11)]
    + [f"str_mask{i}" for i in range(1, 11)]
    + [f"atr_mask{i}" for i in range(1, 9)]
    + [f"fx_mask{i}" for i in range(1, 11)]
    + [f"to_precentral_mask{i}" for i in range(1, 13)]
    + [f"to_postcentral_mask{i}" for i in range(1, 13)]
)
SCPCT_TARGET_ORDER = [f"scpct_mask{i}" for i in range(1, 16)]


def write_tracts_list(out_dir, structure=None):
    """Write the tract-density file list, in canonical target order.

    The resulting tracts_list.txt is picked up automatically by
    `localise predict` and `localise train`, guaranteeing that the feature
    order matches the one the models were trained with.
    """
    targets = CANONICAL_TARGET_ORDER + (
        SCPCT_TARGET_ORDER if structure == 'vim' else []
    )
    tracts_list_file = Path(out_dir) / "tracts_list.txt"
    with open(tracts_list_file, 'w') as f:
        for target in targets:
            f.write(f"seeds_to_{target}.nii.gz\n")
    return str(tracts_list_file)


def create_target_lists(masks_dir):
    """Create target mask lists for tractography."""
    masks_path = Path(masks_dir)
    
    # Create general targets list
    targets_file = masks_path / "targets.txt"
    
    with open(targets_file, 'w') as f:
        # AR masks
        ar_masks = [f"{masks_dir}/ar_mask{i}.nii.gz" for i in range(1, 9)]
        f.write(" ".join(ar_masks) + "\n")
        
        # OR masks  
        or_masks = [f"{masks_dir}/or_mask{i}.nii.gz" for i in range(1, 11)]
        f.write(" ".join(or_masks) + "\n")
        
        # STR masks
        str_masks = [f"{masks_dir}/str_mask{i}.nii.gz" for i in range(1, 11)]
        f.write(" ".join(str_masks) + "\n")
        
        # ATR masks
        atr_masks = [f"{masks_dir}/atr_mask{i}.nii.gz" for i in range(1, 9)]
        f.write(" ".join(atr_masks) + "\n")
        
        # FX masks
        fx_masks = [f"{masks_dir}/fx_mask{i}.nii.gz" for i in range(1, 11)]
        f.write(" ".join(fx_masks) + "\n")
        
        # M1 masks
        m1_masks = [f"{masks_dir}/to_precentral_mask{i}.nii.gz" for i in range(1, 13)]
        f.write(" ".join(m1_masks) + "\n")
        
        # S1 masks
        s1_masks = [f"{masks_dir}/to_postcentral_mask{i}.nii.gz" for i in range(1, 13)]
        f.write(" ".join(s1_masks) + "\n")
        
        # Cortical parcels
        cortical_masks = [f"{masks_dir}/{i}.nii.gz" for i in range(1, 76)]
        f.write(" ".join(cortical_masks) + "\n")
    
    return str(targets_file)


def create_scpct_target_list(masks_dir):
    """Create SCPCT-specific target list."""
    masks_path = Path(masks_dir)
    targets_file = masks_path / "targets_scpct.txt"  # Different filename to avoid conflicts
    
    with open(targets_file, 'w') as f:
        # SCPCT masks
        scpct_masks = [f"{masks_dir}/scpct_mask{i}.nii.gz" for i in range(1, 16)]
        f.write(" ".join(scpct_masks) + "\n")
        
        # Cerebellum target
        f.write(f"{masks_dir}/cerebellum_target.nii.gz\n")
    
    return str(targets_file)


def create_brain_mask_if_needed(bpx_dir):
    """Create brain mask if it doesn't exist."""
    brain_mask_path = Path(bpx_dir) / "nodif_brain_mask"
    
    # Check if brain mask already exists
    existing_mask = find_mask_file(brain_mask_path)
    if existing_mask:
        return existing_mask
    
    print(f"Brain mask not found. Creating one from {bpx_dir}/mean_fsumsamples...")
    
    # Create brain mask from mean_fsumsamples
    mean_samples = Path(bpx_dir) / "mean_fsumsamples"
    output_mask = str(brain_mask_path)
    
    run_fsl_command([
        fsl_bin('fslmaths'), str(mean_samples), '-thr', '0', '-bin', output_mask
    ])

    return output_mask + '.nii.gz'


def resolve_seed_path(seed, masks_dir, hemisphere):
    """Resolve seed mask path for specific hemisphere."""
    masks_path = Path(masks_dir)
    
    # If not found, try without extension
    hemi_seed_base = masks_path / hemisphere / Path(seed)
    if hemi_seed_base.exists():
        return str(hemi_seed_base)
    else:
        # If not found, try without extension
        found_seed = find_mask_file(masks_path / hemisphere / Path(seed).stem)
        if found_seed:
            return found_seed
    
    raise FileNotFoundError(f"Seed mask '{seed}' not found in {masks_path / hemisphere}")


def build_probtrackx_command(fsl_dir, prog, seed, samples, brain_mask, out_dir, 
                           target_masks, avoid_mask, stop_mask, warp=None, 
                           ref=None, ptx_opts=None):
    """Build the probtrackx2 command."""
    cmd = [f"{fsl_dir}/bin/{prog}"]
    
    # Basic options
    cmd.extend(['-x', seed])
    cmd.extend(['-s', f"{samples}/merged"])
    cmd.extend(['-m', brain_mask])
    cmd.extend(['--dir', out_dir])
    
    # Default parameters (can be overridden by ptx_opts)
    default_opts = {
        '--verbose': '1',
        '--cthr': '0.2',
        '--nsteps': '2000', 
        '--steplength': '0.5',
        '--nsamples': '5000',
        '--fibthresh': '0.01',
        '--distthresh': '0.0',
        '--sampvox': '0.0'
    }
    
    # Add default options first
    for opt, value in default_opts.items():
        cmd.extend([opt, value])
    
    # Transformations
    if warp and len(warp) >= 2:
        cmd.extend(['--xfm', warp[0]])
        cmd.extend(['--invxfm', warp[1]])
        
    if ref:
        cmd.extend(['--seedref', ref])
    else:
        cmd.extend(['--seedref', seed])
    
    # Add fixed flags
    cmd.extend(['-l', '--onewaycondition', '--forcedir', '--pd', '--opd', 
                '--os2t', '--modeuler'])
    
    # Add masks
    cmd.extend(['--targetmasks', target_masks])
    cmd.extend(['--avoid', avoid_mask])
    cmd.extend(['--stop', stop_mask])
    
    # Override with custom options if provided
    if ptx_opts and Path(ptx_opts).exists():
        with open(ptx_opts, 'r') as f:
            custom_opts = shlex.split(f.read().strip())
            cmd.extend(custom_opts)
    
    return cmd


def execute_probtrackx(cmd, log_dir, job_name, gpu=False, dependency=None):
    """Run probtrackx directly, or submit it via fsl_sub when queues exist.

    Returns the job id when submitted to a queue, None when run locally.
    """
    if check_fsl_sub_queues():
        log_path = Path(log_dir) / 'logs'
        log_path.mkdir(parents=True, exist_ok=True)
        sub_cmd = ['fsl_sub', '-N', job_name, '-l', str(log_path)]
        if gpu:
            sub_cmd.extend(['--coprocessor', 'cuda'])
        if dependency:
            sub_cmd.extend(['-j', dependency])
        result = run_fsl_command(sub_cmd + cmd)
        job_id = result.stdout.strip()
        print(f"Submitted {job_name} to the queue (job {job_id}).")
        return job_id

    run_fsl_command(cmd)
    return None


def create_tracts(bpx, masks, out, hemisphere, structure=None, warp=None, ref=None,
                 ptx_opts=None, gpu=False, seed=None):
    """
    Main function to create tracts using probabilistic tractography.
    
    Args:
        bpx: Path to bedpostX folder
        masks: Path to masks folder  
        out: Path to output folder
        hemisphere: Hemisphere for seed mask (left/right)
        structure: Structure to be localised (e.g., vim, lgn)
        warp: Optional transformation matrices [ref2diff, diff2ref]
        ref: Optional path to reference image
        ptx_opts: Optional probtrackx2 options file
        gpu: Whether to use GPU version
        seed: Optional custom seed mask
    """
    # Check FSL environment
    fsl_dir = check_fsl_environment()
    
    # Validate required paths
    bpx_path = Path(bpx)
    masks_path = Path(masks)
    
    if not bpx_path.exists():
        raise FileNotFoundError(f"BedpostX directory does not exist: {bpx}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks directory does not exist: {masks}")
    
    # Validate warp if provided
    if warp and len(warp) != 2:
        raise ValueError("Warp parameter must contain exactly 2 transformation matrices")
    
    # Get absolute paths
    masks_abs = get_absolute_path(masks)
    bpx_abs = get_absolute_path(bpx)
    
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    out_abs = get_absolute_path(out)
    
    # Resolve seed mask path
    seed_path = resolve_seed_path(seed, masks_abs, hemisphere)
    
    # Create or find brain mask
    brain_mask = create_brain_mask_if_needed(bpx_abs)
    
    # Choose program (GPU or CPU)
    prog = "probtrackx2_gpu" if gpu else "probtrackx2"
    
    print(f"Creating tracts from seed mask {seed_path} using {prog}...")
    
    # Determine hemisphere-specific masks directory
    hemi_masks_path = Path(masks_abs) / hemisphere
    hemi_masks_path.mkdir(parents=True, exist_ok=True)
    hemi_masks_dir = str(hemi_masks_path)
    
    hemi_out_path = Path(out_abs) / hemisphere
    hemi_out_path.mkdir(parents=True, exist_ok=True) 
    hemi_out_dir = str(hemi_out_path)  
    
    # Create target lists and run first tractography (general targets)
    print("\n=== Running tractography for general white matter targets ===")
    targets_file = create_target_lists(hemi_masks_dir)
    
    # Build command for general tractography
    # Find avoid and stop masks with proper extensions
    avoid_mask = find_mask_file(Path(hemi_masks_dir) / "cerebellum")
    stop_mask = find_mask_file(Path(masks_abs) / "cortex")
    
    if not avoid_mask:
        raise FileNotFoundError(f"Cerebellum mask not found in {hemi_masks_dir}")
    if not stop_mask:
        raise FileNotFoundError(f"Cortex mask not found in {masks_abs}")
    
    cmd = build_probtrackx_command(
        fsl_dir=fsl_dir,
        prog=prog,
        seed=seed_path,
        samples=bpx_abs,
        brain_mask=brain_mask,
        out_dir=hemi_out_dir,
        target_masks=targets_file,
        avoid_mask=avoid_mask,
        stop_mask=stop_mask,
        warp=warp,
        ref=ref,
        ptx_opts=ptx_opts
    )
    
    # Run first tractography (submitted via fsl_sub when queues are available)
    job_id = execute_probtrackx(
        cmd, hemi_out_dir, f"ptx_{structure or 'seed'}_{hemisphere}", gpu=gpu
    )

    # Create SCPCT target list and run second tractography - only needed for vim
    if structure == 'vim':
        print("\n=== Running tractography for SCPCT targets ===")
        scpct_targets_file = create_scpct_target_list(hemi_masks_dir)
        
        # Build command for SCPCT tractography
        avoid_mask_scpct = find_mask_file(Path(hemi_masks_dir) / "exclusion")
        stop_mask_scpct = find_mask_file(Path(hemi_masks_dir) / "stop_for_scpct")
        
        if not avoid_mask_scpct:
            raise FileNotFoundError(f"Exclusion mask not found in {hemi_masks_dir}")
        if not stop_mask_scpct:
            raise FileNotFoundError(f"Stop_for_scpct mask not found in {hemi_masks_dir}")
        
        cmd_scpct = build_probtrackx_command(
            fsl_dir=fsl_dir,
            prog=prog,
            seed=seed_path,
            samples=bpx_abs,
            brain_mask=brain_mask,
            out_dir=hemi_out_dir,
            target_masks=scpct_targets_file,
            avoid_mask=avoid_mask_scpct,
            stop_mask=stop_mask_scpct,
            warp=warp,
            ref=ref,
            ptx_opts=ptx_opts
        )
        
        # Run second tractography: both runs write into the same output
        # directory (fdt_paths etc.), so on a queue the second job waits for
        # the first via -j rather than running in parallel
        execute_probtrackx(
            cmd_scpct, hemi_out_dir, f"ptx_{structure}_scpct_{hemisphere}",
            gpu=gpu, dependency=job_id
        )

    # record the feature order for train/predict to pick up
    write_tracts_list(hemi_out_dir, structure=structure)

    print(f"\nTractography completed. Results saved to: {out_abs}")


DESCRIPTION = "Create tracts using probabilistic tractography"
EPILOG = textwrap.dedent("""
    Examples:
        # Process both hemispheres
        localise prepare-tracts --bpx subj001/dMRI.bedpostX --masks subj001/masks
                                --structure vim --out subj001/tracts

        # Process only left hemisphere
        localise prepare-tracts --bpx subj001/dMRI.bedpostX --masks subj001/masks
                                --structure vim --out subj001/tracts --hemisphere left

        # Use GPU and transformations
        localise prepare-tracts --bpx subj001/dMRI.bedpostX --masks subj001/masks
                                --structure vim --out subj001/tracts --gpu
                                --warp ref2diff.mat diff2ref.mat --ref subj001/T1.nii.gz

    This creates tracts for the specified structure using fiber samples
    estimated via BedpostX. The program will look for seed masks in the
    appropriate hemisphere folders (masks/left/ and masks/right/).
""")


def add_arguments(parser):
    """Add prepare-tracts arguments to an argparse parser."""
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--bpx', required=True,
        help='Path to bedpostX folder'
    )
    required.add_argument(
        '--masks', required=True,
        help='Path to masks folder (e.g., the output folder of create_masks)'
    )
    required.add_argument(
        '--out', required=True,
        help='Path to output folder'
    )
    
    # Optional arguments
    parser.add_argument(
        '--structure',
        help='Structure to be localised (currently supported: vim, lgn)'
    )
    parser.add_argument(
        '--seed',
        help='Seed mask filename (e.g., tha.nii.gz, lgn_bin.nii.gz). Will be looked up in hemisphere-specific folders.'
    )
    parser.add_argument(
        '--hemisphere', choices=['left', 'right'],
        help='Hemisphere for tractography (left/right)'
    )
    parser.add_argument(
        '--warp', nargs=2, metavar=('REF2DIFF', 'DIFF2REF'),
        help='Transformation matrices between reference and diffusion space'
    )
    parser.add_argument(
        '--ref',
        help='Path to reference image (usually T1 native space)'
    )
    parser.add_argument(
        '--ptx-opts',
        help='Text file with extra probtrackx2 options to override defaults'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Use GPU version of probtrackx2'
    )

    return parser


def resolve_default_seed(args):
    """Set the default seed based on the structure if not provided."""
    if not args.seed:
        if args.structure == 'vim':
            args.seed = 'tha.nii.gz'
        elif args.structure == 'lgn':
            args.seed = 'lgn_bin.nii.gz'
        elif not args.structure:
            raise ValueError("Please specify --structure if --seed is not specified.")
        else:
            raise ValueError(f"Default seed not defined for structure: {args.structure}")
    return args


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG
    )
    add_arguments(parser)
    return resolve_default_seed(parser.parse_args())


def run(args):
    """Run tractography from parsed arguments (both hemispheres by default)."""
    resolve_default_seed(args)
    if not args.hemisphere:
        hemispheres = ['left', 'right']
    else:
        hemispheres = [args.hemisphere]
    for h in hemispheres:
        create_tracts(
            bpx=args.bpx,
            masks=args.masks,
            out=args.out,
            hemisphere=h,
            structure=args.structure,
            warp=args.warp,
            ref=args.ref,
            ptx_opts=args.ptx_opts,
            gpu=args.gpu,
            seed=args.seed
        )


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        run(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()