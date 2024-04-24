#!/usr/bin/env python
import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor
from localise.utils import run_command
from pathlib import Path
from fsl.wrappers import flirt, fnirt, fslmaths, applyxfm


RESOURCES_PATH = Path(__file__).parent.parent / 'resources' / 'MNI_roi'

def parse_mask_arguments():
    p = argparse.ArgumentParser(description="Prepare anatomical masks for tractography.")
    
    # required arguments
    p.add_argument('--ref', '-r', required=True, type=str, dest='ref',
                   help="The reference image, usually in the individual T1 native space")
    p.add_argument('--warp', '-w', required=True, type=str, dest='warp', 
                   help="The warp field that maps the MNI standard space to individual T1 space")
    
    # optional arguments
    p.add_argument('--out', '-o', required=False, type=str, dest='out', default='roi',
                   help="The output directory to store the anatomical masks. "\
                        "if not provided, create a folder called roi in the same directory "\
                        "as the reference image")
    
    p.add_argument('--aparc', '-a', required=True, type=str, dest='aparc', 
                   help="The cortical segmentation file from Freesurfer, "\
                        "either in *.mgz or *.nii.gz/*.nii - it is strongly recommended to have this file")
    
    p.add_argument('--brain_mask', '-m', required=False, type=str, dest='brain_mask', 
                   help="The binary brain mask in the reference space")
    
    p.add_argument('--njobs', '-n', required=False, type=str, dest="njobs", 
                   help="The number of jobs to run in parallel.")
    
    args = p.parse_args()
    
    # do some checking
    args_to_check = [args.ref, args.warp, args.aparc]
    if args.brain_mask is not None:
        args_to_check.append(args.brain_mask)
        
    for arg in args_to_check:
        if not os.path.exists(arg):
            p.error(f'{arg} does not exist.')

    if not os.path.isdir(args.out):
        os.mkdir(args.out)
        
    for hemi in ['left', 'right']:
        hemi_dir = os.path.join(args.out, hemi)
        if not os.path.isdir(hemi_dir):
            os.mkdir(hemi_dir)
    
    return args


def prep_aparc(aparc, out, ref, flirt_cmd, fslmaths_cmd, njobs):
    # Ensure FREESURFER_HOME is set if needed
    if aparc.endswith('.mgz') and "FREESURFER_HOME" not in os.environ:
        raise Exception("Can't find the freesurfer home directory. Make sure FREESURFER_HOME is set.")
    
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(aparc))[0]
    
    # Construct output paths
    out_aparc = os.path.join(out, f"{base_name}_reshape_to_ref.nii.gz")
    
    if aparc.endswith('.mgz'):
        # Convert the mgz file into nii
        print(f"Convert {aparc} to NIfTI...")
        cmd = [os.path.join(os.environ['FREESURFER_HOME'], 'bin', 'mri_convert'), aparc, out_aparc]
        run_command(cmd)
        
        # Apply transformation
        cmd = [flirt_cmd, '-in', out_aparc, '-ref', ref, '-out', out_aparc, 
               '-applyxfm', '-usesqform', '-interp', 'nearestneighbour']
        run_command(cmd)
        
    elif aparc.endswith(('.nii', '.nii.gz')):
        cmd = [flirt_cmd, '-in', aparc, '-ref', ref, '-out', out_aparc, 
               '-applyxfm', '-usesqform', '-interp', 'nearestneighbour']
        run_command(cmd)
        
    else:
        raise ValueError("The aparc file must have a .mgz, .nii, or .nii.gz extension.")
    
    # list of commands to separate cortical ROIs
    cmds = []
    for hemi, offset in zip(['left', 'rgith'], [11100, 12100]):
        cmd = [fslmaths_cmd, out_aparc, '-thr', 11100, '-uthr', 12175, '-bin', f'{out}/{hemi}/cortex']
        cmds.append(cmd)
        for k in range(1, 76):
            cmd = [fslmaths_cmd, out_aparc, '-thr', str(k + offset), '-uthr', str(k + offset), '-bin', f'{out}/{hemi}/{kk}']
    
    # cerebellum and csf
    cmds.append([fslmaths_cmd, out_aparc, '-thr', '6', 
                 '-uthr', '8', '-bin', f'{out}/left/cerebellum'])
    cmds.append([fslmaths_cmd, out_aparc, '-thr', '45', 
                 '-uthr', '47', '-bin', f'{out}/right/cerebellum'])
    cmds.append([fslmaths_cmd, out_aparc, '-thr', '7', 
                 '-uthr', '7', '-bin', f'{out}/left/cerebellum_target'])
    cmds.append([fslmaths_cmd, out_aparc, '-thr', '46', 
                 '-uthr', '46', '-bin', f'{out}/right/cerebellum_target'])
    cmds.append([fslmaths_cmd, out_aparc, '-thr', '24', 
                 '-uthr', '24', '-bin', f'{out}/csf'])

    with ProcessPoolExecutor(max_workers=njobs) as executor:
        executor.map(run_command, cmds)
          
    return out_aparc

def prep_subcortical(out, ref, warp, aparc, applywarp_cmd, njobs):
    cmds = []
    for hemi in ['left', 'right']:
        # wm ROIs
        wm_tracts = ['scpct_mask', 'str_mask', 'atr_mask', 
                    'ar_mask', 'or_mask', 'fx_mask', 
                    'to_precentral_mask', 'to_postcentral_mask']
        wm_nums = [15, 10, 8, 8, 10, 10, 12, 12]
        for wm, nums in zip(wm_tracts, wm_nums):
            for k in range(1, nums+1):
                cmd = [applywarp_cmd, '-r', ref, '-w', warp, '--interp=nn', 
                       '-o', os.path.join(out, hemi, f'{wm}{k}.nii.gz'), 
                       '-i', os.path.join(RESOURCES_PATH, hemi, f'{wm}{k}.nii.gz')]
                cmds.append(cmd)
        # other ROIs
        rois = ['tha', 'vim', 'exclusion']
        # if aparc is not specified, use standard aparc
        for roi in rois:
            cmd = [applywarp_cmd, '-r', ref, '-w', warp, '--interp=nn', 
                   '-o', os.path.join(out, hemi, f'{roi}.nii.gz'), 
                   '-i', os.path.join(RESOURCES_PATH, hemi, f'{roi}.nii.gz')]
            cmds.append(cmd)    
    return 1

def prep_brainmask(out, ref, warp, applywarp_cmd):
    brain_mask = os.path.join(out, 'brain_mask.nii.gz')
    cmd = [applywarp_cmd, '-r', ref, '-i', 
           os.path.join(RESOURCES_PATH, 'MNI152_T1_1mm_brain_mask.nii.gz'), 
           '-w', warp, '-o', brain_mask, '--interp=nn']
    run_command(cmd)
    return brain_mask

def prep_scpct(out, ref, warp, applywarp_cmd, fslmaths_cmd, brain_mask):
    for hemi in ['left', 'right']:
        run_command([applywarp_cmd, '-r', ref, '-w', warp, 
                     '-o', os.path.join(out, hemi, 'SCPCT'), 
                     '-i', os.path.join(RESOURCES_PATH, hemi, 'SCPCT')])
        run_command([fslmaths_cmd, os.path.join(out, hemi, 'SCPCT'), 
                     '-thr', '0.0001', '-bin', 
                     os.path.join(out, hemi, 'SCPCT_bin')])
        run_command([fslmaths_cmd, brain_mask, '-rem', 
                     os.path.join(out, hemi, 'SCPCT_bin'), 
                     '-bin', '-rem', os.path.join(out, hemi, 'tha'), 
                     '-bin', os.path.join(out, hemi, 'stop_for_scpct')])
    return 1

if __name__ == "__main__":
    args = parse_mask_arguments()