#!/usr/bin/env python
import os
import argparse, textwrap, logging
from localise.load import load_data, load_features
from localise.train import train_loop, val_loop, train, apply_pretrained_model
from utils import save_nifti


def parse_arguments():
    p = argparse.ArgumentParser(description="Localise")
    #required = p.add_argument_group('required arguments')
    #optional = p.add_argument_group('additional options')
    
    # required arguments
    p.add_argument("--train", action="store_true", help="Train the model")
    p.add_argument("--predict", action="store_true", help="Predict with the model")    
    p.add_argument('--subject', '-s', required=True, type=str, 
                          dest='subject', help=textwrap.dedent('''\
                              Path to the subject directory, 
                              e.g., /path/to/subj001, or a txt file containing
                              the paths to subject folders. The txt file should look like
                              /path/to/subj001
                              /path/to/subj002
                              ...
                              '''))
    p.add_argument('--mask', '-m', required=True, type=str, dest='mask', 
                          help=textwrap.dedent('''\
                          Path to the binary seed mask, relative to the subject folder.
                          For example, if the subject folder is /path/to/subject001,
                          and the path to the binary mask is /path/to/subject001/roi/left/tha.nii.gz,
                          you should feed in --mask=roi/left/tha.nii.gz.
                          '''))
    p.add_argument('--structure', '-s', required=True, dest='structure', type=str,
                   help=textwrap.dedent('''\
                       Structure to be localised.
                       if in the --predict mode, should be the name of the structure.
                       if in the --train model, should be the NIfTI file, providing 
                       a rough position of the structure. 
                       '''))
    p.add_argument('--data_type', '-t', type=str, dest='data_type', required=True, 
                   help=textwrap.dedent('''\
                       Data_type (or modality). can be singleshell, rs...
                       '''))
    
    # optional arguments
    p.add_argument('--target_path', '-p', required=False, type=str, dest='target_path', 
                          help=textwrap.dedent('''\
                          Path to the folder that contains connectivity features, 
                          relative to the subject folder.
                          For example, if the subject folder is /path/to/subject001,
                          and the path to the folder is /path/to/subject001/stremlines/left/,
                          you should feed in --data=streamlines/left.
                          '''))
    p.add_argument('--target_list', '-l', required=False, type=str, dest='target_list', 
                          help=textwrap.dedent('''\
                          A txt file that contains streamline distribution files, 
                          The txt file should look like
                          seeds_to_target1.nii.gz
                          seeds_to_target2.nii.gz
                          ...
                          '''))
    p.add_argument('--data', '-d', required=False, type=str, dest='data', 
                          help=textwrap.dedent('''\
                          Path to the *.npy file of connectivity features, relative to the subject folder.
                          For example, if the subject folder is /path/to/subject001,
                          and the path to the *.npy file is 
                          /path/to/subject001/stremlines/left/features.npy,
                          you should feed in --data=streamlines/left/features.npy.
                          '''))
    p.add_argument('--atlas', '-a', required=False, type=str, dest='atlas', 
                          help=textwrap.dedent('''\
                          Path to the atlas of the structure to be localised.
                          For example, if the subject folder is /path/to/subject001,
                          and the path to the atlas file is 
                          /path/to/subject001/roi/left/atlas.nii.gz,
                          you should feed in --atlas=roi/left/atlas.nii.gz.
                          This file must be in the same space as the connectivity features.
                          '''))
    p.add_argument('--out', '-o', required=False, type=str, dest='out', 
                          help=textwrap.dedent('''\
                          Output filename for the localised structure. 
                          '''))
    p.add_argument('--out_model', '-o', required=False, type=str, dest='out_model', 
                          help=textwrap.dedent('''\
                          Output filename of the trained model.
                          '''))
    p.add_argument('--verbose', '-v', action='store_true', help="Increase output verbosity")

    args = p.parse_args()

    if args.train == args.predict: # this is True either if both are False, or both are True
        p.error("Exactly one of --train or --predict must be specified.")

    if args.predict and args.out is None:
        p.error("Please specify the output filename.")
        
    if args.train and args.out_model is None:
        p.error('Please specify the filename for the trained model.')
        
    if args.data is None and args.target_path is None:
        p.error('At least one of --data or --target_path must be provided.')

    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.predict:
        logging.info('Predict mode on.\n')
        # load data
        if os.path.isfile(args.subject):
            with open(args.subject, 'r') as file:
                subjects = [line.strip() for line in file]
        elif os.path.isdir(args.subject):
            subjects = [args.subject]
        else:
            raise ValueError('Please specify the correct subject dir or txt file.')
        structure = args.structure
        if args.target_list is None:
            # load default target list
            with open(f'{structure}_default_target_list.txt', 'r') as f:
                target_list = [line.strip() for line in file]
        data = [load_features(subject=subject, 
                              mask=args.mask, 
                              target_path=args.target_path, 
                              target_list=target_list, 
                              data=args.data, 
                              atlas=args.atlas) 
                for subject in subjects]

        # make predictions
        predictions = apply_pretrained_model(data, f'{structure}_model.pth')
        
        # save to nii files
        for subject, prediction in zip(subjects, predictions):
            save_nifti(prediction, os.path.join(subject, args.mask, args.out))