#!/usr/bin/env python
import os
import argparse, textwrap, logging
from localise.load import load_data, load_features
from localise.train import train_loop, val_loop, train, apply_pretrained_model
from utils import save_nifti, get_subjects


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
    p.add_argument('--model', '-m', required=False, type=str, dest='model', 
                          help=textwrap.dedent('''\
                          Filename of the trained model.
                          '''))
    p.add_argument('--model', '-m', required=False, type=str, dest='model', 
                   help=textwrap.dedent('Filename of the trained model.'))
    p.add_argument("--spatial", action="store_true", help="Use conditional random field.")
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


def predict_mode(subject, mask, structure, 
                 target_path, target_list, 
                 data, atlas, out, model, spatial):

    logging.info('Predict mode on.\n')
    subjects = get_subjects(subject)

    # checking whether or not to use default
    if data is None and target_list is None:
        # load default target list
        logging.info('Using default target list.')
        with open(f'{structure}_default_target_list.txt', 'r') as f:
            target_list = [line.strip() for line in f]

        if model is not None:
            logging.info(f'''\
                         You are using the default target list for {structure} but not the default model.
                         Please double check to make sure the model matches your target list.
                         ''')
    if data is not None or target_list is not None:
        if model is None:
            logging.info(f'''\
                         You are using the default model for {structure}.
                         Please make sure your data or target_list matches the default target list.
                         ''')

    # load connectivity features
    data = [
        load_features(
            subject=subject, 
            mask=mask, 
            target_path=target_path, 
            target_list=target_list, 
            data=data, 
            atlas=atlas
        ) 
        for subject in subjects
    ]

    # make predictions
    if model is None:
        model_dir = os.path.join('resources', 'models')
        model_name = f'{structure}_crf_model.pth' if spatial else f'{structure}_model.pth'
        model = os.path.join(model_dir, model_name)

    predictions = apply_pretrained_model(data, model, spatial=spatial)

    # save to nii files
    for subject, prediction in zip(subjects, predictions):
        save_nifti(prediction, os.path.join(subject, mask), os.path.join(subject, out))


if __name__ == "__main__":
    args = parse_arguments()
    if args.predict:
        predict_mode(args.subject, args.mask, args.structure, 
                     args.target_path, args.target_list, 
                     args.data, args.atlas, args.out, args.model, args.spatial)