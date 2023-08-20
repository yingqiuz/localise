#!/usr/bin/env python
import os
import argparse, textwrap, logging
import pkg_resources
from pathlib import Path
from localise.load import load_data, load_features, ShuffledDataLoader
from localise.train import train, train_with_val, train_without_val
from localise.predict import apply_pretrained_model
from localise.utils import save_nifti, get_subjects


#PKG_PATH = pkg_resources.get_distribution('localise').location
PKG_PATH = Path(__file__).parent.parent

def parse_arguments():
    p = argparse.ArgumentParser(description="Localise")
    
    # Train or Predict
    p.add_argument('--train', action='store_true', help='Train the model')
    p.add_argument('--predict', action='store_true', help='Predict with the model')

    predict_group = p.add_argument_group('Prediction mode')
    train_group = p.add_argument_group('Training mode')
    
    # shared arguments  
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
    p.add_argument('--spatial', action='store_true', required=False,
                   help='Use conditional random field (recommended).')
    p.add_argument('--verbose', '-v', action='store_true', help="Increase output verbosity")
    
    # predict group
    predict_group.add_argument('--structure', '-r', required=False, dest='structure', type=str,
                               help=textwrap.dedent('''\
                                Structure to be localised.
                                if in the --predict mode, should be the name of the structure.
                                '''))

    predict_group.add_argument('--data_type', '-t', type=str, dest='data_type', required=False, 
                               help=textwrap.dedent('''\
                                    Data_type (or modality). can be singleshell, resting-state...
                                    '''))
    predict_group.add_argument('--out', '-o', required=False, type=str, dest='out', 
                                help=textwrap.dedent('''\
                                Output filename for the localised structure. 
                                '''))
    predict_group.add_argument('--model', required=False, type=str, dest='model', 
                               help='Filename of the pre-trained model.')
    
    # training mode args
    train_group.add_argument('--label', required=False, type=str, dest='label', 
                             help=textwrap.dedent('''\
                                Path to the training labels of the structure (required for training).
                                For example, if the subject folder is /path/to/subject001,
                                and the path to the atlas file is 
                                /path/to/subject001/roi/left/labels.nii.gz,
                                you should feed in --atlas=roi/left/labels.nii.gz.
                                This file must be in the same space as the connectivity features.
                                '''))
    train_group.add_argument('--out_model', required=False, type=str, dest='out_model', 
                             help=textwrap.dedent('''\
                             Output filename of the trained model.
                             '''))
    train_group.add_argument('--epochs', '-e', required=False, type=int, dest='epochs', 
                             default=100,
                             help='Number of epochs for training.')

    args = p.parse_args()

    if args.predict == args.train:
        p.error('Exactly one of --predict or --train must be provided.')

    if args.predict and args.out is None:
        p.error("Please specify the output filename.")
        
    if args.train and args.out_model is None:
        p.error('Please specify the filename for the trained model.')
        
    if args.train and args.label is None:
        p.error('Please specify the training labels.')
        
    if args.data is None and args.target_path is None:
        p.error('At least one of --data or --target_path must be provided.')

    return args


def predict_mode(subject, mask, structure, 
                 target_path, target_list, 
                 data, atlas, out, model, spatial, data_type):

    logging.info('Predict mode on.\n')
    subjects = get_subjects(subject)

    if model is None:
        # error checking
        if structure is None:
            raise ValueError('When using the default model, you must specify the structure.')
        if data_type is None:
            raise ValueError('When using the default model, you must specify the data_type.')

        logging.info(f'Using the default model for {structure} on {data_type}.')
        # load the default model.
        model_dir = os.path.join(PKG_PATH, 'resources', 'models', structure, data_type)
        model_name = f'{structure}_crf_model.pth' if spatial else f'{structure}_model.pth'
        model = os.path.join(model_dir, model_name)

        if not os.path.exists(model):
            raise ValueError(f'We dont have a pretrained model for {structure} {data_type}.')

        # checking whether or not to use default
        if data is None and target_list is None:
            # load default target list
            logging.info('Using default target list.')
            target_list_fname = os.path.join(PKG_PATH, 'resources', 'data', 
                                             f'{structure}_default_target_list.txt')
            with open(target_list_fname, 'r') as f:
                target_list = [line.strip() for line in f]

        else:
            logging.info(f'Please make sure your data or target_list matches the order of the default target list {target_list_fname}.')

    else:
        logging.info(f'Using the model stored in {model}.')

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

    predictions = apply_pretrained_model(data, model, spatial=spatial)

    # save to nii files
    for subject, prediction in zip(subjects, predictions):
        save_nifti(prediction, os.path.join(subject, mask), os.path.join(subject, out))

    return predictions


def train_mode(subject, mask, label, target_path,
               target_list, data, atlas, out_model, 
               spatial, epochs):
    
    logging.info('Training mode on.\n')
    subjects = get_subjects(subject)
    
    if data is None and target_list is None:
        raise ValueError('Please specify --target_list or --data.')
    
    data = [
        load_data(
            subject=subject, 
            mask=mask, 
            label_name=label,
            target_path=target_path, 
            target_list=target_list, 
            data=data, 
            atlas=atlas
        ) 
        for subject in subjects
    ]
    
    dataloader = ShuffledDataLoader(data)
    model = train_without_val(dataloader, n_epochs=epochs, 
                              spatial_model=spatial, 
                              model_save_path=out_model)
    
    return model
    

if __name__ == "__main__":
    args = parse_arguments()
    if args.predict:
        predict_mode(args.subject, args.mask, args.structure, 
                     args.target_path, args.target_list, 
                     args.data, args.atlas, args.out, 
                     args.model, args.spatial, args.data_type)
    elif args.train:
        train_mode(args.subject, args.mask, args.label, 
                   args.target_path, args.target_list, 
                   args.data, args.atlas, args.out_model, 
                   args.spatial, args.epochs)
    else:
        raise ValueError("Exactly one of --train or --predict must be specified.")