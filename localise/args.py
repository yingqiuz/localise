import os
import argparse, textwrap


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

    p.add_argument('--seed', '-x', required=True, type=str, dest='seed', 
                   help=textwrap.dedent('''\
                          The binary seed mask (doesn't need to include the path). 
                          This script assumes that the mask can be found under 
                          the folder of anatomical masks (--mask_dir),
                          For example, if the subject folder is /path/to/subject001,
                          then the binary mask(s) should be stored under /path/to/subject001/roi/left 
                          or /path/to/subject001/roi/right, depending on the hemisphere.
                          '''))

    p.add_argument('--hemisphere', '-c', required=True, type=str, dest='hemisphere', 
                   help='Hemisphere of the structure to be localised. Can be left or right.')
    
    p.add_argument('--mask_dir', '-i', required=True, type=str, dest='mask_dir', 
                   help=textwrap.dedent('''\
                          Path to the folder that contains anatomical masks, 
                          relative to the subject folder.
                          For example, if the subject folder is /path/to/subject001,
                          and the path to the folder is /path/to/subject001/roi,
                          you should feed in --mask_dir=roi.
                          '''))

    p.add_argument('--target_dir', '-p', required=True, type=str, dest='target_dir', 
                   help=textwrap.dedent('''\
                          Path to the folder that contains connectivity features, 
                          relative to the subject folder.
                          For example, if the subject folder is /path/to/subject001,
                          and the path to the folder is /path/to/subject001/stremlines,
                          you should feed in --target_dir=streamlines.
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
                          The *.npy file of connectivity features (doesn't need to include the path).
                          This script assumes that the file can be found under the folder 
                          containing connectivity features (--target_dir).
                          For example, if the subject folder is /path/to/subject001,
                          and the folder of connectivity features is /path/to/subject001/streamlines,
                          then the file should be stored under /path/to/subject001/streamlines/left/features.npy or 
                          /path/to/subject001/streamlines/right/features.npy, depending on the hemisphere.  
                          /path/to/subject001/stremlines/left/features.npy,
                          you should feed in --data=streamlines/left/features.npy.
                          '''))

    p.add_argument('--atlas', '-a', required=False, type=str, dest='atlas', 
                          help=textwrap.dedent('''\
                          Path to the atlas (group-average proability map) of the structure to be localised, relative to the subject folder.
                          For example, if the subject folder is /path/to/subject001,
                          and the path to the atlas file is 
                          /path/to/subject001/atlases/atlas_left.nii.gz,
                          you should feed in --atlas=atlases/atlas_left.nii.gz.
                          This file must be in the same reference space as the tract-density maps.
                          This serves as a prior features to contrain predictions to be not too far away from the group-average position.
                          If you are under the predict mode and you have specified the structure to be localised (--structure, -r),
                          you can simply set --atlas=default or -a default, and the script will automatically find the atlas for you.
                          '''))
    
    p.add_argument('--spatial', action='store_true', required=False,
                   help='Use conditional random field (recommended).')
    
    p.add_argument('--verbose', '-v', action='store_true', required=False, 
                   help="Increase output verbosity")
    
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
    
    predict_group.add_argument('--model', '-m', required=False, type=str, dest='model', 
                               help='Filename of the pre-trained model.')
    
    # training mode args
    train_group.add_argument('--label', '-b', required=False, type=str, dest='label', 
                             help=textwrap.dedent('''\
                                Path to the training labels of the structure (required for training).
                                For example, if the subject folder is /path/to/subject001,
                                and the path to the atlas file is 
                                /path/to/subject001/roi/left/labels.nii.gz,
                                you should feed in --label=roi/left/labels.nii.gz.
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

    if args.train:
        if args.atlas == 'default':
            p.error('Please specify the atlas (group-average proability map) for training.')
        # under training mode, either specify the target list or the presaved data
        if args.data is None and args.target_list is None:
            p.error('Please specify the connectivity features or the target list for training.')

    if args.predict:
        if args.atlas == 'default' and args.structure is None:
            p.error('Please specify the atlas (group-average proability map) for prediction.')

        if args.structure is None and args.model is None:
            p.error('Please specify the structure to be localised or the pre-trained model for prediction.')

        if args.model is None and args.data_type is None:
            p.error('When using the default model, you must specify the data_type.')

    return args