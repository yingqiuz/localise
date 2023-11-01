import os, logging
from pathlib import Path
from localise.load import load_data, load_features, ShuffledDataLoader
from localise.train import train_without_val
from localise.predict import apply_pretrained_model
from localise.utils import save_nifti


PKG_PATH = Path(__file__).parent.parent

def get_subjects(subject_path):
    """Load subjects from file or directory."""
    if os.path.isfile(subject_path):
        with open(subject_path, 'r') as f:
            return [line.strip() for line in f]
    elif os.path.isdir(subject_path):
        return [subject_path]
    else:
        raise ValueError(f'Invalid subject path: {subject_path}. Please specify a correct subject dir or txt file.')

def return_hemisphere(hemisphere):
    ## a function to check hemisphere
    if isinstance(hemisphere, str):
        if hemisphere.lower() in ['left', 'l']:
            hemisphere = 'left'
        elif hemisphere.lower() in ['right', 'r']:
            hemisphere = 'right'
        else:
            raise ValueError(f'Invalid hemisphere: {hemisphere}. Please specify left or right.')
    else:
        raise ValueError(f'Invalid hemisphere: {hemisphere}. Please specify left or right.')

    return hemisphere

def check_params(mask, mask_dir, target_dir, data, hemisphere):
    # create mask name
    if mask_dir is None:
        raise ValueError('Please specify the directory for anatomical masks (relative to the subject folder).')

    mask = os.path.join(mask_dir, hemisphere, mask)

    # create target_path
    if target_dir is None:
        raise ValueError('Please specify the directory for target masks (relative to the subject folder).')

    target_path = os.path.join(target_dir, hemisphere)

    if data is not None:
        data = os.path.join(target_dir, hemisphere, data)
    
    return mask, mask_dir, target_path, data

def predict_mode(subject, mask, mask_dir=None, structure=None, target_dir=None, target_list=None, 
                 data=None, atlas='default', out=None, model=None, spatial=True, 
                 data_type=None, hemisphere=None):

    logging.info('Predict mode on.\n')
    subjects = get_subjects(subject)
    hemisphere = return_hemisphere(hemisphere)
    mask, mask_dir, target_path, data = check_params(mask, mask_dir, target_dir, data, hemisphere)

    # create output name
    if out is None: 
        raise ValueError('Please specify the output name.')

    if atlas == 'default':
        if structure is None:
            raise ValueError('Please specify the structure (--structure, -s) if using the default atlases.')
        # create default atlas path
        atlas = os.path.join(mask_dir, hemisphere, f'{structure}.nii.gz')

    if model is None:
        # error checking
        if structure is None:
            raise ValueError('When using the default model, you must specify the structure.')
        if data_type is None:
            raise ValueError('When using the default model, you must specify the data_type.')

        logging.info(f'Using the default model for {structure} on {data_type}.')
        
        # load the default model.
        model_dir = os.path.join(PKG_PATH, 'resources', 'models', structure, data_type, hemisphere)
        model_name = 'spatial_model.pth' if spatial else 'model.pth'
        model = os.path.join(model_dir, model_name)

        if not os.path.exists(model):
            raise ValueError(f'We dont have a pretrained model for {structure} {data_type}.')

        target_list_fname = os.path.join(PKG_PATH, 'resources', 'data', 
                                         f'{structure}_default_target_list.txt')
        # checking whether or not to use default
        if data is None and target_list is None:
            # load default target list
            logging.info('Using default target list.')
            
            with open(target_list_fname, 'r') as f:
                target_list = [line.strip() for line in f]

        else:
            logging.info(f'Please make sure your data or target_list matches the order of the default target list {target_list_fname}.')

    else:
        logging.info(f'Using the model stored in {model}.')

        # check errors. either specify --data, or specify both --target_path and --target_list
        if data is None:
            if target_path is None:
                raise ValueError("Please specify --target_path if you didn't specify --data")
            if target_list is None:
                raise ValueError("Please specify --target_list if you didn't specify --data when you are not using the default model.")        

    # load connectivity features
    if not spatial:
        atlas = None

    data = [
        load_features(
            subject=subject, 
            mask_name=mask, 
            target_path=target_path, 
            target_list=target_list, 
            data=data, 
            atlas=atlas,
            power=[2, 1, 0.5]
        ) 
        for subject in subjects
    ]

    predictions = apply_pretrained_model(data, model, spatial_model=spatial)

    # save to nii files
    for subject, prediction in zip(subjects, predictions):
        save_nifti(prediction.detach().numpy()[:, -1], os.path.join(subject, mask), os.path.join(subject, out))

    return predictions


def train_mode(subject, mask, label, mask_dir=None, target_dir=None,
               target_list=None, data=None, atlas=None, out_model=None, 
               spatial=True, hemisphere=None, epochs=100):
    
    logging.info('Training mode on.\n')
    subjects = get_subjects(subject)
    hemisphere = return_hemisphere(hemisphere)
    mask, mask_dir, target_path, data = check_params(mask, mask_dir, target_dir, data, hemisphere)

    if data is None and target_list is None:
        raise ValueError("Please specify --target_list if you didn't specify --data.")
    
    data = [
        load_data(
            subject=subject, 
            mask_name=mask, 
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