import os, logging
from pathlib import Path
from localise.load import load_data, load_features, ShuffledDataLoader
from localise.train import train_without_val
from localise.predict import apply_pretrained_model
from localise.utils import save_nifti, get_resources_path


RESOURCES_PATH = get_resources_path()
SEEDS_DICT = {
    'vim': 'tha',
    'lgn': 'lgn_bin',
}
        
def check_values(arg):
    """
    Check if the input argument is a file or a directory.
    If it is a txt file, return a list of values.
    """
    if arg is not None:
        path = Path(arg)
        if path.suffix == '.txt':
            with open(path, 'r') as f:
                return [Path(line.strip()) for line in f]
        else:
            return [path]
    else:
        return None

def return_hemisphere(hemisphere):
    """Check hemisphere. If not specified, keep it as None."""
    if isinstance(hemisphere, str):
        if hemisphere.lower() in ['left', 'l']:
            return 'left'
        elif hemisphere.lower() in ['right', 'r']:
            return 'right'
    else:
        raise ValueError(
            f'Invalid hemisphere: {hemisphere}. '
            'Please specify left or right.'
        )

def check_prediction_params(masks, seed=None, structure=None, tracts=None, 
                 tracts_list=None, data=None, atlas=None, out=None, model=None, 
                 data_type=None, hemisphere=None, spatial=True):
    """
    Validate and process input parameters for tract analysis. (single subject)
    
    Returns:
    dict: A dictionary containing the validated and processed parameters.
    """
    params = locals()
    params['masks'] = check_values(masks)
    # check params and return path objects
    if out is None: 
        raise ValueError('Please specify the output name.')
    params['out'] = [out_path / hemisphere / 'probmap.nii.gz' 
                     for out_path in check_values(out)]

    if atlas == 'default':
        if structure is None:
            raise ValueError(
                'When using a custom model, either set `atlas=None`, '
                'or specify your own group-average probability map.'
            )
        params['atlas'] = [mask_path / hemisphere / f'{structure}.nii.gz'
                           for mask_path in params['masks']]
    elif atlas is not None:
        params['atlas'] = check_values(atlas)

    if tracts is not None:
        params['tracts'] = [tract_path / hemisphere
                            for tract_path in check_values(tracts)]
    
    if seed is None:
        if structure is None:
            raise ValueError(
                'When using a custom model, please specify the seed mask.'
            )
        params['seed'] = [mask_path / hemisphere / f'{SEEDS_DICT[structure]}.nii.gz' 
                          for mask_path in params['masks']]
    else:
        params['seed'] = check_values(seed)

    if data is None:
        if tracts is None:
            raise ValueError(
                'Please specify path to the tract-density maps '
                'or use the presaved data.'
            )
        if tracts_list is None and structure is None:
            raise ValueError(
                'When using a custom model, '
                'you should either specify the list containing the targets.'
                'or use pre-saved data.'
            )
    else:
        params['data'] = check_values(data)

    if model is None:
        params['model'] = _handle_default_model(
            structure=structure, data_type=data_type, hemisphere=hemisphere, 
            spatial=spatial, atlas=atlas
        )
        # checking whether or not to use default target list
        if params['data'] is None and params['tracts_list'] is None:
            params['tracts_list'] = (RESOURCES_PATH / 'data' / 
                                     f'{structure}_default_target_list.txt')
            logging.info('Using default tract list.')
        # the user is using the default model but not the default target list
        else:
            # either tracts_list is none (when using presaved data) 
            # or a specified list
            logging.info('Please make sure your data or tracts_list matches ' + \
                         f'the order of the default target list.')
    else:
        logging.info(f'Using the custom model stored in {params["model"]}.')
        params['model'] = Path(params['model']).resolve()
    return params

def _handle_default_model(structure, data_type, hemisphere, spatial, atlas):
    """handle the case when a default model is used"""
    if structure is None or data_type is None:
        raise ValueError('When using the default model, you must specify '
                         'both the structure to be localised and the data_type.')

    logging.info(f'Using the default model for localising {structure} on {data_type}...')

    model_name = (
        f'{"spatial_model" if spatial else "model"}_with_prior.pth' 
        if atlas is not None 
        else f'{"spatial_model" if spatial else "model"}.pth'
    )
    model_path = (RESOURCES_PATH / 'models' / structure / data_type / 
             hemisphere / model_name)

    if not model_path.exists():
        raise ValueError(f"We haven't implemented a pretrained model for {structure} on {data_type}.")

    return model_path

def predict_mode(masks, seed=None, structure=None, tracts=None, 
                 tracts_list=None, data=None, atlas=None, out=None, model=None, 
                 spatial=True, data_type=None, hemisphere=None, verbose=True):
    """Main function for prediction mode."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info('Predict mode on. \n')
    ## subjects = check_values(subject)
    # check params
    params = check_prediction_params(
        masks=masks, seed=seed, structure=structure, tracts=tracts, 
        tracts_list=tracts_list, data=data, atlas=atlas, out=out, model=model, 
        data_type=data_type, hemisphere=hemisphere, spatial=spatial
    )

    # if data is not specified, load the data from the target list
    if params['data'] is None:
        with open(params['tracts_list'], 'r') as f:
            target_list = [line.strip() for line in f]

    data = load_features(
        masks=params['masks'], 
        tracts=params['tracts'], 
        target_list=target_list, 
        data=params['data'], 
        atlas=params['atlas'],
        power=[2, 1, 0.5]
    )

    prediction = apply_pretrained_model(data, params['model'], spatial_model=spatial)

    logging.info('Localise done. Now saving results...')
    # save to nii files
    save_nifti(prediction.detach().numpy()[:, -1], params['seed'], params['out'])

    logging.info('Done.')

    return prediction

def check_training_params(masks, labels, tracts=None, tracts_list=None, 
                          seed=None, data=None, atlas=None, out_model=None, 
                          hemisphere=None, epochs=100):
    """Validate and process input parameters for training mode."""
    params = locals()
    params['hemisphere'] = return_hemisphere(hemisphere)
    params['masks'] = Path(masks).resolve()
    params['labels'] = Path(labels).resolve()
    
    if seed is None:
        raise ValueError('Please specify the seed mask '
                         'when training a custom model.')
    params['seed'] = Path(seed).resolve()
    
    if out_model is None:
        raise ValueError('Please specify the output model name.')
    params['out_model'] = Path(out_model)
    
    if tracts is not None:
        params['tracts'] = Path(tracts) / params['hemisphere']
    
    if data is None:
        if tracts is None or tracts_list is None:
            raise ValueError(
                'Please specify path to the tract-density maps '
                'and the list of targetsor use the presaved data.'
            )
    else:
        params['data'] = Path(data).resolve()

    if atlas is not None:
        params['atlas'] = Path(atlas).resolve()
    return params
    
def train_mode(masks, labels, tracts=None, tracts_list=None, 
               seed=None, data=None, atlas=None, out_model=None, 
               spatial=True, hemisphere=None, epochs=100, verbose=True):
    """Main function for training mode."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info('Training mode on.\n')
    params = check_training_params(
        masks=masks, labels=labels, tracts=tracts, tracts_list=tracts_list, 
        seed=seed, data=data, atlas=atlas, out_model=out_model, 
        hemisphere=hemisphere, epochs=epochs
    )

    data = load_data(
        masks=params['masks'], 
        labels=params['labels'],
        tracts=params['tracts'], 
        tracts_list=tracts_list, 
        data=params['data'], 
        atlas=params['atlas']
    )
    
    dataloader = ShuffledDataLoader(data)
    model = train_without_val(dataloader, n_epochs=epochs, 
                              spatial_model=spatial, 
                              model_save_path=params['out_model'])
    
    return model
