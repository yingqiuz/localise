import torch
from torch.nn import Linear
from localise.forward import FlexibleClassifier, MLP
import logging
from typing import Iterable, Tuple, Union

def apply_pretrained_model(
    data: Iterable, 
    model_save_path: str, 
    model: Union[str, torch.nn.Module] = "Linear", 
    spatial: bool = True
) -> list:
    """
    Load a pre-trained PyTorch model and apply it to new data.

    Args:
        data: Iterable of tuples containing the new data (FlattenedCRFBatchTensor, torch.tensor).
        model_save_path: Path where the trained model is saved.
        model: Type of model to use. Options: "Linear", "MLP", or PyTorch model class. Default: "Linear".
        spatial: If True, use spatial model. Default: True.

    Returns:
        list: Predictions for the input data.
    """

    # get dimensions
    X = next(iter(data))
    n_features, n_classes, n_kernels = X.X.shape[1], X.K, X.f.shape[0]

    model_class = {
        "Linear": lambda: Linear(n_features, n_classes),
        "MLP": lambda: MLP(n_features, 2, n_classes),
    }.get(model, lambda: model)

    # Define a model
    m = FlexibleClassifier(
        model_class(),
        n_classes=n_classes,
        n_kernels=n_kernels,
        is_crf=spatial
    )

    state_dict = torch.load(model_save_path, weights_only=True)

    # catch model/data mismatches up front with actionable errors:
    # strict=False would silently skip the CRF weights on a spatial mismatch
    if spatial and 'smooth_weight' not in state_dict:
        raise ValueError(
            f'{model_save_path} contains no CRF weights, but spatial '
            'prediction was requested; drop the --spatial flag or use '
            'a spatial model.'
        )
    if not spatial and 'smooth_weight' in state_dict:
        # shipped non-spatial models carry leftover CRF params, so this is
        # not necessarily an error - but flag a possibly forgotten --spatial
        logging.warning(
            f'{model_save_path} contains CRF weights that will be ignored; '
            'if it was trained as a spatial model, add the --spatial flag.'
        )
    if 'layer.weight' in state_dict:
        n_saved = state_dict['layer.weight'].shape[1]
        if n_saved != n_features:
            raise ValueError(
                f'{model_save_path} expects {n_saved} features per voxel, '
                f'but the data provides {n_features}. Check that the tract '
                'list and the atlas/prior option match the ones used '
                'to train the model.'
            )

    # Load the saved model parameters
    m.load_state_dict(state_dict, strict=False)
    
    # Ensure model is in evaluation mode
    m.eval()
    
    # Now we can use the model for prediction on the new data
    with torch.no_grad():
        return [m(X) for X in data]

def apply_model(data: Iterable, model: torch.nn.Module) -> list:
    """_summary_

    Parameters:
    data (Iterables): Iterable of tuples 
        containing the new data, (FlattenedCRFBatchTensor, torch.tensor)
    model (torch.nn.Module): The trained PyTorch model

    Returns:
        predictions: list
    """
    model.eval()
    with torch.no_grad():
        return [model(X) for X in data]
