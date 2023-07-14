import torch
from torch.nn import Linear
from localise.flatten_forward import FlexibleClassifier, MLP
import logging


def apply_pretrained_model(data, model_save_path, model="Linear", spatial_model=True):
    """
    Function to load a pre-trained PyTorch model and apply it to new data.

    Parameters:
    data (Iterables): Iterable of tuples 
        containing the new data, (FlattenedCRFBatchTensor, torch.tensor)
    model_save_path (str): Path where the trained model is saved
    model (str, optional): String representing the type of model to use. 
        Options: "Linear", "MLP", or PyTorch model class. Default: "Linear"
    spatial_model (bool, optional): If True, use spatial model. Default: True
    """

    # get dimensions
    X = data[0]
    n_features = X.X.shape[1]
    n_classes = X.K
    n_kernels = X.f.shape[0]

    # Define a model
    if model == "Linear":
        m = FlexibleClassifier(Linear(n_features, n_classes), n_classes=n_classes, 
                                          n_kernels=n_kernels, is_crf=spatial_model)
    elif model == "MLP":
        m = FlexibleClassifier(MLP(n_features, 2, n_classes), n_classes=n_classes, 
                                       n_kernels=n_kernels, is_crf=spatial_model)
    else:
        m = FlexibleClassifier(model, n_classes=n_classes, 
                                   n_kernels=n_kernels, is_crf=spatial_model)

    # Load the saved model parameters
    m.load_state_dict(torch.load(model_save_path))
    
    # Ensure model is in evaluation mode
    m.eval()
    
    # Now we can use the model for prediction on the new data
    predictions = []
    with torch.no_grad():
        for X in data:
            pred = m(X)
            predictions.append(pred)

    return predictions

def apply_model(data, model):
    """_summary_

    Parameters:
    data (Iterables): Iterable of tuples 
        containing the new data, (FlattenedCRFBatchTensor, torch.tensor)
    model (torch.nn.Module): The trained PyTorch model

    Returns:
        predictions: list
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for X in data:
            pred = model(X)
            predictions.append(pred)

    return predictions