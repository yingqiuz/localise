import torch
from torch import nn
from torch import optim
from torch.nn import Linear
from localise.flatten_forward import FlexibleClassifier, MLP
import logging


def train_loop(data, model, loss_fn, optimizer, lambda_l1, lambda_l2, print_freq=10):
    """
    Function to execute one training loop epoch.

    Parameters:
    data (iterables): Iterable of tuples 
        containing the training data, (FlattenedCRFBatchTensor, torch.tensor)
    model (nn.Module): PyTorch model to train
    loss_fn (nn.modules.loss): Loss function to optimize
    optimizer (optim.Optimizer): Optimizer to use for training
    lambda_l1 (float): L1 regularization weight
    lambda_l2 (float): L2 regularization weight
    print_freq (int, optional): Frequency to print training loss. Default: 10
    """

    size = len(data)
    model.train()

    for batch, (X, y) in enumerate(data):
        pred = model(X)
        loss = loss_fn(pred, y)

        # add L1 and L2 penalty
        layer_weights = torch.stack([x for x in model.layer.parameters() if x.dim() > 1])
        loss += lambda_l1 * torch.norm(layer_weights, 1) + lambda_l2 * torch.norm(layer_weights, 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % print_freq == 0:
            logging.info(f"loss: {loss.item():>7f}  [{batch + 1:>5d}/{size:>5d}]")


def val_loop(data, model, loss_fn):
    """
    Function to execute one validation loop epoch.

    Parameters:
    data (iterables): Iterable of tuples 
        containing the validation data, (FlattenedCRFBatchTensor, torch.tensor)
    model (nn.Module): PyTorch model to evaluate
    loss_fn (nn.modules.loss): Loss function to compute loss
    """

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in data:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    logging.info(f"Test Error: \n Avg loss: {test_loss / len(data):>8f} \n")


def train(training_data, test_data, 
          model="Linear",
          loss_fn=nn.CrossEntropyLoss(), 
          optimizer=optim.Adam, 
          n_epochs=100, 
          spatial_model=True, 
          l1_lambda=1e-3, l2_lambda=1e-3, lr=1e-3,
          print_freq=10, model_save_path=None):
    """
    Function to train a PyTorch model and evaluate it using provided training and test data.

    Parameters:
    training_data (Iterables): Iterable of tuples 
        containing the training data, (FlattenedCRFBatchTensor, torch.tensor)
    test_data (Iterables): Iterable of tuples 
        containing the test data, (FlattenedCRFBatchTensor, torch.tensor)
    model (str, optional): String representing the type of model to use. 
        Options: "Linear", "MLP", or PyTorch model class. Default: "Linear"
    loss_fn (torch.nn.modules.loss, optional): 
        Loss function. Default: CrossEntropyLoss
    optimizer (torch.optim.Optimizer, optional): 
        Optimizer to use for training. Default: Adam
    n_epochs (int, optional): Number of training epochs. Default: 100
    spatial_model (bool, optional): If True, use spatial model. Default: True
    l1_lambda (float, optional): Weight for L1 regularization. Default: 1e-3
    l2_lambda (float, optional): Weight for L2 regularization. Default: 1e-3
    lr (float, optional): Learning rate for the optimizer. Default: 1e-3
    print_freq (int, optional): Frequency of loss print statements. Default: 10
    model_save_path (str, optional): Path where the trained model will be saved. If None, the model will not be saved. Default: None

    Returns:
    m (torch.nn.Module): The trained PyTorch model
    """

    # get dimensions
    X, y = training_data[0]
    n_features = X.X.shape[1]
    n_classes = y.shape[1]
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

    optimizer = optimizer(m.parameters(), lr=lr)
    for t in range(n_epochs):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train_loop(training_data, m, loss_fn, optimizer, l1_lambda, l2_lambda, print_freq)
        val_loop(test_data, m, loss_fn)

    logging.info("Training Done!")
    
    # save the model if requested 
    if model_save_path is not None:
        torch.save(m.state_dict(), model_save_path)
        logging.info(f"Model saved to {model_save_path}")

    return m

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
    X, _ = data[0]
    n_features = X.X.shape[1]
    n_classes = X.shape[1]
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
        for X, _ in data:
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