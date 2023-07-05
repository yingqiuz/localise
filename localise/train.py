import torch
from torch import nn
from torch import optim
import time
from functools import partial
from torch.nn import Linear
from localise.flatten_forward import FlexibleClassifier, MLP


def train_loop(data, model, loss_fn, optimizer, lambda_l1, lambda_l2, print_freq=10):
    size = len(data)

    # Set the model to training mode
    model.train()
    for batch, (X, y) in enumerate(data):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Compute penalty parameters before the loop
        layer_weights = torch.stack([x for x in model.layer.parameters() if x.dim() > 1])

        # Add L1 and L2 regularization
        l1_penalty = lambda_l1 * torch.norm(layer_weights, 1)
        l2_penalty = lambda_l2 * torch.norm(layer_weights, 2)
        loss += l1_penalty + l2_penalty

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % print_freq == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(data, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    #size = len(data)
    num_batches = len(data)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for _, (X, y) in enumerate(data):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    #correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def train(training_data, test_data, 
          model_constructor="Linear",
          loss_fn=nn.CrossEntropyLoss(), 
          optimizer_constructor=optim.Adam, 
          n_epochs=100, 
          spatial_model=True, 
          l1_lambda=1e-3, l2_lambda=1e-3, lr=1e-3,
          print_freq=10, model_save_path=None):

    # get dimensions
    X, y = training_data[0]
    n_features = X.X.shape[1]
    n_classes = y.shape[1]
    n_kernels = X.f.shape[0]

    # Define a model
    if model_constructor == "Linear":
        model = FlexibleClassifier(Linear(n_features, n_classes), n_classes=n_classes, 
                                          n_kernels=n_kernels, is_crf=spatial_model)
    elif model_constructor == "MLP":
        model = FlexibleClassifier(MLP(n_features, 2, n_classes), n_classes=n_classes, 
                                       n_kernels=n_kernels, is_crf=spatial_model)
    else:
        model = FlexibleClassifier(model_constructor, n_classes=n_classes, 
                                   n_kernels=n_kernels, is_crf=spatial_model)

    optimizer = optimizer_constructor(model.parameters(), lr=lr)
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(training_data, model, loss_fn, optimizer, l1_lambda, l2_lambda, print_freq)
        test_loop(test_data, model, loss_fn)

    print("Done!")
    
    # save the model if requested 
    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    return model