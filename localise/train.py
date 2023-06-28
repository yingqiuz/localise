import torch
from torch import nn
from torch import optim
import time
from functools import partial

def l1_regularization(weight):
    return torch.norm(weight, 1)

def l2_regularization(weight):
    return torch.norm(weight, 2)

def train_loop(model, training_data, test_data, loss_func=nn.BCEWithLogitsLoss(), opt=None, λ1=1e-4, λ2=1e-4, n_epochs=20, timeout=20):
    opt = optim.Adam(model.parameters(), lr=0.01) if opt is None else opt
    start_time = time.time()

    def loss(x, y):
        return loss_func(model(x), y) + λ2 * l2_regularization(model.W) + λ1 * l1_regularization(model.W)

    def mean_loss():
        losses = []
        for (x, y) in test_data:
            loss_value = loss(x, y)
            losses.append(loss_value.item())
        return sum(losses) / len(losses)

    for epoch in range(n_epochs):
        for (x, y) in training_data:
            opt.zero_grad()
            loss_value = loss(x, y)
            loss_value.backward()
            opt.step()

        if (time.time() - start_time) > timeout:
            start_time = time.time()
            print('Mean Loss:', mean_loss())
    
    return model
