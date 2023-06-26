import torch
from torch.nn import Module, Parameter, Linear
from torch.nn.functional import relu, softmax

class Affine(Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.layer = Linear(n_features, n_classes)

    def forward(self, data):
        y = self.layer(data.X)
        return y

class AffineCRF(Module):
    def __init__(self, n_features, n_classes, n_kernels, n_iter=3):
        super().__init__()
        self.layer = Linear(n_features, n_classes)
        self.w = Parameter(torch.randn(n_kernels))
        self.μ = Parameter((torch.ones(n_classes, n_classes) - torch.eye(n_classes)).float(), requires_grad=False)
        self.n_iter = n_iter

    def forward(self, data):
        y = self.layer(data.X)
        h = torch.zeros_like(y)
        for _ in range(self.n_iter):  # three iterations are sufficient for a single CRF step
            h = softmax(y, dim=1)
            h2 = torch.zeros_like(h)
            for k, f in enumerate(data.f):
                h2 += h * f * self.w[k]
            h = y - self.μ @ h2
        return h

class Perceptron(Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super().__init__()
        self.layer1 = Linear(n_features, n_hidden)
        self.layer2 = Linear(n_hidden, n_classes)

    def forward(self, data):
        y = self.layer2(relu(self.layer1(data.X)))
        return y

class PerceptronCRF(Module):
    def __init__(self, n_features, n_hidden, n_classes, n_kernels, n_iter=3):
        super().__init__()
        self.layer1 = Linear(n_features, n_hidden)
        self.layer2 = Linear(n_hidden, n_classes)
        self.w = Parameter(torch.randn(n_kernels))
        self.μ = Parameter((torch.ones(n_classes, n_classes) - torch.eye(n_classes)).float(), requires_grad=False)
        self.n_iter = n_iter

    def forward(self, data):
        y = self.layer2(relu(self.layer1(data.X)))
        h = torch.zeros_like(y)
        for _ in range(self.n_iter):
            h = softmax(y, dim=1)
            h2 = torch.zeros_like(h)
            for k, f in enumerate(data.f):
                h2 += h * f * self.w[k]
            h = y - self.μ @ h2
        return h
