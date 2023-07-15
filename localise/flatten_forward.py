import torch
from torch.nn import Module, Parameter, Linear, ReLU
from torch.nn.functional import relu, softmax


class FlexibleClassifier(Module):
    """
    A flexible classifier model, which can be optionally run with Conditional Random Field (CRF) logic.
    
    Args:
        layer (nn.Module): The layer to be used for classification.
        n_classes (int, optional): The number of output classes. Defaults to 2.
        n_kernels (int, optional): The number of kernels to use. Defaults to 1.
        is_crf (bool, optional): If True, uses CRF for forward computation. Defaults to False.
        n_iter (int, optional): The number of iterations for CRF computation. Defaults to 3.
    """
    def __init__(self, layer, n_classes=2, n_kernels=1, is_crf=False, n_iter=3):
        super().__init__()
        self.layer = layer
        self.is_crf = is_crf
        self.n_iter = n_iter
        self.n_classes = n_classes
        self.n_kernels = n_kernels

        if self.is_crf:
            self.smooth_weight = Parameter(torch.randn(n_kernels))
            self.compatibility = Parameter((torch.ones(n_classes, n_classes) - torch.eye(n_classes)).float(), requires_grad=False)
    
    #@property
    #def parameters(self):
        #if self.is_crf:
            #return list(super().parameters()) + [self.w, self.μ]
        #else:
            #return super().parameters()
    def forward(self, data):
        """
        Forward propagation for the classifier.
        
        Args:
            data: The input data.
        
        Returns:
            The output from the forward propagation.
        """
        return self.forward_crf(data) if self.is_crf else self.forward_no_crf(data)

    def forward_no_crf(self, data):
        """
        Forward propagation without using CRF.
        
        Args:
            data: The input data.
        
        Returns:
            The output from the forward propagation.
        """
        y = self.layer(data.X)
        return softmax(y, dim=1)

    def forward_crf(self, data):
        """
        Forward propagation using CRF.
        
        Args:
            data: The input data.
        
        Returns:
            The output from the forward propagation.
        """
        y = self.layer(data.X) # negative unary potential
        h = y.clone()
        for _ in range(self.n_iter):
            h = softmax(h, dim=1)
            h = sum(f @ h * w for f, w in zip(data.f, self.smooth_weight)) # vectorised version
            h = y - h @ self.compatibility
        return softmax(h, dim=1)


class MLP(Module):
    """
    A simple Multi-layer Perceptron (MLP) model.
    
    Args:
        input_dim (int): The dimension of the input.
        hidden_dim (int): The dimension of the hidden layer.
        output_dim (int): The dimension of the output layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, output_dim)
        self.relu = ReLU()

    def forward(self, x):
        """
        Forward propagation for the MLP.
        
        Args:
            x: The input data.
        
        Returns:
            The output from the forward propagation.
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    

class FlexibleCRF(Module):
    def __init__(self, layer, n_classes=2, n_kernels=1, is_crf=False, n_iter=3):
        super().__init__()
        self.layer = layer
        self.is_crf = is_crf
        self.n_iter = n_iter

        if self.is_crf:
            self.smooth_weight = Parameter(torch.randn(n_kernels))
            self.compatibility = Parameter((torch.ones(n_classes, n_classes) - torch.eye(n_classes)).float(), requires_grad=False)

    def forward(self, data):
        y = self.layer(data.X) # negative unary potential
        if self.is_crf:
            h = y.clone()
            for _ in range(self.n_iter):
                h = softmax(h, dim=1)
                #h2 = torch.zeros_like(h)
                h = sum(f @ h * w for f, w in zip(data.f, self.smooth_weight)) # vectorised version
                #for k, f in enumerate(data.f):
                    #h2 += f @ h * self.smooth_weight[k]
                h = y - h @ self.compatibility
            return softmax(h, dim=1)
        return softmax(y, dim=1)
      

class Affine(Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.layer = Linear(n_features, n_classes)
        #self.W = Parameter(torch.randn(n_classes, n_features))
        #self.b = Parameter(torch.randn(n_classes, 1))

    def forward(self, data):
        y = self.layer(data.X)
        #y = self.W @ data.X + self.b
        return y

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
