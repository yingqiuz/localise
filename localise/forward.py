"""
This module provides flexible classifier models with optional CRF functionality
for brain structure localisation tasks.
"""

import torch
from torch.nn import Module, Parameter, Linear, ReLU
from torch.nn.functional import relu, softmax


class FlexibleClassifier(Module):
    """
    A flexible classifier with optional Conditional Random Field (CRF) post-processing.
    
    This classifier can operate in two modes:
    1. Standard mode: Direct classification without spatial smoothing
    2. CRF mode: Iterative refinement using spatial neighborhood information
    
    Args:
        layer: The base neural network layer for feature extraction
        n_classes: Number of output classes (default: 2)
        n_kernels: Number of spatial kernels for CRF smoothing (default: 1)
        is_crf: Enable CRF post-processing (default: False)
        n_iter: Number of CRF iterations (default: 3)
        init_weight: Initial value for CRF smoothing weights (default: 0.5)
    
    Example:
        >>> base_layer = MLP(input_dim=100, hidden_dim=50, output_dim=2)
        >>> classifier = FlexibleClassifier(base_layer, n_classes=2, is_crf=True)
        >>> output = classifier(data)
    """
    
    def __init__(self, layer, n_classes=2, n_kernels=1, is_crf=False, n_iter=3, init_weight=0.5):
        super().__init__()
        self.layer = layer
        self.is_crf = is_crf
        self.n_iter = n_iter
        self.n_classes = n_classes
        self.n_kernels = n_kernels

        if self.is_crf:
            # CRF parameters
            self.smooth_weight = Parameter(torch.full((n_kernels,), init_weight))
            # Compatibility matrix: encourages different labels for neighboring voxels
            compatibility_matrix = torch.ones(n_classes, n_classes) - torch.eye(n_classes)
            self.compatibility = Parameter(compatibility_matrix.float(), requires_grad=False)
    
    def forward(self, data):
        """
        Forward pass through the classifier.
        
        Args:
            data: Input data object with attributes:
                - X: Feature tensor of shape (batch_size, n_features)
                - f: List of spatial filter matrices (required only for CRF mode)
        
        Returns:
            Probability tensor of shape (batch_size, n_classes)
        """
        self._validate_input(data)
        return self._forward_crf(data) if self.is_crf else self._forward_standard(data)

    def _forward_standard(self, data):
        """Standard forward pass without CRF."""
        logits = self.layer(data.X)
        return softmax(logits, dim=1)

    def _forward_crf(self, data):
        """
        Forward pass with CRF post-processing using mean field approximation.
        
        The CRF iteratively refines predictions by incorporating spatial smoothness
        constraints through neighborhood information.
        """
        # Get initial unary potentials from base layer
        unary_potentials = self.layer(data.X)
        
        # Initialize mean field with unary potentials
        mean_field = unary_potentials.clone()
        
        # Iterative mean field updates
        for _ in range(self.n_iter):
            # Convert to probabilities
            probabilities = softmax(mean_field, dim=1)
            
            # Apply spatial smoothing using filter kernels
            smoothed = sum(
                kernel @ probabilities * weight 
                for kernel, weight in zip(data.f, self.smooth_weight)
            )
            
            # Update mean field: unary - compatibility * smoothed
            mean_field = unary_potentials - smoothed @ self.compatibility
        
        return softmax(mean_field, dim=1)
    
    def _validate_input(self, data):
        """Validate input data structure."""
        if not hasattr(data, 'X'):
            raise AttributeError("Input data must have attribute 'X' (feature tensor)")
        
        if self.is_crf and not hasattr(data, 'f') or data.f is None:
            raise AttributeError("CRF mode requires input data to have attribute 'f' (spatial filters)")
        
        if self.is_crf and len(data.f) != self.n_kernels:
            raise ValueError(
                f"Number of spatial filters ({len(data.f)}) must match n_kernels ({self.n_kernels})"
            )


class MLP(Module):
    """
    Multi-layer Perceptron with single hidden layer.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer  
        output_dim: Dimension of output layer
        activation: Activation function (default: ReLU)
    
    Example:
        >>> mlp = MLP(input_dim=784, hidden_dim=128, output_dim=10)
        >>> output = mlp(input_tensor)
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, activation=ReLU()):
        super().__init__()
        self.layer1 = Linear(input_dim, hidden_dim)
        self.activation = activation
        self.layer2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class Affine(Module):
    """
    Simple affine transformation (linear layer) for data objects.
    
    This is a wrapper around nn.Linear that accepts data objects
    with .X attribute instead of raw tensors.
    
    Args:
        n_features: Number of input features
        n_classes: Number of output classes
    """
    
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.layer = Linear(n_features, n_classes)

    def forward(self, data):
        """
        Forward pass through affine layer.
        
        Args:
            data: Input data object with .X attribute
            
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        if not hasattr(data, 'X'):
            raise AttributeError("Input data must have attribute 'X'")
        return self.layer(data.X)


class Perceptron(Module):
    """
    Single hidden layer perceptron for data objects.
    
    Similar to MLP but accepts data objects with .X attribute
    instead of raw tensors.
    
    Args:
        n_features: Number of input features
        n_hidden: Number of hidden units
        n_classes: Number of output classes
    """
    
    def __init__(self, n_features, n_hidden, n_classes):
        super().__init__()
        self.layer1 = Linear(n_features, n_hidden)
        self.layer2 = Linear(n_hidden, n_classes)

    def forward(self, data):
        """
        Forward pass through perceptron.
        
        Args:
            data: Input data object with .X attribute
            
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        if not hasattr(data, 'X'):
            raise AttributeError("Input data must have attribute 'X'")
        
        hidden = relu(self.layer1(data.X))
        output = self.layer2(hidden)
        return output


# class PerceptronCRF(Module):
#     def __init__(self, n_features, n_hidden, n_classes, n_kernels, n_iter=3):
#         super().__init__()
#         self.layer1 = Linear(n_features, n_hidden)
#         self.layer2 = Linear(n_hidden, n_classes)
#         self.w = Parameter(torch.randn(n_kernels))
#         self.μ = Parameter((torch.ones(n_classes, n_classes) - torch.eye(n_classes)).float(), requires_grad=False)
#         self.n_iter = n_iter

#     def forward(self, data):
#         y = self.layer2(relu(self.layer1(data.X)))
#         h = torch.zeros_like(y)
#         for _ in range(self.n_iter):
#             h = softmax(y, dim=1)
#             h2 = torch.zeros_like(h)
#             for k, f in enumerate(data.f):
#                 h2 += h * f * self.w[k]
#             h = y - self.μ @ h2
#         return h


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