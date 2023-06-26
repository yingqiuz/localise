import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRF(nn.Module):
    def __init__(self, n_spatial_dims, filter_size=3, n_iter=5, requires_grad=True,
                 returns='logits', smoothness_weight=1, smoothness_theta=1):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_iter = n_iter
        self.filter_size = np.broadcast_to(filter_size, n_spatial_dims)
        self.returns = returns
        self.requires_grad = requires_grad
        self._set_param('smoothness_weight', smoothness_weight)
        
    def _set_param(self, name, init_value):
        setattr(self, name, nn.Parameter(torch.tensor(init_value, dtype=torch.float, requires_grad=self.requires_grad)))

    def forward(self, x, spatial_spacings=None):
        """
        Propagates the input through the model.

        This function is the forward pass of the model. It takes the input tensor, applies the 
        Conditional Random Field (CRF) process including initialisation, message passing, 
        penalizing incompatibility, and adding unary potentials. It finally returns the output 
        based on the value of the attribute `returns`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_classes, *spatial)``.
        spatial_spacings : torch.Tensor, optional
            Tensor indicating the spacing between each dimension in the spatial dimensions. 
            If not provided, it is assumed to be a tensor of ones. 

        Returns
        -------
        output : torch.Tensor
            Output tensor after the CRF process. Its type depends on the value of the attribute `returns`. 
            If `returns` is 'logits', it returns the logit values. 
            If `returns` is 'proba', it returns probabilities after applying the softmax function. 
            If `returns` is 'log-proba', it returns log probabilities.

        Raises
        ------
        ValueError
            If the spatial dimensions of the input data and model do not match or
            if the attribute `returns` is not one of 'logits', 'proba', or 'log-proba'.
        """
        batch_size, n_classes, *spatial = x.shape
        if len(spatial) != self.n_spatial_dims:
            raise ValueError("Spatial dimensions of the data and model do not match.")
        
        if n_classes == 1:
            x = torch.cat([x, torch.zeros(x.shape).to(x)], dim=1)
        
        if spatial_spacings is None:
            spatial_spacings = torch.ones((batch_size, self.n_spatial_dims))
            
        negative_unary = x.clone()
        
        for i in range(self.n_iter):
            # initialisation
            x = F.softmax(x, dim=1)
            
            # message passing
            x = self.smoothness_weight * self._smoothing_filter(x, spatial_spacings)
            
            # penalise incompatibility
            x = self._compatibility_transform(x)
            
            # add back unary potentials
            x = negative_unary - x
            
        if self.returns == 'logits':
            output = x
        elif self.returns == 'proba':
            output = F.softmax(x, dim=1)
        elif self.returns == 'log-proba':
            output = F.log_softmax(x, dim=1)
        else:
            raise ValueError("Attribute ``returns`` must be 'logits', 'proba' or 'log-proba'.")
        
        if n_classes == 1:
            output = output[:, 0] - output[:, 1] if self.returns == 'logits' else output[:, 0]
            output.unsqueeze_(1)
        
        return output
    
    def _smoothing_filter(self, x, spatial_spacings):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. logits.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``.
        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        return torch.stack([self._single_smoothing_filter(x[i], spatial_spacings[i]) for i in range(x.shape[0])])
    
    @staticmethod
    def _pad(x, filter_size):
        """
        Pads the input tensor along all dimensions using the given filter sizes.

        This function calculates the padding for each dimension of the tensor
        as half of the corresponding filter size (rounded down, if necessary).
        It then applies symmetric padding to the tensor along all dimensions.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be padded. Can have any number of dimensions.
        filter_size : list of int
            The filter sizes for each dimension of the tensor. The length of this
            list should match the number of dimensions in `x`.

        Returns
        -------
        torch.Tensor
            The input tensor `x`, padded along all dimensions according to the 
            provided filter sizes.

        """        
        padding = []
        for fs in filter_size:
            padding += 2 * [fs // 2]
        
        return F.pad(x, list(reversed(padding)))
    
    def _single_smooth_filter(self, x):
        """
        Applies a single smoothing filter on the input tensor along each dimension.

        This function pads the input tensor, performs a 1-D convolution along each
        dimension using a filter of ones, and then reshapes the tensor to its original 
        shape. This operation can be interpreted as applying a smoothing filter to the
        tensor along each dimension independently.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(n, *spatial)``, where ``n`` is the number of 
            instances and ``*spatial`` is the spatial dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as `x`, i.e., ``(n, *spatial)``, after
            applying the smoothing filter along each dimension.

        Notes
        -----
        The operation is performed independently along each spatial dimension 
        of the tensor, not in a multivariate sense.
        """
        x = self._pad(x, self.filter_size)
        for dim in range(1, x.ndim):
            x = x.transpose(dim, -1)
            shape_before = x.shape[:-1]
            x = x.flatten(0, -2).unsqueeze(1)
            
            # equal weights in the filter
            kernel = torch.ones(self.filter_size, ).view(1, 1, -1).to(x)
            x = F.conv1d(x, kernel)
            
            # reshape
            x = x.squeeze(1).view(*shape_before, x.shape[-1]).transpose(-1, dim)
            
        return x
    
    def _compatibility_transform(self, x):
        """
        Applies a compatibility transform on the input tensor.

        This function creates a compatibility matrix by comparing every pair of labels 
        in the input tensor. Then it performs an Einstein summation between the input tensor
        and the compatibility matrix to generate the transformed tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_classes, *spatial)``.

        Returns
        -------
        output : torch.Tensor
            Transformed tensor of shape ``(batch_size, n_classes, *spatial)``.

        Notes
        -----
        This function is generally used in the context of Conditional Random Field (CRF) models 
        to model the compatibility (or incompatibility) between different labels.
        """
        labels = torch.arange(x.shape[1])
        compatibility_matrix = self._compatibility_function(labels, labels.unsqueeze(1)).to(x)
        # dot product across the n_class dimension
        return torch.einsum('ij..., jk -> ik...', x, compatibility_matrix)

    @staticmethod
    def _compatibility_function(label1, label2):
        """
        Computes the compatibility between two input tensors.

        This function returns a tensor with the same shape as the inputs containing
        the compatibility score between each pair of corresponding labels in the input tensors.
        The compatibility is computed as the negative of the equality comparison 
        between the input labels, cast to float.

        Parameters
        ----------
        label1 : torch.Tensor
            Input tensor containing label values.
        label2 : torch.Tensor
            Input tensor containing label values. Must be broadcastable to `label1`.

        Returns
        -------
        compatibility : torch.Tensor
            Tensor containing compatibility scores, with the same shape as the inputs.

        Notes
        -----
        The input tensors must be broadcastable to each other.
        """
        return -(label1 == label2).float()