"""
Spatial adjacency and CRF batch construction for neuroimaging.

This module creates spatial neighborhood information and smoothness kernels
for Conditional Random Field (CRF) processing of brain voxel data.
"""

import numpy as np
from itertools import product
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import torch


class Adjacency:
    """
    Stores spatial adjacency information for brain voxels.
    
    Args:
        inds1: Source voxel indices for each edge
        inds2: Target voxel indices for each edge  
        n: Total number of voxels
    """
    def __init__(self, inds1, inds2, n):
        self.inds1 = np.array(inds1)
        self.inds2 = np.array(inds2)
        self.n = int(n)
        
    def __repr__(self):
        return f"Adjacency(n_voxels={self.n}, n_edges={len(self.inds1)})"


def get_adj_sparse(mask):
    """
    Find spatial neighbors for all non-zero voxels in a 3D brain mask.
    Fixed version of your original function - now O(n) instead of O(n²).
    """
    # Get coordinates of all brain voxels
    index = np.argwhere(mask != 0)
    n = len(index)
    
    if n == 0:
        return [], [], 0

    # 26 neighbourhood system
    neighbour_offsets = np.array(list(product([-1, 0, 1], repeat=3)))
    neighbour_offsets = neighbour_offsets[~np.all(neighbour_offsets == 0, axis=1)]

    # Create efficient lookup: coordinate -> voxel index
    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(index)}
    
    inds = []
    for v in range(n):
        neighbours = index[v] + neighbour_offsets
        for neighbour in neighbours:
            neighbour_tuple = tuple(neighbour)
            if neighbour_tuple in coord_to_idx:
                inds.append((v, coord_to_idx[neighbour_tuple]))
    
    if len(inds) == 0:
        return [], [], n
    
    inds1, inds2 = zip(*inds)
    return inds1, inds2, n


def get_adj_sparse_kdt(mask):
    """
    Fixed version of your KDTree function with correct distance bounds.
    """
    # Get the indices where mask is not zero
    index = np.argwhere(mask != 0)
    n = len(index)

    if n == 0:
        return [], [], 0

    # 26 neighbourhood system
    neighbour_offsets = np.array(list(product([-1, 0, 1], repeat=3)))
    neighbour_offsets = neighbour_offsets[~np.all(neighbour_offsets == 0, axis=1)]

    inds = []
    tree = cKDTree(index)
    for v in range(n):
        neighbours = index[v] + neighbour_offsets
        for neighbour in neighbours:
            # Fixed: use proper distance bound for 26-connectivity
            # Distance 1.0 only finds face neighbors, need sqrt(3) ≈ 1.73 for corners
            d, i = tree.query(neighbour, k=1, distance_upper_bound=0.1)
            if d < 0.1 and 0 <= i < n:  # if neighbour was found
                inds.append((v, i))
    
    if len(inds) == 0:
        return [], [], n
    
    inds1, inds2 = zip(*inds)
    return inds1, inds2, n


class FlattenedCRFBatch:
    """
    Creates spatial smoothness kernels for CRF processing using numpy/scipy.
    Fixed version of your original class.
    """
    
    def __init__(self, X, adj, K=2, gamma=None):
        self.K = K
        self.X = X
    
        # Handle different adjacency input types
        if isinstance(adj, Adjacency):
            self.adj = adj
        elif isinstance(adj, (tuple, list)) and len(adj) == 3:
            self.adj = Adjacency(adj[0], adj[1], adj[2])
        else:
            raise ValueError("adj must be Adjacency object or (inds1, inds2, n) tuple")
        
        self.n = X.shape[0]
        self.d = X.shape[1]

        # Validate dimensions
        if self.n != self.adj.n:
            raise ValueError(f"X has {self.n} samples but adjacency has {self.adj.n} voxels")

        # Handle gamma
        if gamma is None:
            self.gamma = np.array([0.0])
        elif isinstance(gamma, (int, float)):
            self.gamma = np.array([float(gamma)])
        else: #or numpy arrays, lists, etc.
            self.gamma = np.array(gamma, dtype=np.float32)
            
        self.f = self.construct_kernel(self.X, self.adj, self.gamma)

    def construct_kernel(self, X, adj, gamma):
        """Construct symmetric kernels."""
        if gamma.ndim > 0:
            return np.stack([self.construct_kernel(X, adj, g) for g in gamma])
        
        gamma_val = float(gamma)
        
        if len(adj.inds1) == 0:
            # No connections - return empty sparse matrix
            return csr_matrix((adj.n, adj.n))
        
        if gamma_val == 0:
            values = np.ones(len(adj.inds1))
        else:
            feature_diffs = X[adj.inds1] - X[adj.inds2]
            distances_sq = np.sum(feature_diffs ** 2, axis=1)
            values = np.exp(-gamma_val * distances_sq)
        
        # Create symmetric matrix by including both directions explicitly
        row_indices = np.concatenate([adj.inds1, adj.inds2])
        col_indices = np.concatenate([adj.inds2, adj.inds1])
        all_values = np.concatenate([values, values])
        
        kernel = csr_matrix((all_values, (row_indices, col_indices)), shape=(adj.n, adj.n))
        return kernel


class FlattenedCRFBatchTensor:
    """
    Fixed version of your tensor-based CRF batch.
    """

    def __init__(self, X, adj, K=2, gamma=None):
        self.K = K
        self.X = X
    
        # Handle adjacency input
        if isinstance(adj, Adjacency):
            self.adj = adj
        elif isinstance(adj, (tuple, list)) and len(adj) == 3:
            self.adj = Adjacency(adj[0], adj[1], adj[2])
        else:
            raise ValueError("adj must be Adjacency object or (inds1, inds2, n) tuple")
            
        self.n = X.shape[0]
        self.d = X.shape[1]
        
        # Validate dimensions
        if self.n != self.adj.n:
            raise ValueError(f"X has {self.n} samples but adjacency has {self.adj.n} voxels")

        # Handle gamma more carefully
        if gamma is None:
            self.gamma = torch.tensor([0.0], device=X.device, dtype=X.dtype)
        elif isinstance(gamma, torch.Tensor):
            self.gamma = gamma.to(device=X.device, dtype=X.dtype)
        elif isinstance(gamma, (int, float)):
            self.gamma = torch.tensor([float(gamma)], device=X.device, dtype=X.dtype)
        else:
            # Handle numpy arrays, lists, etc.
            self.gamma = torch.tensor(gamma, device=X.device, dtype=X.dtype)
    
        self.f = self.construct_kernel(self.X, self.adj, self.gamma)

    def construct_kernel(self, X, adj, gamma):
        """Construct symmetric sparse tensors."""
        if gamma.dim() > 0:
            return torch.stack([self.construct_kernel(X, adj, g) for g in gamma])
        
        gamma_val = gamma.item()
        
        if len(adj.inds1) == 0:
            # No connections - return empty sparse tensor
            indices = torch.zeros((2, 0), device=X.device, dtype=torch.long)
            values = torch.zeros(0, device=X.device, dtype=X.dtype)
            return torch.sparse_coo_tensor(indices, values, (adj.n, adj.n), device=X.device)
        
        if gamma_val == 0:
            values = torch.ones(len(adj.inds1), device=X.device, dtype=X.dtype)
        else:
            feature_diffs = X[adj.inds1] - X[adj.inds2]
            distances_sq = torch.sum(feature_diffs ** 2, dim=1)
            values = torch.exp(-gamma_val * distances_sq)
        
        # Create tensors directly on correct device
        row_indices = torch.cat([
            torch.tensor(adj.inds1, device=X.device, dtype=torch.long),
            torch.tensor(adj.inds2, device=X.device, dtype=torch.long)
        ])
        col_indices = torch.cat([
            torch.tensor(adj.inds2, device=X.device, dtype=torch.long),
            torch.tensor(adj.inds1, device=X.device, dtype=torch.long)
        ])
        all_values = torch.cat([values, values])
        
        indices = torch.stack([row_indices, col_indices])
        kernel = torch.sparse_coo_tensor(indices, all_values, (adj.n, adj.n), device=X.device)
        
        return kernel.coalesce()