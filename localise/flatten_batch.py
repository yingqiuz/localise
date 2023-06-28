import numpy as np
from itertools import product
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import torch


class Adjacency:
    def __init__(self, inds1, inds2, n):
        self.inds1 = inds1
        self.inds2 = inds2
        self.n = n

def get_adj_sparse(mask):
    # Get the indices where mask is not zero
    index = np.argwhere(mask != 0)
    n = len(index)

    # 26 neighbourhood system
    neighbour_offsets = np.array(list(product([-1, 0, 1], repeat=3)))
    neighbour_offsets = neighbour_offsets[~np.all(neighbour_offsets == 0, axis=1)]  # exclude the center point (0, 0, 0)

    inds = []
    for v in range(n):
        neighbours = index[v] + neighbour_offsets
        for neighbour in neighbours:
            m = np.all(index == neighbour, axis=1)
            inds += [(v, i) for i in np.where(m)[0]]
    
    inds1, inds2 = zip(*inds)
    return inds1, inds2, n

def get_adj_sparse_kdt(mask):
    """
    This function generates indices of neighbors in a 3D grid for all non-zero elements of the input mask using a k-d tree.

    It first identifies the indices of non-zero elements in the mask, then uses a k-d tree for efficient nearest neighbor search. 
    The function generates a 26-point neighborhood around each point (excluding the point itself), and checks whether each 
    of these neighbors is present in the k-d tree (i.e., whether it corresponds to a non-zero element of the mask).

    Parameters
    ----------
    mask : ndarray
        A 3D numpy array which acts as a mask. The function will only consider the points for which the corresponding 
        mask value is non-zero.

    Returns
    -------
    inds1 : tuple
        A tuple of indices representing the first coordinate of the neighbor pairs.

    inds2 : tuple
        A tuple of indices representing the second coordinate of the neighbor pairs.

    n : int
        The number of non-zero elements in the mask.

    Note
    ----
    The function assumes that the mask is a 3D array. If the mask has more or fewer dimensions, the function might 
    not work as expected.

    Also, the function does not check whether the mask contains only binary (0 or 1) values. If the mask contains 
    other values, these will be treated as "non-zero", which might lead to unexpected results.

    The function returns a sparse representation of the neighbor pairs, meaning that only the pairs for which both 
    elements are non-zero in the mask are returned. If a full (dense) representation is needed, additional processing 
    might be required.
    """
    # Get the indices where mask is not zero
    index = np.argwhere(mask != 0)
    n = len(index)

    # 26 neighbourhood system
    neighbour_offsets = np.array(list(product([-1, 0, 1], repeat=3)))
    neighbour_offsets = neighbour_offsets[~np.all(neighbour_offsets == 0, axis=1)]  # exclude the center point (0, 0, 0)

    inds = []
    tree = cKDTree(index)  # create k-d tree
    for v in range(n):
        neighbours = index[v] + neighbour_offsets
        for neighbour in neighbours:
            # query the k-d tree for the neighbour
            d, i = tree.query(neighbour, k=1, distance_upper_bound=1)
            if d != np.inf:  # if neighbour was found
                inds.append((v, i))
    
    inds1, inds2 = zip(*inds)
    return inds1, inds2, n

class FlattenedCRFBatch:
    """
    This class represents a batch for Conditional Random Fields (CRF) using numpy and scipy. It calculates the kernel of the input data.

    Attributes:
        K (int): Number of CRFs in the batch.
        X (numpy.array): Input data.
        adj (Adjacency): Adjacency information for the input data.
        n (int): Number of samples in the input data.
        d (int): Number of features in the input data.
        gamma (numpy.array): Gamma parameter for the Radial Basis Function (RBF) kernel.
        f (scipy.sparse.csr_matrix): Calculated kernel for the input data.

    Methods:
        construct_kernel(X, adj, gamma): Constructs the kernel for the input data.
    """

    def __init__(self, X, adj, K=2, gamma=None):
        """
        Initializes an instance of FlattenedCRFBatch.

        Args:
            X (numpy.array): Input data.
            adj (Adjacency): Adjacency information for the input data.
            K (int, optional): Number of CRFs in the batch. Defaults to 2.
            gamma (numpy.array, optional): Gamma parameter for the Radial Basis Function (RBF) kernel. Defaults to a numpy array with value 0.
        """
        self.K = K
        self.X = X
        self.adj = adj
        self.n = X.shape[1]
        self.d = X.shape[0]
        self.gamma = gamma if gamma is not None else np.array([0])
        if not isinstance(self.gamma, np.ndarray):
            self.gamma = np.array([self.gamma], dtype=X.dtype)
        self.f = self.construct_kernel(self.X, self.adj, self.gamma)

    def construct_kernel(self, X, adj, gamma):
        """
        Constructs the kernel for the input data.

        Args:
            X (numpy.array): Input data.
            adj (Adjacency): Adjacency information for the input data.
            gamma (numpy.array): Gamma parameter for the Radial Basis Function (RBF) kernel.

        Returns:
            scipy.sparse.csr_matrix: Kernel for the input data.
        """
        if gamma.ndim > 0:
            return [self.construct_kernel(X, adj, el) for el in gamma]
        else:
            if gamma == 0:
                return csr_matrix((np.ones(len(adj.inds1)), (adj.inds1, adj.inds2)), shape=(adj.n, adj.n))
            else:
                indices = [(i, j) for i, j in zip(adj.inds1, adj.inds2) if i < j]
                inds1, inds2 = zip(*indices)
                vals = [np.exp(-np.sum((X[:, i] - X[:, j]) ** 2) * gamma) for i, j in indices]
                f = csr_matrix((vals, (inds1, inds2)), shape=(adj.n, adj.n))
                return f + f.T

class FlattenedCRFBatchTensor:
    """
    This class creates a batch for Conditional Random Fields (CRF) and calculates its kernel for the given data.

    Attributes:
        K (int): Number of CRFs in the batch.
        X (torch.Tensor): Input data.
        adj (Adjacency): Adjacency information for the input data.
        n (int): Number of samples in the input data.
        d (int): Number of features in the input data.
        gamma (torch.Tensor): Gamma parameter for the Radial Basis Function (RBF) kernel.
        f (torch.sparse_coo_tensor): Calculated kernel for the input data.

    Methods:
        construct_kernel(X, adj, gamma): Constructs the kernel for the input data.
    """

    def __init__(self, X, adj, K=2, gamma=None):
        """
        Initializes a new instance of FlattenedCRFBatchTorch.

        Args:
            X (torch.Tensor): Input data.
            adj (Adjacency): Adjacency information for the input data.
            K (int, optional): Number of CRFs in the batch. Defaults to 2.
            gamma (torch.Tensor, optional): Gamma parameter for the Radial Basis Function (RBF) kernel. If not specified, it defaults to a tensor with value 0 on the same device and with the same datatype as X.
        """
        self.K = K
        self.X = X
        self.adj = adj
        self.n = X.shape[1]
        self.d = X.shape[0]
        self.gamma = gamma.to(X) if gamma is not None else torch.tensor([0], device=X.device, dtype=X.dtype)
        if not isinstance(self.gamma, torch.Tensor):
            self.gamma = torch.tensor([self.gamma], device=X.device, dtype=X.dtype)
        self.f = self.construct_kernel(self.X, self.adj, self.gamma)

    def construct_kernel(self, X, adj, gamma):
        """
        Constructs the kernel for the input data.

        Args:
            X (torch.Tensor): Input data.
            adj (Adjacency): Adjacency information for the input data.
            gamma (torch.Tensor): Gamma parameter for the Radial Basis Function (RBF) kernel.

        Returns:
            torch.sparse_coo_tensor: Kernel for the input data.
        """
        if gamma.dim() > 0:
            return torch.stack([self.construct_kernel(X, adj, el.item()) for el in gamma])
        else:
            indices = torch.tensor([adj.inds1, adj.inds2], device=X.device)
            if gamma == 0:
                values = torch.ones(len(adj.inds1), device=X.device, dtype=X.dtype)
                size = adj.n, adj.n
                return torch.sparse_coo_tensor(indices, values, size)
            else:
                indices_mask = adj.inds1 < adj.inds2
                inds1 = torch.tensor(adj.inds1[indices_mask], device=X.device)
                inds2 = torch.tensor(adj.inds2[indices_mask], device=X.device)
                vals = torch.exp(-torch.sum((X[:, inds1] - X[:, inds2]) ** 2, dim=0) * gamma)
                f = torch.sparse_coo_tensor(torch.stack([inds1, inds2]), vals, (adj.n, adj.n))
                return f + f.coalesce().t().coalesce()
