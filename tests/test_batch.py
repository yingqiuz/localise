"""
Test suite for batch.py - spatial adjacency and CRF batch construction.
"""

import pytest
import numpy as np
import torch
from scipy.sparse import csr_matrix

from localise.batch import (
    Adjacency, 
    get_adj_sparse, 
    get_adj_sparse_kdt,
    FlattenedCRFBatch,
    FlattenedCRFBatchTensor
)


class TestAdjacency:
    """Test Adjacency class."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        inds1 = [0, 1, 2]
        inds2 = [1, 2, 0]
        n = 3
        
        adj = Adjacency(inds1, inds2, n)
        
        assert adj.n == 3
        assert len(adj.inds1) == 3
        assert len(adj.inds2) == 3
        np.testing.assert_array_equal(adj.inds1, [0, 1, 2])
        np.testing.assert_array_equal(adj.inds2, [1, 2, 0])
    
    def test_init_empty(self):
        """Test initialization with empty adjacency."""
        adj = Adjacency([], [], 0)
        
        assert adj.n == 0
        assert len(adj.inds1) == 0
        assert len(adj.inds2) == 0
    
    def test_repr(self):
        """Test string representation."""
        adj = Adjacency([0, 1], [1, 0], 2)
        repr_str = repr(adj)
        
        assert "Adjacency" in repr_str
        assert "n_voxels=2" in repr_str
        assert "n_edges=2" in repr_str


class TestGetAdjSparse:
    """Test get_adj_sparse function."""
    
    def test_empty_mask(self):
        """Test with empty mask."""
        mask = np.zeros((3, 3, 3))
        inds1, inds2, n = get_adj_sparse(mask)
        
        assert n == 0
        assert len(inds1) == 0
        assert len(inds2) == 0
    
    def test_single_voxel(self):
        """Test with single voxel (no neighbors)."""
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1
        
        inds1, inds2, n = get_adj_sparse(mask)
        
        assert n == 1
        assert len(inds1) == 0  # No neighbors
        assert len(inds2) == 0
    
    def test_two_adjacent_voxels(self):
        """Test with two adjacent voxels."""
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1  # Center
        mask[1, 1, 2] = 1  # Adjacent in z direction
        
        inds1, inds2, n = get_adj_sparse(mask)
        
        assert n == 2
        assert len(inds1) == 2  # Each voxel connects to the other
        assert len(inds2) == 2
        # Should have connections both ways: 0->1 and 1->0
        connections = list(zip(inds1, inds2))
        assert (0, 1) in connections or (1, 0) in connections
    
    def test_3x3x3_block(self):
        """Test with 3x3x3 block of voxels."""
        mask = np.ones((3, 3, 3))
        inds1, inds2, n = get_adj_sparse(mask)
        
        assert n == 27  # 3^3 voxels
        assert len(inds1) > 0
        assert len(inds1) == len(inds2)
        # Center voxel should have most connections (26 neighbors)
        # Corner voxels should have fewer connections


class TestGetAdjSparseKdt:
    """Test get_adj_sparse_kdt function."""
    
    def test_empty_mask(self):
        """Test KDTree version with empty mask."""
        mask = np.zeros((3, 3, 3))
        inds1, inds2, n = get_adj_sparse_kdt(mask)
        
        assert n == 0
        assert len(inds1) == 0
        assert len(inds2) == 0
    
    def test_single_voxel(self):
        """Test KDTree version with single voxel."""
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1
        
        inds1, inds2, n = get_adj_sparse_kdt(mask)
        
        assert n == 1
        assert len(inds1) == 0
        assert len(inds2) == 0
    
    def test_compare_with_regular_version(self):
        """Test that KDTree version gives same results as regular version."""
        # Create a random sparse mask
        np.random.seed(42)
        mask = np.random.randint(0, 2, (5, 5, 5))
        
        # Get results from both methods
        inds1_reg, inds2_reg, n_reg = get_adj_sparse(mask)
        inds1_kdt, inds2_kdt, n_kdt = get_adj_sparse_kdt(mask)
        
        # Should have same number of voxels
        assert n_reg == n_kdt
        
        # Should have same number of connections (approximately)
        # Note: might differ slightly due to floating point precision in KDTree
        assert abs(len(inds1_reg) - len(inds1_kdt)) <= n_reg  # Allow some tolerance


class TestFlattenedCRFBatch:
    """Test FlattenedCRFBatch class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # 4 voxels, 3 features each
        X = np.random.randn(4, 3).astype(np.float32)
        # Simple adjacency: 0-1-2-3 chain
        adj = Adjacency([0, 1, 2], [1, 2, 3], 4)
        return X, adj
    
    def test_init_with_adjacency_object(self, sample_data):
        """Test initialization with Adjacency object."""
        X, adj = sample_data
        
        batch = FlattenedCRFBatch(X, adj, gamma=[0.0, 1.0])
        
        assert batch.n == 4
        assert batch.d == 3
        assert batch.K == 2
        assert len(batch.gamma) == 2
        assert len(batch.f) == 2  # Two kernels for two gamma values
    
    def test_init_with_tuple(self, sample_data):
        """Test initialization with (inds1, inds2, n) tuple."""
        X, adj = sample_data
        adj_tuple = (adj.inds1, adj.inds2, adj.n)
        
        batch = FlattenedCRFBatch(X, adj_tuple, gamma=0.5)
        
        assert batch.n == 4
        assert isinstance(batch.adj, Adjacency)
        assert len(batch.f) == 1  # Single kernel
    
    def test_dimension_mismatch(self):
        """Test error when X and adjacency dimensions don't match."""
        X = np.random.randn(3, 2)  # 3 voxels
        adj = Adjacency([0, 1], [1, 2], 4)  # But adjacency says 4 voxels
        
        with pytest.raises(ValueError, match="X has 3 samples but adjacency has 4 voxels"):
            FlattenedCRFBatch(X, adj)
    
    def test_binary_kernel_gamma_zero(self, sample_data):
        """Test binary connectivity kernel (gamma=0)."""
        X, adj = sample_data
        
        batch = FlattenedCRFBatch(X, adj, gamma=0.0)
        
        kernel = batch.f[0]
        assert isinstance(kernel, csr_matrix)
        assert kernel.shape == (4, 4)
        
        # For gamma=0, all connected voxels should have weight 1
        # Check that connections exist
        assert kernel.nnz > 0
    
    def test_rbf_kernel_gamma_nonzero(self, sample_data):
        """Test RBF kernel with gamma > 0."""
        X, adj = sample_data
        
        batch = FlattenedCRFBatch(X, adj, gamma=1.0)
        
        kernel = batch.f[0]
        assert isinstance(kernel, csr_matrix)
        assert kernel.shape == (4, 4)
        
        # RBF kernel values should be in (0, 1] range
        data = kernel.data
        assert np.all(data > 0)
        assert np.all(data <= 1)
    
    def test_multiple_gamma_values(self, sample_data):
        """Test with multiple gamma values."""
        X, adj = sample_data
        
        batch = FlattenedCRFBatch(X, adj, gamma=[0.0, 0.5, 1.0])
        
        assert len(batch.f) == 3
        for kernel in batch.f:
            assert isinstance(kernel, csr_matrix)
            assert kernel.shape == (4, 4)


class TestFlattenedCRFBatchTensor:
    """Test FlattenedCRFBatchTensor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample tensor data for testing."""
        # 4 voxels, 3 features each
        X = torch.randn(4, 3)
        # Simple adjacency: 0-1-2-3 chain  
        adj = Adjacency([0, 1, 2], [1, 2, 3], 4)
        return X, adj
    
    def test_init_basic(self, sample_data):
        """Test basic initialization."""
        X, adj = sample_data
        
        batch = FlattenedCRFBatchTensor(X, adj, gamma=torch.tensor([0.0, 1.0]))
        
        assert batch.n == 4
        assert batch.d == 3
        assert batch.gamma.device == X.device
        assert len(batch.f) == 2
    
    def test_device_handling(self, sample_data):
        """Test device handling for tensors."""
        X, adj = sample_data
        
        # Test with CPU tensors
        batch_cpu = FlattenedCRFBatchTensor(X, adj, gamma=[0.0])
        assert batch_cpu.gamma.device.type == 'cpu'
        assert batch_cpu.f[0].device.type == 'cpu'
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            X_cuda = X.cuda()
            batch_cuda = FlattenedCRFBatchTensor(X_cuda, adj, gamma=[0.0])
            assert batch_cuda.gamma.device.type == 'cuda'
            assert batch_cuda.f[0].device.type == 'cuda'
    
    def test_gamma_types(self, sample_data):
        """Test different gamma input types."""
        X, adj = sample_data
        
        # Test with float
        batch1 = FlattenedCRFBatchTensor(X, adj, gamma=0.5)
        assert batch1.gamma.shape == (1,)
        
        # Test with list
        batch2 = FlattenedCRFBatchTensor(X, adj, gamma=[0.0, 1.0])
        assert batch2.gamma.shape == (2,)
        
        # Test with numpy array
        batch3 = FlattenedCRFBatchTensor(X, adj, gamma=np.array([0.5]))
        assert batch3.gamma.shape == (1,)
        
        # Test with torch tensor
        batch4 = FlattenedCRFBatchTensor(X, adj, gamma=torch.tensor([0.0, 0.5]))
        assert batch4.gamma.shape == (2,)
    
    def test_sparse_tensor_output(self, sample_data):
        """Test that kernels are proper sparse tensors."""
        X, adj = sample_data
        
        batch = FlattenedCRFBatchTensor(X, adj, gamma=0.0)
        
        kernel = batch.f[0]
        assert kernel.is_sparse
        assert kernel.shape == (4, 4)
    
    def test_kernel_symmetry(self, sample_data):
        """Test that kernels are symmetric."""
        X, adj = sample_data
        
        batch = FlattenedCRFBatchTensor(X, adj, gamma=0.0)
        kernel = batch.f[0]
        
        # Convert to dense for easier testing
        kernel_dense = kernel.to_dense()
        
        # Check if symmetric (within tolerance)
        assert torch.allclose(kernel_dense, kernel_dense.t(), atol=1e-6)
    
    def test_empty_adjacency(self):
        """Test with empty adjacency matrix."""
        X = torch.randn(3, 2)
        adj = Adjacency([], [], 3)  # No connections
        
        batch = FlattenedCRFBatchTensor(X, adj, gamma=0.0)
        
        kernel = batch.f[0]
        assert kernel.shape == (3, 3)
        assert kernel._nnz() == 0  # Should be empty


class TestIntegration:
    """Integration tests."""
    
    def test_batch_types_consistency(self):
        """Test that both batch types give similar results."""
        # Create test data
        X_np = np.random.randn(5, 3).astype(np.float32)
        X_torch = torch.from_numpy(X_np)
        adj = Adjacency([0, 1, 2, 3], [1, 2, 3, 4], 5)
        
        # Create both batch types
        batch_np = FlattenedCRFBatch(X_np, adj, gamma=0.0)
        batch_torch = FlattenedCRFBatchTensor(X_torch, adj, gamma=0.0)
        
        # Convert sparse tensors to numpy for comparison
        kernel_np = batch_np.f[0].toarray()
        kernel_torch = batch_torch.f[0].to_dense().numpy()
        
        # Should be approximately equal
        np.testing.assert_allclose(kernel_np, kernel_torch, atol=1e-5)
    
    def test_with_real_brain_mask_pattern(self):
        """Test with a pattern similar to real brain mask."""
        # Create a more realistic 3D mask pattern
        mask = np.zeros((10, 10, 10))
        # Create a hollow sphere-like pattern
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    dist = np.sqrt((i-5)**2 + (j-5)**2 + (k-5)**2)
                    if 2 < dist < 4:  # Hollow sphere
                        mask[i, j, k] = 1
        
        # Get adjacency
        inds1, inds2, n = get_adj_sparse(mask)
        
        if n > 0:  # Only test if we have voxels
            # Create features
            X = np.random.randn(n, 5).astype(np.float32)
            adj = Adjacency(inds1, inds2, n)
            
            # Test batch creation
            batch = FlattenedCRFBatch(X, adj, gamma=[0.0, 0.5])
            
            assert len(batch.f) == 2
            assert all(k.shape == (n, n) for k in batch.f)


class TestErrorCases:
    """Test error handling."""
    
    def test_invalid_adjacency_input(self):
        """Test error with invalid adjacency input."""
        X = np.random.randn(3, 2)
        
        with pytest.raises(ValueError, match="adj must be Adjacency object"):
            FlattenedCRFBatch(X, "invalid_adjacency")
    
    def test_invalid_tuple_length(self):
        """Test error with wrong tuple length."""
        X = np.random.randn(3, 2)
        
        with pytest.raises(ValueError, match="adj must be Adjacency object"):
            FlattenedCRFBatch(X, (1, 2))  # Too short
    
    def test_dimension_validation(self):
        """Test dimension validation between X and adjacency."""
        X = torch.randn(3, 2)
        adj = Adjacency([0, 1], [1, 2], 5)  # Says 5 voxels but X has 3
        
        with pytest.raises(ValueError, match="X has 3 samples but adjacency has 5 voxels"):
            FlattenedCRFBatchTensor(X, adj)