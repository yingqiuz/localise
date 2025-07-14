import pytest
import torch
from torch.nn import Linear, Tanh
from unittest.mock import Mock

from localise.forward import FlexibleClassifier, MLP, Affine, Perceptron


class MockData:
    """Mock data class for testing."""
    def __init__(self, X, f=None):
        self.X = X
        self.f = f


class TestFlexibleClassifier:
    """Test FlexibleClassifier with and without CRF."""
    
    @pytest.fixture
    def base_layer(self):
        """Create a simple base layer for testing."""
        return Linear(10, 3)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = torch.randn(5, 10)  # batch_size=5, features=10
        f = [torch.randn(5, 5), torch.randn(5, 5)]  # 2 spatial filters
        return MockData(X, f)
    
    def test_init_without_crf(self, base_layer):
        """Test initialization without CRF."""
        classifier = FlexibleClassifier(base_layer, n_classes=3, is_crf=False)
        
        assert classifier.layer == base_layer
        assert classifier.n_classes == 3
        assert classifier.is_crf is False
        assert not hasattr(classifier, 'smooth_weight')
        assert not hasattr(classifier, 'compatibility')
    
    def test_init_with_crf(self, base_layer):
        """Test initialization with CRF."""
        classifier = FlexibleClassifier(
            base_layer, n_classes=3, n_kernels=2, is_crf=True, 
            n_iter=5, init_weight=0.7
        )
        
        assert classifier.is_crf is True
        assert classifier.n_iter == 5
        assert classifier.n_kernels == 2
        assert classifier.smooth_weight.shape == (2,)
        assert torch.all(classifier.smooth_weight == 0.7)
        assert classifier.compatibility.shape == (3, 3)
        # Check compatibility matrix (ones - identity)
        expected = torch.ones(3, 3) - torch.eye(3)
        assert torch.allclose(classifier.compatibility, expected)
    
    def test_forward_without_crf(self, base_layer, sample_data):
        """Test forward pass without CRF."""
        classifier = FlexibleClassifier(base_layer, n_classes=3, is_crf=False)
        
        output = classifier(sample_data)
        
        assert output.shape == (5, 3)  # batch_size=5, n_classes=3
        # Check if output is properly normalized (probabilities sum to 1)
        assert torch.allclose(output.sum(dim=1), torch.ones(5), atol=1e-6)
        # Check if all values are between 0 and 1
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_forward_with_crf(self, base_layer, sample_data):
        """Test forward pass with CRF."""
        classifier = FlexibleClassifier(
            base_layer, n_classes=3, n_kernels=2, is_crf=True, n_iter=2
        )
        
        output = classifier(sample_data)
        
        assert output.shape == (5, 3)
        # Check if output is properly normalized
        assert torch.allclose(output.sum(dim=1), torch.ones(5), atol=1e-6)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_validation_missing_f_in_crf_mode(self, base_layer):
        """Test validation when data.f is missing in CRF mode."""
        classifier = FlexibleClassifier(base_layer, is_crf=True)
        bad_data = MockData(torch.randn(5, 10))  # No f attribute
        
        with pytest.raises(AttributeError, match="CRF mode requires input data to have attribute 'f'"):
            classifier(bad_data)
    
    def test_validation_wrong_number_of_kernels(self, base_layer):
        """Test validation when number of filters doesn't match n_kernels."""
        classifier = FlexibleClassifier(base_layer, n_kernels=3, is_crf=True)
        # Create data with only 2 filters instead of 3
        X = torch.randn(5, 10)
        f = [torch.randn(5, 5), torch.randn(5, 5)]  # Only 2 filters
        bad_data = MockData(X, f)
        
        with pytest.raises(ValueError, match="Number of spatial filters \\(2\\) must match n_kernels \\(3\\)"):
            classifier(bad_data)
    
    def test_different_init_weights(self, base_layer):
        """Test different initialization weights."""
        classifier = FlexibleClassifier(base_layer, n_kernels=3, is_crf=True, init_weight=0.9)
        
        assert torch.all(classifier.smooth_weight == 0.9)
    
    def test_single_iteration_crf(self, base_layer, sample_data):
        """Test CRF with single iteration."""
        base_layer = Linear(5, 2)  # 2 output classes to match default
        classifier = FlexibleClassifier(base_layer, n_kernels=2, is_crf=True, n_iter=1)
        
        X = torch.randn(3, 5)  # Match input dimension
        f = [torch.randn(3, 3), torch.randn(3, 3)]  # Single filter matching n_kernels=1
        sample_data = MockData(X, f)
        output = classifier(sample_data)
        assert output.shape == (3, 2)  # Default n_classes=2


class TestMLP:
    """Test MLP model."""
    
    def test_init_default_activation(self):
        """Test MLP initialization with default ReLU activation."""
        mlp = MLP(input_dim=20, hidden_dim=15, output_dim=5)
        
        assert mlp.layer1.in_features == 20
        assert mlp.layer1.out_features == 15
        assert mlp.layer2.in_features == 15
        assert mlp.layer2.out_features == 5
        assert mlp.activation.__class__.__name__ == 'ReLU'
    
    def test_init_custom_activation(self):
        """Test MLP initialization with custom activation."""
        custom_activation = Tanh()
        mlp = MLP(input_dim=10, hidden_dim=8, output_dim=3, activation=custom_activation)
        
        assert mlp.activation == custom_activation
    
    def test_forward(self):
        """Test MLP forward pass."""
        mlp = MLP(input_dim=10, hidden_dim=8, output_dim=3)
        x = torch.randn(4, 10)  # batch_size=4, input_dim=10
        
        output = mlp(x)
        
        assert output.shape == (4, 3)
        # Test that it's not just zeros (network is actually computing)
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_forward_single_sample(self):
        """Test MLP with single sample."""
        mlp = MLP(input_dim=5, hidden_dim=3, output_dim=2)
        x = torch.randn(1, 5)  # Single sample
        
        output = mlp(x)
        assert output.shape == (1, 2)


class TestAffine:
    """Test Affine model."""
    
    def test_init(self):
        """Test Affine initialization."""
        affine = Affine(n_features=15, n_classes=4)
        
        assert affine.layer.in_features == 15
        assert affine.layer.out_features == 4
    
    def test_forward(self):
        """Test Affine forward pass."""
        affine = Affine(n_features=8, n_classes=3)
        data = MockData(torch.randn(6, 8))
        
        output = affine(data)
        
        assert output.shape == (6, 3)


class TestPerceptron:
    """Test Perceptron model."""
    
    def test_init(self):
        """Test Perceptron initialization."""
        perceptron = Perceptron(n_features=12, n_hidden=8, n_classes=4)
        
        assert perceptron.layer1.in_features == 12
        assert perceptron.layer1.out_features == 8
        assert perceptron.layer2.in_features == 8
        assert perceptron.layer2.out_features == 4
    
    def test_forward(self):
        """Test Perceptron forward pass."""
        perceptron = Perceptron(n_features=10, n_hidden=6, n_classes=3)
        data = MockData(torch.randn(7, 10))
        
        output = perceptron(data)
        
        assert output.shape == (7, 3)
        # Check that ReLU is working (no negative values after first layer)
        # We can't directly check this, but we can verify output is reasonable
        assert not torch.all(output == 0)


class TestIntegration:
    """Integration tests using models together."""
    
    def test_mlp_as_base_layer_in_flexible_classifier(self):
        """Test using MLP as base layer in FlexibleClassifier."""
        base_mlp = MLP(input_dim=15, hidden_dim=10, output_dim=4)
        classifier = FlexibleClassifier(base_mlp, n_classes=4, is_crf=False)
        
        data = MockData(torch.randn(3, 15), [torch.randn(3, 3)])
        output = classifier(data)
        
        assert output.shape == (3, 4)
        assert torch.allclose(output.sum(dim=1), torch.ones(3), atol=1e-6)
    
    def test_affine_as_base_layer_in_crf_classifier(self):
        """Test using Affine-like layer in CRF classifier."""
        base_layer = Linear(8, 2)
        classifier = FlexibleClassifier(
            base_layer, n_classes=2, n_kernels=1, is_crf=True, n_iter=2
        )
        
        X = torch.randn(4, 8)
        f = [torch.randn(4, 4)]  # Single spatial filter
        data = MockData(X, f)
        
        output = classifier(data)
        
        assert output.shape == (4, 2)
        assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-6)


# Edge case tests
class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_flexible_classifier_with_zero_iterations(self):
        """Test CRF with zero iterations (should be like standard forward)."""
        base_layer = Linear(5, 2)
        classifier = FlexibleClassifier(base_layer, is_crf=True, n_iter=0)
        
        X = torch.randn(3, 5)
        f = [torch.randn(3, 3)]
        data = MockData(X, f)
        
        output = classifier(data)
        assert output.shape == (3, 2)
    
    def test_single_class_output(self):
        """Test with single class output."""
        base_layer = Linear(5, 1)
        classifier = FlexibleClassifier(base_layer, n_classes=1, is_crf=False)
        
        data = MockData(torch.randn(2, 5), [torch.randn(2, 2)])
        output = classifier(data)
        
        assert output.shape == (2, 1)
        assert torch.allclose(output, torch.ones(2, 1), atol=1e-6)  # Should be all 1s
    
    def test_large_batch_size(self):
        """Test with larger batch size."""
        mlp = MLP(input_dim=10, hidden_dim=5, output_dim=3)
        x = torch.randn(100, 10)  # Large batch
        
        output = mlp(x)
        assert output.shape == (100, 3)