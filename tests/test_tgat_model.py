"""
Comprehensive tests for TGAT model
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import numpy as np

# Import our modules
from gnn_trading.models.tgat import TGATModel


class TestTGATModel:
    """Test suite for TGAT model"""
    
    def test_model_initialization(self):
        """Test model initialization with default parameters"""
        model = TGATModel()
        assert model.in_dim == 1
        assert model.hidden_dim == 32
        assert model.num_layers == 2
        assert model.heads == 2
        assert model.dropout == 0.1
        
    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters"""
        model = TGATModel(
            in_dim=10,
            hidden_dim=64,
            num_layers=3,
            heads=4,
            dropout=0.2
        )
        assert model.in_dim == 10
        assert model.hidden_dim == 64
        assert model.num_layers == 3
        assert model.heads == 4
        assert model.dropout == 0.2
        
    def test_weight_initialization(self):
        """Test that weights are properly initialized"""
        model = TGATModel(hidden_dim=32)
        
        # Check that GAT layers have parameters
        for layer in model.gat_layers:
            assert hasattr(layer, 'lin_l')
            assert hasattr(layer, 'lin_r')
            
        # Check GRU initialization
        for name, param in model.gru.named_parameters():
            if 'weight' in name:
                # Xavier initialization should have specific variance
                std = np.sqrt(2.0 / (param.size(0) + param.size(1)))
                assert param.std().item() < 2 * std
            elif 'bias' in name:
                assert torch.allclose(param, torch.zeros_like(param))
                
        # Check output layer initialization
        assert model.fc_out.bias.abs().max().item() < 1e-6
        
    def test_forward_pass_empty_batch(self):
        """Test forward pass with empty batch"""
        model = TGATModel()
        with pytest.raises(ValueError, match="Empty batch_data"):
            model.forward([])
            
    def test_get_node_embeddings_empty_batch(self):
        """Test node embeddings with empty batch"""
        model = TGATModel()
        with pytest.raises(ValueError, match="Empty batch_data"):
            model.get_node_embeddings([])
            
    @patch('torch_geometric.data.Data')
    def test_forward_pass_single_graph(self, mock_data):
        """Test forward pass with single graph"""
        # Mock the data object
        mock_graph = Mock()
        mock_graph.x = torch.randn(10, 1)  # 10 nodes, 1 feature
        mock_graph.edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
        
        model = TGATModel(in_dim=1, hidden_dim=32)
        
        # Mock the _encode_graph method to avoid torch_geometric dependency
        with patch.object(model, '_encode_graph') as mock_encode:
            mock_encode.return_value = torch.randn(10, 32)
            
            result = model.forward([mock_graph])
            
            assert result.shape == (10, 1)
            mock_encode.assert_called_once_with(mock_graph)
            
    @patch('torch_geometric.data.Data')
    def test_forward_pass_multiple_graphs(self, mock_data):
        """Test forward pass with multiple graphs (time sequence)"""
        # Create mock graphs
        mock_graphs = []
        for _ in range(5):  # 5 time steps
            mock_graph = Mock()
            mock_graph.x = torch.randn(10, 1)
            mock_graph.edge_index = torch.randint(0, 10, (2, 20))
            mock_graphs.append(mock_graph)
            
        model = TGATModel(in_dim=1, hidden_dim=32)
        
        # Mock the _encode_graph method
        with patch.object(model, '_encode_graph') as mock_encode:
            mock_encode.return_value = torch.randn(10, 32)
            
            result = model.forward(mock_graphs)
            
            assert result.shape == (10, 1)
            assert mock_encode.call_count == 5
            
    @patch('torch_geometric.data.Data')
    def test_get_node_embeddings(self, mock_data):
        """Test node embeddings extraction"""
        mock_graphs = []
        for _ in range(3):
            mock_graph = Mock()
            mock_graph.x = torch.randn(5, 1)
            mock_graph.edge_index = torch.randint(0, 5, (2, 10))
            mock_graphs.append(mock_graph)
            
        model = TGATModel(in_dim=1, hidden_dim=16)
        
        with patch.object(model, '_encode_graph') as mock_encode:
            mock_encode.return_value = torch.randn(5, 16)
            
            embeddings = model.get_node_embeddings(mock_graphs)
            
            assert embeddings.shape == (5, 16)
            assert mock_encode.call_count == 3
            
    def test_model_parameters_count(self):
        """Test that model has reasonable number of parameters"""
        model = TGATModel(in_dim=5, hidden_dim=32, num_layers=2)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters (not too few, not too many)
        assert 1000 < total_params < 100000
        
    def test_model_eval_mode(self):
        """Test model in evaluation mode"""
        model = TGATModel()
        model.eval()
        
        # In eval mode, dropout should be disabled
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                assert not module.training
                
    def test_model_train_mode(self):
        """Test model in training mode"""
        model = TGATModel()
        model.train()
        
        # In train mode, dropout should be enabled
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                assert module.training
                
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model"""
        model = TGATModel(in_dim=1, hidden_dim=16)
        
        # Create mock data
        mock_graph = Mock()
        mock_graph.x = torch.randn(5, 1, requires_grad=True)
        mock_graph.edge_index = torch.randint(0, 5, (2, 8))
        
        with patch.object(model, '_encode_graph') as mock_encode:
            mock_encode.return_value = torch.randn(5, 16, requires_grad=True)
            
            # Forward pass
            output = model.forward([mock_graph])
            
            # Compute loss and backward pass
            target = torch.randn(5, 1)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            # Check that model parameters have gradients
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    
    def test_model_device_compatibility(self):
        """Test model can be moved to different devices"""
        model = TGATModel()
        
        # Test CPU
        model = model.cpu()
        for param in model.parameters():
            assert param.device.type == 'cpu'
            
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            for param in model.parameters():
                assert param.device.type == 'cuda'
                
    def test_model_state_dict_save_load(self):
        """Test model state dict save and load"""
        model1 = TGATModel(hidden_dim=32)
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Create new model and load state dict
        model2 = TGATModel(hidden_dim=32)
        model2.load_state_dict(state_dict)
        
        # Check that parameters are the same
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
            
    def test_attention_weights_extraction(self):
        """Test attention weights extraction functionality"""
        model = TGATModel()
        
        mock_graph = Mock()
        mock_graph.x = torch.randn(5, 1)
        mock_graph.edge_index = torch.randint(0, 5, (2, 8))
        
        # Mock GAT layer with attention weights
        with patch.object(model.gat_layers[0], '__call__') as mock_gat:
            mock_gat.return_value = torch.randn(5, 64)  # heads * hidden_dim
            
            embeddings = model._encode_graph(mock_graph)
            assert embeddings.shape[0] == 5  # num_nodes
            
    def test_model_reproducibility(self):
        """Test model reproducibility with same random seed"""
        torch.manual_seed(42)
        model1 = TGATModel()
        
        torch.manual_seed(42)
        model2 = TGATModel()
        
        # Models should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


class TestTGATModelIntegration:
    """Integration tests for TGAT model"""
    
    def test_overfitting_single_batch(self):
        """Test that model can overfit a single batch (sanity check)"""
        model = TGATModel(in_dim=1, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Create mock data
        mock_graph = Mock()
        mock_graph.x = torch.randn(5, 1)
        mock_graph.edge_index = torch.randint(0, 5, (2, 8))
        
        target = torch.randn(5, 1)
        
        initial_loss = None
        final_loss = None
        
        with patch.object(model, '_encode_graph') as mock_encode:
            mock_encode.return_value = torch.randn(5, 32, requires_grad=True)
            
            # Train for several iterations
            for i in range(100):
                optimizer.zero_grad()
                output = model.forward([mock_graph])
                loss = criterion(output, target)
                
                if i == 0:
                    initial_loss = loss.item()
                    
                loss.backward()
                optimizer.step()
                
            final_loss = loss.item()
            
        # Model should be able to overfit
        assert final_loss < initial_loss * 0.1
        
    def test_model_memory_usage(self):
        """Test model memory usage is reasonable"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        model = TGATModel(hidden_dim=64).cuda()
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create large batch
        mock_graphs = []
        for _ in range(10):
            mock_graph = Mock()
            mock_graph.x = torch.randn(100, 1).cuda()
            mock_graph.edge_index = torch.randint(0, 100, (2, 200)).cuda()
            mock_graphs.append(mock_graph)
            
        with patch.object(model, '_encode_graph') as mock_encode:
            mock_encode.return_value = torch.randn(100, 64, requires_grad=True).cuda()
            
            output = model.forward(mock_graphs)
            
        final_memory = torch.cuda.memory_allocated()
        memory_used = final_memory - initial_memory
        
        # Should use less than 1GB for this size
        assert memory_used < 1024 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
