"""
Comprehensive tests for trainer module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import json

from gnn_trading.models.trainer import ModelTrainer


class TestModelTrainer:
    """Test suite for ModelTrainer"""
    
    def test_trainer_initialization(self, sample_config):
        """Test trainer initialization"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = ModelTrainer(
                config=sample_config,
                model=Mock(),
                train_loader=Mock(),
                val_loader=Mock(),
                save_dir=Path(tmp_dir)
            )
            
            assert trainer.config == sample_config
            assert trainer.save_dir == Path(tmp_dir)
            assert trainer.device.type in ['cpu', 'cuda']
            assert trainer.best_val_loss == float('inf')
            assert trainer.patience_counter == 0
            
    def test_trainer_initialization_with_device(self, sample_config):
        """Test trainer initialization with specific device"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = ModelTrainer(
                config=sample_config,
                model=Mock(),
                train_loader=Mock(),
                val_loader=Mock(),
                save_dir=Path(tmp_dir),
                device='cpu'
            )
            
            assert trainer.device.type == 'cpu'
            
    def test_save_checkpoint(self, sample_config):
        """Test checkpoint saving"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            
            # Mock model with state_dict
            mock_model = Mock()
            mock_model.state_dict.return_value = {'param1': 'value1'}
            
            # Mock optimizer with state_dict
            mock_optimizer = Mock()
            mock_optimizer.state_dict.return_value = {'lr': 0.001}
            
            trainer = ModelTrainer(
                config=sample_config,
                model=mock_model,
                train_loader=Mock(),
                val_loader=Mock(),
                save_dir=save_dir
            )
            trainer.optimizer = mock_optimizer
            
            # Save checkpoint
            trainer.save_checkpoint(epoch=5, val_loss=0.123)
            
            # Check that checkpoint file exists
            checkpoint_path = save_dir / "checkpoint_epoch_5.pt"
            assert checkpoint_path.exists()
            
    def test_load_checkpoint(self, sample_config):
        """Test checkpoint loading"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            
            # Create mock checkpoint file
            checkpoint_data = {
                'epoch': 10,
                'model_state_dict': {'param1': 'value1'},
                'optimizer_state_dict': {'lr': 0.001},
                'val_loss': 0.456,
                'config': sample_config
            }
            
            checkpoint_path = save_dir / "checkpoint_epoch_10.pt"
            
            # Mock torch.save and torch.load
            with patch('torch.save') as mock_save, \
                 patch('torch.load', return_value=checkpoint_data) as mock_load:
                
                # Save first (to create file)
                mock_save.return_value = None
                
                mock_model = Mock()
                mock_optimizer = Mock()
                
                trainer = ModelTrainer(
                    config=sample_config,
                    model=mock_model,
                    train_loader=Mock(),
                    val_loader=Mock(),
                    save_dir=save_dir
                )
                trainer.optimizer = mock_optimizer
                
                # Load checkpoint
                epoch = trainer.load_checkpoint(checkpoint_path)
                
                assert epoch == 10
                mock_model.load_state_dict.assert_called_once_with({'param1': 'value1'})
                mock_optimizer.load_state_dict.assert_called_once_with({'lr': 0.001})
                
    def test_compute_metrics(self, sample_config):
        """Test metrics computation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = ModelTrainer(
                config=sample_config,
                model=Mock(),
                train_loader=Mock(),
                val_loader=Mock(),
                save_dir=Path(tmp_dir)
            )
            
            # Mock predictions and targets
            with patch('torch.tensor') as mock_tensor, \
                 patch('numpy.corrcoef') as mock_corrcoef:
                
                mock_tensor.side_effect = lambda x: MockTensor(x)
                mock_corrcoef.return_value = [[1.0, 0.75], [0.75, 1.0]]
                
                predictions = [1.0, 2.0, 3.0, 4.0]
                targets = [1.1, 1.9, 3.1, 3.9]
                
                metrics = trainer.compute_metrics(predictions, targets)
                
                assert 'mse' in metrics
                assert 'mae' in metrics
                assert 'correlation' in metrics
                assert 'directional_accuracy' in metrics
                
                # Correlation should be approximately 0.75
                assert abs(metrics['correlation'] - 0.75) < 0.1
                
    def test_train_epoch(self, sample_config):
        """Test training epoch"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock model, optimizer, criterion
            mock_model = Mock()
            mock_model.return_value = MockTensor([1.0, 2.0, 3.0])
            
            mock_optimizer = Mock()
            mock_criterion = Mock()
            mock_criterion.return_value = MockTensor([0.5])
            
            # Mock data loader
            mock_batch1 = (['graph1'], MockTensor([1.1, 2.1, 3.1]))
            mock_batch2 = (['graph2'], MockTensor([0.9, 1.9, 2.9]))
            mock_train_loader = [mock_batch1, mock_batch2]
            
            trainer = ModelTrainer(
                config=sample_config,
                model=mock_model,
                train_loader=mock_train_loader,
                val_loader=Mock(),
                save_dir=Path(tmp_dir)
            )
            trainer.optimizer = mock_optimizer
            trainer.criterion = mock_criterion
            
            # Train epoch
            epoch_loss = trainer.train_epoch()
            
            # Check that optimizer was called
            assert mock_optimizer.zero_grad.call_count == 2
            assert mock_optimizer.step.call_count == 2
            
            # Check that model was called
            assert mock_model.call_count == 2
            
            # Check that loss is reasonable
            assert isinstance(epoch_loss, (int, float))
            assert epoch_loss >= 0
            
    def test_validate_epoch(self, sample_config):
        """Test validation epoch"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock model and criterion
            mock_model = Mock()
            mock_model.return_value = MockTensor([1.0, 2.0])
            
            mock_criterion = Mock()
            mock_criterion.return_value = MockTensor([0.3])
            
            # Mock validation data loader
            mock_batch = (['graph1'], MockTensor([1.1, 2.1]))
            mock_val_loader = [mock_batch]
            
            trainer = ModelTrainer(
                config=sample_config,
                model=mock_model,
                train_loader=Mock(),
                val_loader=mock_val_loader,
                save_dir=Path(tmp_dir)
            )
            trainer.criterion = mock_criterion
            
            # Validate epoch
            val_loss, metrics = trainer.validate_epoch()
            
            # Check results
            assert isinstance(val_loss, (int, float))
            assert val_loss >= 0
            assert isinstance(metrics, dict)
            
    def test_early_stopping(self, sample_config):
        """Test early stopping mechanism"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = ModelTrainer(
                config=sample_config,
                model=Mock(),
                train_loader=Mock(),
                val_loader=Mock(),
                save_dir=Path(tmp_dir)
            )
            
            # Test improving validation loss
            assert not trainer.early_stopping(0.5)  # First call
            assert trainer.best_val_loss == 0.5
            assert trainer.patience_counter == 0
            
            assert not trainer.early_stopping(0.4)  # Improvement
            assert trainer.best_val_loss == 0.4
            assert trainer.patience_counter == 0
            
            # Test non-improving validation loss
            assert not trainer.early_stopping(0.6)  # No improvement
            assert trainer.best_val_loss == 0.4
            assert trainer.patience_counter == 1
            
            # Continue until patience is exceeded
            for i in range(2, sample_config['training']['patience']):
                assert not trainer.early_stopping(0.6)
                assert trainer.patience_counter == i
                
            # Should trigger early stopping
            assert trainer.early_stopping(0.6)
            
    def test_train_full_cycle(self, sample_config):
        """Test full training cycle"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock all components
            mock_model = Mock()
            mock_model.return_value = MockTensor([1.0])
            
            mock_optimizer = Mock()
            mock_criterion = Mock()
            mock_criterion.return_value = MockTensor([0.1])
            
            # Simple mock data loaders
            mock_train_batch = (['graph'], MockTensor([1.1]))
            mock_val_batch = (['graph'], MockTensor([1.1]))
            
            trainer = ModelTrainer(
                config=sample_config,
                model=mock_model,
                train_loader=[mock_train_batch],
                val_loader=[mock_val_batch],
                save_dir=Path(tmp_dir)
            )
            trainer.optimizer = mock_optimizer
            trainer.criterion = mock_criterion
            
            # Mock methods to avoid torch dependencies
            trainer.train_epoch = Mock(return_value=0.1)
            trainer.validate_epoch = Mock(return_value=(0.05, {'mse': 0.05}))
            trainer.save_checkpoint = Mock()
            
            # Train
            history = trainer.train(epochs=3)
            
            # Check results
            assert isinstance(history, dict)
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) <= 3  # May stop early
            
    def test_learning_rate_scheduling(self, sample_config):
        """Test learning rate scheduling"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_model = Mock()
            mock_optimizer = Mock()
            mock_optimizer.param_groups = [{'lr': 0.001}]
            
            trainer = ModelTrainer(
                config=sample_config,
                model=mock_model,
                train_loader=Mock(),
                val_loader=Mock(),
                save_dir=Path(tmp_dir)
            )
            trainer.optimizer = mock_optimizer
            
            # Mock scheduler
            mock_scheduler = Mock()
            trainer.scheduler = mock_scheduler
            
            # Test scheduler step is called
            trainer.step_scheduler(0.1)
            mock_scheduler.step.assert_called_once_with(0.1)
            
    def test_model_device_movement(self, sample_config):
        """Test model device movement"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_model = Mock()
            
            trainer = ModelTrainer(
                config=sample_config,
                model=mock_model,
                train_loader=Mock(),
                val_loader=Mock(),
                save_dir=Path(tmp_dir),
                device='cpu'
            )
            
            # Check that model was moved to device
            mock_model.to.assert_called_once_with(trainer.device)


class MockTensor:
    """Mock tensor class for testing"""
    
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
        
    def item(self):
        return self.data[0] if len(self.data) == 1 else self.data
        
    def cpu(self):
        return self
        
    def numpy(self):
        import numpy as np
        return np.array(self.data)
        
    def backward(self):
        pass
        
    def __float__(self):
        return float(self.data[0]) if len(self.data) == 1 else sum(self.data) / len(self.data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
