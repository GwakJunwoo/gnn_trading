"""
tests.test_ensemble
===================

Comprehensive tests for the ensemble model system
Tests all ensemble strategies, performance tracking, and edge cases
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from gnn_trading.models.ensemble import (
    EnsemblePredictor, 
    EnsembleConfig, 
    EnsembleStrategy,
    ModelPerformance
)


class TestEnsembleConfig:
    """Test EnsembleConfig validation and initialization"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EnsembleConfig()
        assert config.strategy == EnsembleStrategy.WEIGHTED_AVERAGE
        assert config.max_models == 5
        assert config.min_models == 2
        assert config.performance_window == 100
        assert config.rebalance_frequency == 10
        assert config.auto_remove_threshold == 0.3
        
    def test_custom_config(self):
        """Test custom configuration values"""
        config = EnsembleConfig(
            strategy=EnsembleStrategy.VOTING,
            max_models=10,
            min_models=3,
            performance_window=200,
            rebalance_frequency=20,
            auto_remove_threshold=0.1
        )
        assert config.strategy == EnsembleStrategy.VOTING
        assert config.max_models == 10
        assert config.min_models == 3
        assert config.performance_window == 200
        assert config.rebalance_frequency == 20
        assert config.auto_remove_threshold == 0.1
        
    def test_invalid_config(self):
        """Test invalid configuration values"""
        with pytest.raises(ValueError):
            EnsembleConfig(max_models=1, min_models=2)
            
        with pytest.raises(ValueError):
            EnsembleConfig(auto_remove_threshold=-0.1)
            
        with pytest.raises(ValueError):
            EnsembleConfig(auto_remove_threshold=1.1)


class TestModelPerformance:
    """Test ModelPerformance tracking"""
    
    def test_initialization(self):
        """Test performance tracker initialization"""
        perf = ModelPerformance("test_model", window_size=50)
        assert perf.model_name == "test_model"
        assert perf.window_size == 50
        assert len(perf.scores) == 0
        assert len(perf.predictions) == 0
        assert len(perf.actuals) == 0
        
    def test_add_result(self):
        """Test adding prediction results"""
        perf = ModelPerformance("test_model", window_size=3)
        
        # Add results
        perf.add_result(0.8, 0.75, 1.0)
        perf.add_result(0.7, 0.65, 0.9)
        perf.add_result(0.9, 0.85, 1.1)
        
        assert len(perf.scores) == 3
        assert len(perf.predictions) == 3
        assert len(perf.actuals) == 3
        
        # Test window overflow
        perf.add_result(0.6, 0.55, 0.8)
        assert len(perf.scores) == 3  # Should maintain window size
        assert perf.scores[0] == 0.7  # First score should be removed
        
    def test_metrics_calculation(self):
        """Test performance metrics calculation"""
        perf = ModelPerformance("test_model")
        
        # Add some results
        perf.add_result(0.8, 0.75, 1.0)
        perf.add_result(0.7, 0.65, 0.9)
        perf.add_result(0.9, 0.85, 1.1)
        
        metrics = perf.get_metrics()
        
        assert "mean_score" in metrics
        assert "std_score" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert "correlation" in metrics
        assert "sample_count" in metrics
        
        assert metrics["sample_count"] == 3
        assert np.isclose(metrics["mean_score"], np.mean([0.8, 0.7, 0.9]))
        
    def test_empty_metrics(self):
        """Test metrics with no data"""
        perf = ModelPerformance("test_model")
        metrics = perf.get_metrics()
        
        assert metrics["mean_score"] == 0.0
        assert metrics["std_score"] == 0.0
        assert metrics["sample_count"] == 0


class TestEnsemblePredictor:
    """Test EnsemblePredictor functionality"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model"""
        model = Mock()
        model.eval.return_value = None
        model.train.return_value = None
        return model
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return {
            'node_features': torch.randn(100, 10),
            'edge_index': torch.randint(0, 100, (2, 200)),
            'edge_attr': torch.randn(200, 5),
            'y': torch.randn(100, 1)
        }
        
    def test_initialization(self):
        """Test ensemble initialization"""
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        
        assert ensemble.config == config
        assert len(ensemble.models) == 0
        assert len(ensemble.model_performances) == 0
        assert len(ensemble.weights) == 0
        
    def test_add_model(self, mock_model):
        """Test adding models to ensemble"""
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        
        # Add model
        ensemble.add_model("model_1", mock_model)
        
        assert "model_1" in ensemble.models
        assert "model_1" in ensemble.model_performances
        assert ensemble.models["model_1"] == mock_model
        
    def test_add_duplicate_model(self, mock_model):
        """Test adding duplicate model names"""
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        
        ensemble.add_model("model_1", mock_model)
        
        with pytest.raises(ValueError, match="Model model_1 already exists"):
            ensemble.add_model("model_1", mock_model)
            
    def test_max_models_limit(self, mock_model):
        """Test maximum models limit"""
        config = EnsembleConfig(max_models=2)
        ensemble = EnsemblePredictor(config)
        
        ensemble.add_model("model_1", Mock())
        ensemble.add_model("model_2", Mock())
        
        with pytest.raises(ValueError, match="Maximum number of models"):
            ensemble.add_model("model_3", Mock())
            
    def test_remove_model(self, mock_model):
        """Test removing models"""
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        
        ensemble.add_model("model_1", mock_model)
        ensemble.add_model("model_2", Mock())
        
        # Remove model
        ensemble.remove_model("model_1")
        
        assert "model_1" not in ensemble.models
        assert "model_1" not in ensemble.model_performances
        
    def test_remove_nonexistent_model(self):
        """Test removing non-existent model"""
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        
        with pytest.raises(ValueError, match="Model nonexistent not found"):
            ensemble.remove_model("nonexistent")
            
    def test_weighted_average_strategy(self, sample_data):
        """Test weighted average ensemble strategy"""
        config = EnsembleConfig(strategy=EnsembleStrategy.WEIGHTED_AVERAGE)
        ensemble = EnsemblePredictor(config)
        
        # Create mock models with different predictions
        model1 = Mock()
        model1.eval.return_value = None
        model1.return_value = torch.tensor([[0.8], [0.7]])
        
        model2 = Mock()
        model2.eval.return_value = None
        model2.return_value = torch.tensor([[0.6], [0.9]])
        
        ensemble.add_model("model_1", model1)
        ensemble.add_model("model_2", model2)
        
        # Set equal weights
        ensemble.weights = {"model_1": 0.5, "model_2": 0.5}
        
        with patch.object(ensemble, '_get_model_predictions') as mock_predictions:
            mock_predictions.return_value = {
                "model_1": torch.tensor([[0.8], [0.7]]),
                "model_2": torch.tensor([[0.6], [0.9]])
            }
            
            prediction, metadata = ensemble.predict(sample_data)
            
            expected = torch.tensor([[0.7], [0.8]])  # (0.8*0.5 + 0.6*0.5), (0.7*0.5 + 0.9*0.5)
            assert torch.allclose(prediction, expected)
            assert "strategy" in metadata
            assert "model_contributions" in metadata
            
    def test_voting_strategy(self, sample_data):
        """Test voting ensemble strategy"""
        config = EnsembleConfig(strategy=EnsembleStrategy.VOTING)
        ensemble = EnsemblePredictor(config)
        
        # Add mock models
        ensemble.add_model("model_1", Mock())
        ensemble.add_model("model_2", Mock())
        ensemble.add_model("model_3", Mock())
        
        with patch.object(ensemble, '_get_model_predictions') as mock_predictions:
            # Mock predictions where model_1 and model_2 agree
            mock_predictions.return_value = {
                "model_1": torch.tensor([[0.8], [0.7]]),
                "model_2": torch.tensor([[0.9], [0.6]]),
                "model_3": torch.tensor([[0.2], [0.8]])
            }
            
            prediction, metadata = ensemble.predict(sample_data)
            
            # Should take median values
            expected = torch.tensor([[0.8], [0.7]])
            assert torch.allclose(prediction, expected)
            
    def test_stacking_strategy(self, sample_data):
        """Test stacking ensemble strategy"""
        config = EnsembleConfig(strategy=EnsembleStrategy.STACKING)
        ensemble = EnsemblePredictor(config)
        
        # Add mock models
        ensemble.add_model("model_1", Mock())
        ensemble.add_model("model_2", Mock())
        
        # Mock meta-learner
        meta_learner = Mock()
        meta_learner.eval.return_value = None
        meta_learner.return_value = torch.tensor([[0.75], [0.65]])
        ensemble.meta_learner = meta_learner
        
        with patch.object(ensemble, '_get_model_predictions') as mock_predictions:
            mock_predictions.return_value = {
                "model_1": torch.tensor([[0.8], [0.7]]),
                "model_2": torch.tensor([[0.6], [0.9]])
            }
            
            prediction, metadata = ensemble.predict(sample_data)
            
            expected = torch.tensor([[0.75], [0.65]])
            assert torch.allclose(prediction, expected)
            
    def test_fit_weighted_average(self, sample_data):
        """Test fitting ensemble with weighted average strategy"""
        config = EnsembleConfig(strategy=EnsembleStrategy.WEIGHTED_AVERAGE)
        ensemble = EnsemblePredictor(config)
        
        # Add mock models
        model1 = Mock()
        model2 = Mock()
        ensemble.add_model("model_1", model1)
        ensemble.add_model("model_2", model2)
        
        # Mock model predictions and performance
        with patch.object(ensemble, '_get_model_predictions') as mock_predictions, \
             patch.object(ensemble, '_calculate_model_performance') as mock_performance:
            
            mock_predictions.return_value = {
                "model_1": torch.tensor([[0.8], [0.7]]),
                "model_2": torch.tensor([[0.6], [0.9]])
            }
            mock_performance.side_effect = [0.8, 0.6]  # model_1 performs better
            
            ensemble.fit(sample_data)
            
            # Check that weights favor better performing model
            assert ensemble.weights["model_1"] > ensemble.weights["model_2"]
            assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6
            
    def test_fit_stacking(self, sample_data):
        """Test fitting ensemble with stacking strategy"""
        config = EnsembleConfig(strategy=EnsembleStrategy.STACKING)
        ensemble = EnsemblePredictor(config)
        
        # Add mock models
        ensemble.add_model("model_1", Mock())
        ensemble.add_model("model_2", Mock())
        
        with patch.object(ensemble, '_get_model_predictions') as mock_predictions, \
             patch.object(ensemble, '_train_meta_learner') as mock_train_meta:
            
            mock_predictions.return_value = {
                "model_1": torch.tensor([[0.8], [0.7]]),
                "model_2": torch.tensor([[0.6], [0.9]])
            }
            
            ensemble.fit(sample_data)
            
            mock_train_meta.assert_called_once()
            
    def test_performance_tracking(self, sample_data):
        """Test model performance tracking"""
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        
        model = Mock()
        ensemble.add_model("model_1", model)
        
        # Simulate predictions and track performance
        predictions = torch.tensor([[0.8], [0.7]])
        actuals = torch.tensor([[0.75], [0.65]])
        
        ensemble._update_performance("model_1", predictions, actuals)
        
        perf = ensemble.model_performances["model_1"]
        assert len(perf.predictions) == 2
        assert len(perf.actuals) == 2
        
    def test_auto_rebalancing(self, sample_data):
        """Test automatic model rebalancing"""
        config = EnsembleConfig(rebalance_frequency=2)
        ensemble = EnsemblePredictor(config)
        
        # Add models
        ensemble.add_model("model_1", Mock())
        ensemble.add_model("model_2", Mock())
        
        with patch.object(ensemble, '_rebalance_weights') as mock_rebalance:
            # First prediction - no rebalancing
            ensemble.prediction_count = 1
            ensemble.predict(sample_data)
            mock_rebalance.assert_not_called()
            
            # Second prediction - should trigger rebalancing
            ensemble.prediction_count = 2
            ensemble.predict(sample_data)
            mock_rebalance.assert_called_once()
            
    def test_auto_model_removal(self, sample_data):
        """Test automatic removal of poor performing models"""
        config = EnsembleConfig(auto_remove_threshold=0.5, min_models=1)
        ensemble = EnsemblePredictor(config)
        
        # Add models
        ensemble.add_model("good_model", Mock())
        ensemble.add_model("bad_model", Mock())
        
        # Mock performance - bad model has low score
        ensemble.model_performances["good_model"].scores = [0.8, 0.9, 0.7]
        ensemble.model_performances["bad_model"].scores = [0.3, 0.2, 0.4]
        
        with patch.object(ensemble, '_check_auto_removal') as mock_check:
            mock_check.return_value = ["bad_model"]
            ensemble.predict(sample_data)
            
            mock_check.assert_called_once()
            
    def test_save_load_ensemble(self, tmp_path):
        """Test saving and loading ensemble"""
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        
        # Add mock models
        ensemble.add_model("model_1", Mock())
        ensemble.weights = {"model_1": 1.0}
        
        # Save ensemble
        save_path = tmp_path / "ensemble.pt"
        ensemble.save(str(save_path))
        
        assert save_path.exists()
        
        # Load ensemble
        new_ensemble = EnsemblePredictor(config)
        new_ensemble.load(str(save_path))
        
        assert "model_1" in new_ensemble.weights
        assert new_ensemble.weights["model_1"] == 1.0
        
    def test_get_model_info(self):
        """Test getting model information"""
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        
        # Add models
        ensemble.add_model("model_1", Mock())
        ensemble.add_model("model_2", Mock())
        
        # Add some performance data
        ensemble.model_performances["model_1"].scores = [0.8, 0.9]
        ensemble.model_performances["model_2"].scores = [0.6, 0.7]
        
        info = ensemble.get_model_info()
        
        assert "models" in info
        assert "total_models" in info
        assert "strategy" in info
        assert len(info["models"]) == 2
        assert info["total_models"] == 2
        
    def test_insufficient_models_error(self, sample_data):
        """Test error when trying to predict with insufficient models"""
        config = EnsembleConfig(min_models=2)
        ensemble = EnsemblePredictor(config)
        
        # Add only one model
        ensemble.add_model("model_1", Mock())
        
        with pytest.raises(ValueError, match="Insufficient models"):
            ensemble.predict(sample_data)
            
    def test_no_models_error(self, sample_data):
        """Test error when trying to predict with no models"""
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        
        with pytest.raises(ValueError, match="No models available"):
            ensemble.predict(sample_data)


class TestEnsembleIntegration:
    """Integration tests for ensemble system"""
    
    def test_full_ensemble_workflow(self, sample_data):
        """Test complete ensemble workflow"""
        config = EnsembleConfig(
            strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
            max_models=3,
            performance_window=10
        )
        ensemble = EnsemblePredictor(config)
        
        # Create multiple mock models
        models = []
        for i in range(3):
            model = Mock()
            model.eval.return_value = None
            models.append(model)
            ensemble.add_model(f"model_{i}", model)
            
        # Mock predictions
        with patch.object(ensemble, '_get_model_predictions') as mock_predictions:
            mock_predictions.return_value = {
                "model_0": torch.tensor([[0.8], [0.7]]),
                "model_1": torch.tensor([[0.6], [0.9]]),
                "model_2": torch.tensor([[0.7], [0.8]])
            }
            
            # Fit ensemble
            ensemble.fit(sample_data)
            
            # Make predictions
            prediction, metadata = ensemble.predict(sample_data)
            
            assert prediction is not None
            assert "strategy" in metadata
            assert "model_contributions" in metadata
            assert len(ensemble.weights) == 3
            
    def test_ensemble_with_performance_tracking(self, sample_data):
        """Test ensemble with realistic performance tracking"""
        config = EnsembleConfig(performance_window=5)
        ensemble = EnsemblePredictor(config)
        
        # Add models
        ensemble.add_model("model_1", Mock())
        ensemble.add_model("model_2", Mock())
        
        # Simulate multiple prediction cycles
        for i in range(10):
            predictions = {
                "model_1": torch.tensor([[0.8 + 0.1 * np.random.randn()], [0.7]]),
                "model_2": torch.tensor([[0.6], [0.9 + 0.1 * np.random.randn()]])
            }
            actuals = torch.tensor([[0.75], [0.8]])
            
            with patch.object(ensemble, '_get_model_predictions') as mock_predictions:
                mock_predictions.return_value = predictions
                
                # Update performance
                for model_name, pred in predictions.items():
                    ensemble._update_performance(model_name, pred, actuals)
                    
        # Check that performance history is maintained
        for model_name in ["model_1", "model_2"]:
            perf = ensemble.model_performances[model_name]
            assert len(perf.scores) <= config.performance_window
            assert len(perf.predictions) <= config.performance_window
