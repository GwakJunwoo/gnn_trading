"""
gnn_trading.models.ensemble
==========================

Model ensemble system for improved prediction accuracy and robustness.
Supports multiple ensemble strategies and model combination methods.
"""

from __future__ import annotations
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

# Local imports will be resolved at runtime
# import torch
# import torch.nn as nn
# from torch_geometric.data import Data


@dataclass
class EnsembleConfig:
    """Configuration for ensemble predictor"""
    combination_method: str = "weighted_average"  # weighted_average, voting, stacking
    weight_method: str = "performance"  # equal, performance, uncertainty, adaptive
    diversity_threshold: float = 0.1
    max_models: int = 10
    refit_frequency: int = 100  # predictions
    validation_size: float = 0.2
    enable_model_selection: bool = True
    enable_uncertainty_estimation: bool = True
    performance_window: int = 50
    min_correlation_threshold: float = 0.8  # for diversity
    
    def __post_init__(self):
        valid_combinations = ["weighted_average", "voting", "stacking", "bayesian"]
        if self.combination_method not in valid_combinations:
            raise ValueError(f"combination_method must be one of {valid_combinations}")
            
        valid_weights = ["equal", "performance", "uncertainty", "adaptive"]
        if self.weight_method not in valid_weights:
            raise ValueError(f"weight_method must be one of {valid_weights}")


class BaseEnsembleMethod(ABC):
    """Base class for ensemble combination methods"""
    
    @abstractmethod
    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Fit the ensemble method"""
        pass
        
    @abstractmethod
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        pass
        
    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """Get model weights"""
        pass


class WeightedAverageEnsemble(BaseEnsembleMethod):
    """Weighted average ensemble"""
    
    def __init__(self, weight_method: str = "performance"):
        self.weight_method = weight_method
        self.weights = None
        self.performance_history = []
        
    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Fit ensemble weights"""
        n_models = predictions.shape[1]
        
        if self.weight_method == "equal":
            self.weights = np.ones(n_models) / n_models
            
        elif self.weight_method == "performance":
            # Weight by inverse MSE
            mse_scores = []
            for i in range(n_models):
                mse = mean_squared_error(targets, predictions[:, i])
                mse_scores.append(mse)
                
            # Inverse MSE as weights
            inv_mse = 1.0 / (np.array(mse_scores) + 1e-8)
            self.weights = inv_mse / inv_mse.sum()
            
        elif self.weight_method == "uncertainty":
            # Weight by prediction uncertainty (lower uncertainty = higher weight)
            uncertainties = np.std(predictions, axis=1).mean()
            model_uncertainties = []
            
            for i in range(n_models):
                uncertainty = np.std(predictions[:, i])
                model_uncertainties.append(uncertainty)
                
            inv_uncertainty = 1.0 / (np.array(model_uncertainties) + 1e-8)
            self.weights = inv_uncertainty / inv_uncertainty.sum()
            
        elif self.weight_method == "adaptive":
            # Adaptive weights based on recent performance
            if len(self.performance_history) > 0:
                recent_performance = np.array(self.performance_history[-10:])  # Last 10 predictions
                avg_performance = recent_performance.mean(axis=0)
                inv_performance = 1.0 / (avg_performance + 1e-8)
                self.weights = inv_performance / inv_performance.sum()
            else:
                self.weights = np.ones(n_models) / n_models
                
        # Store performance for adaptive weighting
        current_performance = []
        for i in range(n_models):
            mse = mean_squared_error(targets, predictions[:, i])
            current_performance.append(mse)
        self.performance_history.append(current_performance)
        
        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
            
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Make weighted prediction"""
        if self.weights is None:
            # Equal weights if not fitted
            n_models = predictions.shape[1]
            self.weights = np.ones(n_models) / n_models
            
        return np.average(predictions, axis=1, weights=self.weights)
        
    def get_weights(self) -> np.ndarray:
        """Get current weights"""
        return self.weights if self.weights is not None else np.array([])


class VotingEnsemble(BaseEnsembleMethod):
    """Voting ensemble for classification-like tasks"""
    
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.weights = None
        
    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Fit voting ensemble"""
        n_models = predictions.shape[1]
        
        # Calculate accuracy for each model
        accuracies = []
        for i in range(n_models):
            pred_binary = (predictions[:, i] > self.threshold).astype(int)
            target_binary = (targets > self.threshold).astype(int)
            accuracy = (pred_binary == target_binary).mean()
            accuracies.append(accuracy)
            
        # Use accuracy as weights
        self.weights = np.array(accuracies)
        self.weights = self.weights / self.weights.sum()
        
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Make voting prediction"""
        if self.weights is None:
            # Equal weights
            n_models = predictions.shape[1]
            self.weights = np.ones(n_models) / n_models
            
        # Convert to binary votes
        votes = (predictions > self.threshold).astype(int)
        
        # Weighted voting
        weighted_votes = votes @ self.weights
        
        # Convert back to continuous predictions
        return weighted_votes
        
    def get_weights(self) -> np.ndarray:
        """Get voting weights"""
        return self.weights if self.weights is not None else np.array([])


class StackingEnsemble(BaseEnsembleMethod):
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, meta_learner=None):
        self.meta_learner = meta_learner
        self.is_fitted = False
        
    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Fit stacking ensemble"""
        # Use simple linear regression as meta-learner if not provided
        if self.meta_learner is None:
            from sklearn.linear_model import LinearRegression
            self.meta_learner = LinearRegression()
            
        # Fit meta-learner on model predictions
        self.meta_learner.fit(predictions, targets)
        self.is_fitted = True
        
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Make stacking prediction"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        return self.meta_learner.predict(predictions)
        
    def get_weights(self) -> np.ndarray:
        """Get stacking weights (coefficients)"""
        if hasattr(self.meta_learner, 'coef_'):
            return self.meta_learner.coef_
        else:
            return np.array([])


class ModelPerformanceTracker:
    """Track individual model performance"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: Dict[str, List[float]] = {}
        self.prediction_history: Dict[str, List[float]] = {}
        
    def update(self, model_id: str, prediction: float, target: float) -> None:
        """Update performance for a model"""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
            self.prediction_history[model_id] = []
            
        # Calculate error
        error = abs(prediction - target)
        
        # Add to history
        self.performance_history[model_id].append(error)
        self.prediction_history[model_id].append(prediction)
        
        # Maintain window size
        if len(self.performance_history[model_id]) > self.window_size:
            self.performance_history[model_id] = self.performance_history[model_id][-self.window_size:]
            self.prediction_history[model_id] = self.prediction_history[model_id][-self.window_size:]
            
    def get_performance(self, model_id: str) -> Dict[str, float]:
        """Get performance metrics for a model"""
        if model_id not in self.performance_history:
            return {"mae": 0.0, "std": 0.0, "count": 0}
            
        errors = self.performance_history[model_id]
        predictions = self.prediction_history[model_id]
        
        return {
            "mae": np.mean(errors),
            "std": np.std(predictions),
            "count": len(errors),
            "recent_mae": np.mean(errors[-10:]) if len(errors) >= 10 else np.mean(errors)
        }
        
    def get_all_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance for all models"""
        return {model_id: self.get_performance(model_id) 
                for model_id in self.performance_history.keys()}


class EnsemblePredictor:
    """Main ensemble predictor class"""
    
    def __init__(
        self,
        config: EnsembleConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # Ensemble method
        self.ensemble_method = self._create_ensemble_method()
        
        # Performance tracking
        self.performance_tracker = ModelPerformanceTracker(
            window_size=config.performance_window
        )
        
        # State
        self.is_fitted = False
        self.prediction_count = 0
        self.last_refit = 0
        
        # Validation data
        self.validation_predictions = []
        self.validation_targets = []
        
    def _create_ensemble_method(self) -> BaseEnsembleMethod:
        """Create ensemble method based on config"""
        if self.config.combination_method == "weighted_average":
            return WeightedAverageEnsemble(self.config.weight_method)
        elif self.config.combination_method == "voting":
            return VotingEnsemble()
        elif self.config.combination_method == "stacking":
            return StackingEnsemble()
        else:
            raise ValueError(f"Unknown combination method: {self.config.combination_method}")
            
    def add_model(
        self,
        model_id: str,
        model: Any,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add model to ensemble"""
        if len(self.models) >= self.config.max_models:
            self.logger.warning(f"Maximum models ({self.config.max_models}) reached")
            return
            
        self.models[model_id] = model
        self.model_metadata[model_id] = metadata or {}
        
        self.logger.info(f"Added model {model_id} to ensemble")
        
    def remove_model(self, model_id: str) -> None:
        """Remove model from ensemble"""
        if model_id in self.models:
            del self.models[model_id]
            del self.model_metadata[model_id]
            self.is_fitted = False  # Need to refit ensemble
            
            self.logger.info(f"Removed model {model_id} from ensemble")
            
    def _get_model_predictions(self, data: Any) -> Dict[str, float]:
        """Get predictions from all models"""
        predictions = {}
        
        for model_id, model in self.models.items():
            try:
                # Handle different model types
                if hasattr(model, 'predict'):
                    # Sklearn-like model
                    pred = model.predict(data)
                elif hasattr(model, 'forward'):
                    # PyTorch model
                    import torch
                    model.eval()
                    with torch.no_grad():
                        if isinstance(data, list):
                            pred = model(data).cpu().numpy()
                        else:
                            pred = model(data).cpu().numpy()
                elif callable(model):
                    # Function-like model
                    pred = model(data)
                else:
                    self.logger.warning(f"Unknown model type for {model_id}")
                    continue
                    
                # Handle different prediction shapes
                if hasattr(pred, 'shape') and len(pred.shape) > 0:
                    pred_value = float(pred.flat[0])  # Take first element
                else:
                    pred_value = float(pred)
                    
                predictions[model_id] = pred_value
                
            except Exception as e:
                self.logger.error(f"Error getting prediction from {model_id}: {e}")
                continue
                
        return predictions
        
    def _calculate_diversity(self, predictions: np.ndarray) -> float:
        """Calculate ensemble diversity"""
        if predictions.shape[1] < 2:
            return 0.0
            
        # Calculate pairwise correlations
        correlations = []
        n_models = predictions.shape[1]
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions[:, i], predictions[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
                    
        return 1.0 - np.mean(correlations) if correlations else 0.0
        
    def fit(self, validation_data: List[Tuple[Any, float]]) -> None:
        """Fit ensemble on validation data"""
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
            
        # Collect predictions and targets
        predictions_list = []
        targets = []
        
        for data, target in validation_data:
            model_preds = self._get_model_predictions(data)
            
            if len(model_preds) > 0:
                pred_values = [model_preds.get(mid, 0.0) for mid in self.models.keys()]
                predictions_list.append(pred_values)
                targets.append(target)
                
        if len(predictions_list) == 0:
            raise ValueError("No valid predictions for fitting")
            
        predictions = np.array(predictions_list)
        targets = np.array(targets)
        
        # Check diversity
        diversity = self._calculate_diversity(predictions)
        if diversity < self.config.diversity_threshold:
            self.logger.warning(f"Low ensemble diversity: {diversity:.3f}")
            
        # Fit ensemble method
        self.ensemble_method.fit(predictions, targets)
        
        # Store validation data
        self.validation_predictions = predictions_list
        self.validation_targets = targets
        
        self.is_fitted = True
        self.last_refit = self.prediction_count
        
        self.logger.info(f"Fitted ensemble with {len(self.models)} models, diversity: {diversity:.3f}")
        
    def predict(self, data: Any) -> Tuple[float, Dict[str, Any]]:
        """Make ensemble prediction"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        # Get individual model predictions
        model_predictions = self._get_model_predictions(data)
        
        if len(model_predictions) == 0:
            raise ValueError("No valid model predictions")
            
        # Prepare prediction array
        pred_values = [model_predictions.get(mid, 0.0) for mid in self.models.keys()]
        predictions = np.array(pred_values).reshape(1, -1)
        
        # Make ensemble prediction
        ensemble_pred = self.ensemble_method.predict(predictions)[0]
        
        # Calculate uncertainty
        uncertainty = np.std(pred_values) if len(pred_values) > 1 else 0.0
        
        # Prepare metadata
        metadata = {
            "individual_predictions": model_predictions,
            "weights": self.ensemble_method.get_weights().tolist(),
            "uncertainty": uncertainty,
            "n_models": len(model_predictions),
            "prediction_count": self.prediction_count
        }
        
        self.prediction_count += 1
        
        # Check if refit needed
        if (self.prediction_count - self.last_refit) >= self.config.refit_frequency:
            self._schedule_refit()
            
        return ensemble_pred, metadata
        
    def predict_with_uncertainty(self, data: Any, n_samples: int = 100) -> Tuple[float, float]:
        """Make prediction with uncertainty estimation"""
        if not self.config.enable_uncertainty_estimation:
            pred, metadata = self.predict(data)
            return pred, metadata.get("uncertainty", 0.0)
            
        # Bootstrap sampling for uncertainty
        predictions = []
        
        for _ in range(n_samples):
            # Sample models randomly
            sample_models = np.random.choice(
                list(self.models.keys()),
                size=min(len(self.models), 3),  # Sample 3 models
                replace=False
            )
            
            # Get predictions from sampled models
            sample_preds = []
            for model_id in sample_models:
                model = self.models[model_id]
                try:
                    pred = self._get_single_prediction(model, data)
                    sample_preds.append(pred)
                except:
                    continue
                    
            if sample_preds:
                predictions.append(np.mean(sample_preds))
                
        if predictions:
            mean_pred = np.mean(predictions)
            uncertainty = np.std(predictions)
            return mean_pred, uncertainty
        else:
            # Fallback to regular prediction
            pred, metadata = self.predict(data)
            return pred, metadata.get("uncertainty", 0.0)
            
    def _get_single_prediction(self, model: Any, data: Any) -> float:
        """Get single prediction from model"""
        if hasattr(model, 'predict'):
            pred = model.predict(data)
        elif hasattr(model, 'forward'):
            import torch
            model.eval()
            with torch.no_grad():
                pred = model(data).cpu().numpy()
        elif callable(model):
            pred = model(data)
        else:
            return 0.0
            
        # Handle different prediction shapes
        if hasattr(pred, 'shape') and len(pred.shape) > 0:
            return float(pred.flat[0])
        else:
            return float(pred)
            
    def update_performance(self, data: Any, target: float) -> None:
        """Update model performance tracking"""
        model_predictions = self._get_model_predictions(data)
        
        for model_id, prediction in model_predictions.items():
            self.performance_tracker.update(model_id, prediction, target)
            
    def _schedule_refit(self) -> None:
        """Schedule ensemble refit"""
        if len(self.validation_predictions) > 0:
            try:
                # Use stored validation data for refit
                predictions = np.array(self.validation_predictions)
                targets = np.array(self.validation_targets)
                
                self.ensemble_method.fit(predictions, targets)
                self.last_refit = self.prediction_count
                
                self.logger.info("Ensemble refitted")
                
            except Exception as e:
                self.logger.error(f"Error during refit: {e}")
                
    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Get model rankings by performance"""
        performance = self.performance_tracker.get_all_performance()
        
        rankings = []
        for model_id, metrics in performance.items():
            score = 1.0 / (metrics["recent_mae"] + 1e-8)  # Lower MAE = higher score
            rankings.append((model_id, score))
            
        return sorted(rankings, key=lambda x: x[1], reverse=True)
        
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics"""
        weights = self.ensemble_method.get_weights()
        performance = self.performance_tracker.get_all_performance()
        
        return {
            "n_models": len(self.models),
            "is_fitted": self.is_fitted,
            "prediction_count": self.prediction_count,
            "combination_method": self.config.combination_method,
            "weight_method": self.config.weight_method,
            "model_weights": dict(zip(self.models.keys(), weights)) if len(weights) > 0 else {},
            "model_performance": performance,
            "model_rankings": self.get_model_rankings()
        }
        
    def save_ensemble(self, path: Path) -> None:
        """Save ensemble to disk"""
        save_data = {
            "config": self.config.__dict__,
            "model_metadata": self.model_metadata,
            "ensemble_method_type": type(self.ensemble_method).__name__,
            "performance_history": self.performance_tracker.performance_history,
            "prediction_history": self.performance_tracker.prediction_history,
            "is_fitted": self.is_fitted,
            "prediction_count": self.prediction_count,
            "last_refit": self.last_refit
        }
        
        # Save ensemble metadata
        with open(path / "ensemble_metadata.json", "w") as f:
            json.dump(save_data, f, indent=2)
            
        # Save individual models
        models_dir = path / "models"
        models_dir.mkdir(exist_ok=True)
        
        for model_id, model in self.models.items():
            try:
                # Try to save model
                if hasattr(model, 'state_dict'):
                    # PyTorch model
                    import torch
                    torch.save(model.state_dict(), models_dir / f"{model_id}.pt")
                else:
                    # Pickle fallback
                    with open(models_dir / f"{model_id}.pkl", "wb") as f:
                        pickle.dump(model, f)
                        
            except Exception as e:
                self.logger.warning(f"Could not save model {model_id}: {e}")
                
        self.logger.info(f"Saved ensemble to {path}")
        
    @classmethod
    def load_ensemble(
        cls,
        path: Path,
        model_loader: Optional[callable] = None,
        logger: Optional[logging.Logger] = None
    ) -> "EnsemblePredictor":
        """Load ensemble from disk"""
        # Load metadata
        with open(path / "ensemble_metadata.json", "r") as f:
            save_data = json.load(f)
            
        # Create config
        config = EnsembleConfig(**save_data["config"])
        
        # Create ensemble
        ensemble = cls(config, logger)
        
        # Restore state
        ensemble.is_fitted = save_data["is_fitted"]
        ensemble.prediction_count = save_data["prediction_count"]
        ensemble.last_refit = save_data["last_refit"]
        ensemble.model_metadata = save_data["model_metadata"]
        
        # Restore performance tracking
        ensemble.performance_tracker.performance_history = save_data["performance_history"]
        ensemble.performance_tracker.prediction_history = save_data["prediction_history"]
        
        # Load models if loader provided
        if model_loader:
            models_dir = path / "models"
            for model_file in models_dir.glob("*"):
                model_id = model_file.stem
                try:
                    model = model_loader(model_file)
                    ensemble.models[model_id] = model
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not load model {model_id}: {e}")
                        
        if logger:
            logger.info(f"Loaded ensemble from {path}")
            
        return ensemble
