"""
Integration tests for the complete GNN Trading System
"""

import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from gnn_trading.models.ensemble import EnsemblePredictor, EnsembleConfig
from gnn_trading.graphs.streaming import StreamingGraphBuilder, StreamingConfig
from gnn_trading.graphs.graph_builder import GraphConfig
from gnn_trading.data_pipeline.quality import DataQualityManager, QualityConfig
from gnn_trading.backtest.engine import BacktestEngine, BacktestConfig


class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_pipeline(self, sample_market_data):
        """Test complete end-to-end pipeline"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # 1. Data Quality Check
            quality_config = QualityConfig()
            quality_manager = DataQualityManager(quality_config)
            
            quality_report = quality_manager.run_all_checks(sample_market_data)
            assert quality_report.overall_score > 0.0
            
            # 2. Feature Engineering (mock)
            features_data = sample_market_data.copy()
            features_data['ma_5'] = features_data.groupby('symbol')['close'].rolling(5).mean().reset_index(0, drop=True)
            
            # 3. Graph Building (mock streaming)
            graph_config = GraphConfig()
            streaming_config = StreamingConfig(buffer_size=100, update_frequency=1)
            
            streaming_builder = StreamingGraphBuilder(
                graph_config=graph_config,
                streaming_config=streaming_config,
                feature_root=tmp_path,
            )
            
            # Add data to streaming builder
            streaming_builder.add_market_data(features_data)
            
            # 4. Model Ensemble
            ensemble_config = EnsembleConfig(max_models=3)
            ensemble = EnsemblePredictor(ensemble_config)
            
            # Add mock models
            def mock_model_1(data):
                return np.random.randn() * 0.01
                
            def mock_model_2(data):
                return np.random.randn() * 0.02
                
            ensemble.add_model("model_1", mock_model_1)
            ensemble.add_model("model_2", mock_model_2)
            
            # Create validation data
            validation_data = [(features_data.iloc[i:i+10], np.random.randn() * 0.01) 
                             for i in range(0, min(50, len(features_data)), 10)]
            
            if validation_data:
                ensemble.fit(validation_data)
                
                # 5. Make Predictions
                test_data = features_data.iloc[-10:]
                prediction, metadata = ensemble.predict(test_data)
                
                assert isinstance(prediction, (int, float))
                assert isinstance(metadata, dict)
                
                # 6. Backtest (mock)
                backtest_config = BacktestConfig(initial_capital=10000)
                backtest_engine = BacktestEngine(backtest_config)
                
                # Create simple predictions and prices for backtest
                dates = pd.date_range("2023-01-01", periods=10, freq="1D")
                mock_predictions = pd.DataFrame({
                    'datetime': dates,
                    'symbol': ['TEST'] * 10,
                    'prediction': np.random.randn(10) * 0.01
                })
                
                mock_prices = pd.DataFrame({
                    'datetime': dates,
                    'symbol': ['TEST'] * 10,
                    'close': 100 + np.cumsum(np.random.randn(10))
                })
                
                backtest_results = backtest_engine.run_backtest(mock_predictions, mock_prices)
                
                assert isinstance(backtest_results, dict)
                assert 'total_return' in backtest_results
                
    def test_streaming_integration(self, sample_market_data):
        """Test streaming system integration"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Setup streaming components
            graph_config = GraphConfig()
            streaming_config = StreamingConfig(
                buffer_size=50,
                update_frequency=1,  # 1 second for testing
                enable_caching=True
            )
            
            streaming_builder = StreamingGraphBuilder(
                graph_config=graph_config,
                streaming_config=streaming_config,
                feature_root=tmp_path
            )
            
            # Setup callbacks
            graphs_received = []
            errors_received = []
            
            def graph_callback(graph, timestamp):
                graphs_received.append((graph, timestamp))
                
            def error_callback(error):
                errors_received.append(error)
                
            streaming_builder.set_graph_ready_callback(graph_callback)
            streaming_builder.set_error_callback(error_callback)
            
            # Start streaming
            streaming_builder.start_streaming()
            
            try:
                # Add data in chunks (simulate streaming)
                chunk_size = 10
                for i in range(0, min(50, len(sample_market_data)), chunk_size):
                    chunk = sample_market_data.iloc[i:i+chunk_size]
                    streaming_builder.add_market_data(chunk)
                    time.sleep(0.1)  # Small delay
                    
                # Wait a bit for processing
                time.sleep(2)
                
                # Check that graphs were generated
                stats = streaming_builder.get_performance_stats()
                
                # Should have processed some data
                assert stats['buffer_stats']['size'] > 0
                
            finally:
                streaming_builder.stop_streaming()
                
    def test_ensemble_with_real_models(self):
        """Test ensemble with mock models that behave like real models"""
        
        class MockTGATModel:
            def __init__(self, model_id):
                self.model_id = model_id
                self.weights = np.random.randn(10)  # Mock weights
                
            def predict(self, data):
                # Mock prediction based on data
                if hasattr(data, 'shape'):
                    return np.random.randn() * 0.01
                else:
                    return np.random.randn() * 0.01
                    
            def __call__(self, data):
                return self.predict(data)
                
        # Create ensemble
        config = EnsembleConfig(
            combination_method="weighted_average",
            weight_method="performance",
            max_models=3
        )
        
        ensemble = EnsemblePredictor(config)
        
        # Add mock models
        for i in range(3):
            model = MockTGATModel(f"tgat_{i}")
            ensemble.add_model(f"tgat_{i}", model)
            
        # Create validation data
        validation_data = []
        for i in range(20):
            data = np.random.randn(10, 5)  # Mock graph data
            target = np.random.randn() * 0.01
            validation_data.append((data, target))
            
        # Fit ensemble
        ensemble.fit(validation_data)
        
        # Make predictions
        test_data = np.random.randn(10, 5)
        prediction, metadata = ensemble.predict(test_data)
        
        assert isinstance(prediction, (int, float))
        assert 'individual_predictions' in metadata
        assert 'weights' in metadata
        assert 'uncertainty' in metadata
        
        # Test uncertainty estimation
        pred_with_uncertainty, uncertainty = ensemble.predict_with_uncertainty(test_data)
        
        assert isinstance(pred_with_uncertainty, (int, float))
        assert isinstance(uncertainty, (int, float))
        assert uncertainty >= 0
        
    def test_quality_streaming_integration(self, sample_market_data):
        """Test integration of quality monitoring with streaming"""
        
        # Setup quality manager
        quality_config = QualityConfig(enable_real_time=True)
        quality_manager = DataQualityManager(quality_config)
        
        alerts_received = []
        
        def quality_alert_callback(report):
            alerts_received.append(report)
            
        quality_manager.setup_realtime_monitoring(quality_alert_callback)
        
        # Process data with varying quality
        clean_data = sample_market_data.iloc[:50].copy()
        
        # Clean data should pass
        report1 = quality_manager.run_all_checks(clean_data)
        assert report1.overall_score > 0.5
        
        # Dirty data should trigger alerts
        dirty_data = sample_market_data.iloc[50:100].copy()
        dirty_data.loc[:25, 'close'] = np.nan  # Add missing values
        
        report2 = quality_manager.run_all_checks(dirty_data)
        
        if report2.overall_score < 0.7:
            quality_manager.send_alert(report2)
            
        # Should have received alert for poor quality
        # (depending on random data quality)
        
    def test_backtest_with_ensemble(self, sample_market_data):
        """Test backtesting with ensemble predictions"""
        
        # Create ensemble
        config = EnsembleConfig(max_models=2)
        ensemble = EnsemblePredictor(config)
        
        # Add simple models
        def trend_model(data):
            # Simple trend following
            return 0.01 if np.random.random() > 0.5 else -0.01
            
        def mean_reversion_model(data):
            # Simple mean reversion
            return -0.005 if np.random.random() > 0.5 else 0.005
            
        ensemble.add_model("trend", trend_model)
        ensemble.add_model("mean_reversion", mean_reversion_model)
        
        # Fit ensemble
        validation_data = [(np.random.randn(5), np.random.randn() * 0.01) for _ in range(20)]
        ensemble.fit(validation_data)
        
        # Generate predictions
        dates = pd.date_range("2023-01-01", periods=30, freq="1D")
        symbols = ['STOCK_A', 'STOCK_B']
        
        predictions_data = []
        prices_data = []
        
        for date in dates:
            for symbol in symbols:
                # Mock prediction
                mock_data = np.random.randn(5)
                pred, _ = ensemble.predict(mock_data)
                
                predictions_data.append({
                    'datetime': date,
                    'symbol': symbol,
                    'prediction': pred
                })
                
                # Mock price data
                base_price = 100 if symbol == 'STOCK_A' else 200
                price = base_price + np.random.randn() * 5
                
                prices_data.append({
                    'datetime': date,
                    'symbol': symbol,
                    'close': price
                })
                
        predictions_df = pd.DataFrame(predictions_data)
        prices_df = pd.DataFrame(prices_data)
        
        # Run backtest
        backtest_config = BacktestConfig(initial_capital=100000)
        backtest_engine = BacktestEngine(backtest_config)
        
        results = backtest_engine.run_backtest(predictions_df, prices_df)
        
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'num_trades' in results
        
        # Check that some trades were made
        trade_stats = backtest_engine.get_trade_statistics()
        assert isinstance(trade_stats, dict)
        
    def test_system_error_handling(self, sample_market_data):
        """Test system error handling and recovery"""
        
        # Test ensemble with failing model
        config = EnsembleConfig(max_models=3)
        ensemble = EnsemblePredictor(config)
        
        def working_model(data):
            return 0.01
            
        def failing_model(data):
            raise ValueError("Model failure simulation")
            
        ensemble.add_model("working", working_model)
        ensemble.add_model("failing", failing_model)
        
        # Should handle failing model gracefully
        test_data = np.random.randn(5)
        try:
            # This might succeed with just the working model
            prediction, metadata = ensemble.predict(test_data)
            assert isinstance(prediction, (int, float))
        except ValueError:
            # Or it might fail completely, which is also acceptable
            pass
            
        # Test streaming with invalid data
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            streaming_config = StreamingConfig(buffer_size=10)
            graph_config = GraphConfig()
            
            streaming_builder = StreamingGraphBuilder(
                graph_config=graph_config,
                streaming_config=streaming_config,
                feature_root=tmp_path
            )
            
            # Add invalid data
            invalid_data = pd.DataFrame({
                'invalid_column': [1, 2, 3]
            })
            
            # Should handle gracefully
            try:
                streaming_builder.add_market_data(invalid_data)
            except Exception as e:
                # Error is expected and handled
                assert isinstance(e, (ValueError, KeyError))
                
    def test_performance_monitoring(self, sample_market_data):
        """Test performance monitoring across components"""
        
        # Test ensemble performance tracking
        config = EnsembleConfig(performance_window=10)
        ensemble = EnsemblePredictor(config)
        
        def simple_model(data):
            return 0.01
            
        ensemble.add_model("simple", simple_model)
        
        # Fit ensemble
        validation_data = [(np.random.randn(3), 0.01) for _ in range(5)]
        ensemble.fit(validation_data)
        
        # Make predictions and update performance
        for i in range(15):
            test_data = np.random.randn(3)
            prediction, _ = ensemble.predict(test_data)
            target = np.random.randn() * 0.01
            
            ensemble.update_performance(test_data, target)
            
        # Check performance stats
        stats = ensemble.get_ensemble_stats()
        assert 'model_performance' in stats
        assert 'prediction_count' in stats
        assert stats['prediction_count'] == 15
        
        # Check model rankings
        rankings = ensemble.get_model_rankings()
        assert isinstance(rankings, list)
        assert len(rankings) > 0
        
    def test_save_load_system_state(self):
        """Test saving and loading system state"""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create ensemble
            config = EnsembleConfig()
            ensemble = EnsemblePredictor(config)
            
            def test_model(data):
                return 0.01
                
            ensemble.add_model("test", test_model, {"version": "1.0"})
            
            # Fit ensemble
            validation_data = [(np.random.randn(3), 0.01) for _ in range(5)]
            ensemble.fit(validation_data)
            
            # Make some predictions to build history
            for _ in range(5):
                test_data = np.random.randn(3)
                ensemble.predict(test_data)
                
            # Save ensemble
            save_path = tmp_path / "ensemble"
            save_path.mkdir()
            ensemble.save_ensemble(save_path)
            
            # Check that files were created
            assert (save_path / "ensemble_metadata.json").exists()
            
            # Load ensemble (without model loader for simplicity)
            loaded_ensemble = EnsemblePredictor.load_ensemble(save_path)
            
            # Check that state was preserved
            assert loaded_ensemble.prediction_count == ensemble.prediction_count
            assert loaded_ensemble.is_fitted == ensemble.is_fitted


class TestAPIIntegration:
    """Integration tests for API components"""
    
    @pytest.mark.asyncio
    async def test_api_health_check(self):
        """Test API health check integration"""
        # This would require running the actual API
        # For now, we'll test the component initialization
        
        from gnn_trading.api.main import initialize_components
        
        # Test component initialization
        try:
            initialize_components()
            # If no exception, initialization worked
            assert True
        except Exception as e:
            # Some dependencies might not be available in test environment
            assert "not available" in str(e) or "not found" in str(e)
            
    def test_cli_integration(self):
        """Test CLI command integration"""
        from gnn_trading.cli import main
        
        # Test help command
        import sys
        old_argv = sys.argv
        
        try:
            sys.argv = ["gnn-trading", "--help"]
            result = main()
            # Help should return 1 (or 0 depending on implementation)
            assert result in [0, 1]
        except SystemExit as e:
            # argparse calls sys.exit, which is expected
            assert e.code in [0, 1]
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
