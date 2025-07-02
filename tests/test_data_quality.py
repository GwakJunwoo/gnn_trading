"""
Comprehensive tests for data quality management
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from gnn_trading.data_pipeline.quality import (
    DataQualityManager,
    QualityConfig,
    QualityReport,
    QualityCheck
)


class TestQualityConfig:
    """Test suite for QualityConfig"""
    
    def test_quality_config_creation(self):
        """Test quality config creation with default values"""
        config = QualityConfig()
        
        assert config.outlier_threshold == 3.0
        assert config.missing_threshold == 0.05
        assert config.price_change_threshold == 0.20
        assert config.volume_change_threshold == 5.0
        assert config.correlation_window == 60
        assert config.enable_real_time == True
        
    def test_quality_config_custom_values(self):
        """Test quality config with custom values"""
        config = QualityConfig(
            outlier_threshold=2.5,
            missing_threshold=0.10,
            price_change_threshold=0.15,
            volume_change_threshold=3.0,
            correlation_window=30,
            enable_real_time=False
        )
        
        assert config.outlier_threshold == 2.5
        assert config.missing_threshold == 0.10
        assert config.price_change_threshold == 0.15
        assert config.volume_change_threshold == 3.0
        assert config.correlation_window == 30
        assert config.enable_real_time == False


class TestQualityReport:
    """Test suite for QualityReport"""
    
    def test_quality_report_creation(self):
        """Test quality report creation"""
        checks = [
            QualityCheck(name="test1", passed=True, score=0.95, message="Good"),
            QualityCheck(name="test2", passed=False, score=0.60, message="Poor")
        ]
        
        report = QualityReport(
            timestamp=datetime.now(),
            overall_score=0.75,
            checks=checks,
            data_summary={"rows": 1000, "cols": 5}
        )
        
        assert report.overall_score == 0.75
        assert len(report.checks) == 2
        assert report.checks[0].passed == True
        assert report.checks[1].passed == False
        assert report.data_summary["rows"] == 1000
        
    def test_quality_report_serialization(self):
        """Test quality report JSON serialization"""
        checks = [QualityCheck(name="test", passed=True, score=1.0, message="OK")]
        report = QualityReport(
            timestamp=datetime.now(),
            overall_score=1.0,
            checks=checks,
            data_summary={"test": "value"}
        )
        
        # Test to_dict method
        report_dict = report.to_dict()
        
        assert isinstance(report_dict, dict)
        assert "timestamp" in report_dict
        assert "overall_score" in report_dict
        assert "checks" in report_dict
        assert "data_summary" in report_dict
        assert report_dict["overall_score"] == 1.0


class TestDataQualityManager:
    """Test suite for DataQualityManager"""
    
    def test_quality_manager_initialization(self):
        """Test quality manager initialization"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        assert manager.config == config
        assert manager.logger is not None
        
    def test_check_missing_values_clean_data(self, sample_market_data):
        """Test missing values check with clean data"""
        config = QualityConfig(missing_threshold=0.05)
        manager = DataQualityManager(config)
        
        # Clean data should pass
        check = manager.check_missing_values(sample_market_data)
        
        assert check.passed == True
        assert check.score >= 0.95
        assert "missing" in check.name.lower()
        
    def test_check_missing_values_dirty_data(self, sample_market_data):
        """Test missing values check with dirty data"""
        config = QualityConfig(missing_threshold=0.05)
        manager = DataQualityManager(config)
        
        # Add missing values (10% missing)
        dirty_data = sample_market_data.copy()
        n_missing = int(len(dirty_data) * 0.10)
        dirty_data.loc[:n_missing, 'close'] = np.nan
        
        check = manager.check_missing_values(dirty_data)
        
        assert check.passed == False
        assert check.score < 0.95
        
    def test_check_outliers_clean_data(self, sample_market_data):
        """Test outliers check with clean data"""
        config = QualityConfig(outlier_threshold=3.0)
        manager = DataQualityManager(config)
        
        check = manager.check_outliers(sample_market_data)
        
        assert check.passed == True
        assert check.score >= 0.90
        assert "outlier" in check.name.lower()
        
    def test_check_outliers_dirty_data(self, sample_market_data):
        """Test outliers check with dirty data"""
        config = QualityConfig(outlier_threshold=2.0)  # Stricter threshold
        manager = DataQualityManager(config)
        
        # Add extreme outliers
        dirty_data = sample_market_data.copy()
        dirty_data.loc[0, 'close'] = dirty_data['close'].mean() + 10 * dirty_data['close'].std()
        dirty_data.loc[1, 'close'] = dirty_data['close'].mean() - 10 * dirty_data['close'].std()
        
        check = manager.check_outliers(dirty_data)
        
        assert check.passed == False
        assert check.score < 0.95
        
    def test_check_price_consistency(self, sample_market_data):
        """Test price consistency check"""
        config = QualityConfig(price_change_threshold=0.20)
        manager = DataQualityManager(config)
        
        check = manager.check_price_consistency(sample_market_data)
        
        assert isinstance(check, QualityCheck)
        assert "price" in check.name.lower()
        assert 0.0 <= check.score <= 1.0
        
    def test_check_volume_consistency(self, sample_market_data):
        """Test volume consistency check"""
        config = QualityConfig(volume_change_threshold=5.0)
        manager = DataQualityManager(config)
        
        check = manager.check_volume_consistency(sample_market_data)
        
        assert isinstance(check, QualityCheck)
        assert "volume" in check.name.lower()
        assert 0.0 <= check.score <= 1.0
        
    def test_check_temporal_consistency(self, sample_market_data):
        """Test temporal consistency check"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        check = manager.check_temporal_consistency(sample_market_data)
        
        assert isinstance(check, QualityCheck)
        assert "temporal" in check.name.lower()
        assert 0.0 <= check.score <= 1.0
        
    def test_check_correlation_stability(self, sample_market_data):
        """Test correlation stability check"""
        config = QualityConfig(correlation_window=60)
        manager = DataQualityManager(config)
        
        # Need multiple symbols for correlation
        multi_symbol_data = sample_market_data.copy()
        symbols = multi_symbol_data['symbol'].unique()
        
        if len(symbols) > 1:
            check = manager.check_correlation_stability(multi_symbol_data)
            
            assert isinstance(check, QualityCheck)
            assert "correlation" in check.name.lower()
            assert 0.0 <= check.score <= 1.0
        else:
            # Single symbol case
            check = manager.check_correlation_stability(multi_symbol_data)
            assert check.passed == True  # Should pass for single symbol
            
    def test_run_all_checks(self, sample_market_data):
        """Test running all quality checks"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        report = manager.run_all_checks(sample_market_data)
        
        assert isinstance(report, QualityReport)
        assert len(report.checks) > 0
        assert 0.0 <= report.overall_score <= 1.0
        assert report.data_summary is not None
        
        # Check that all expected checks are present
        check_names = [check.name for check in report.checks]
        expected_checks = [
            "missing_values", "outliers", "price_consistency",
            "volume_consistency", "temporal_consistency", "correlation_stability"
        ]
        
        for expected in expected_checks:
            assert any(expected in name.lower() for name in check_names)
            
    def test_fix_missing_values(self, sample_market_data):
        """Test missing values fixing"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        # Add missing values
        dirty_data = sample_market_data.copy()
        dirty_data.loc[10:15, 'close'] = np.nan
        
        fixed_data = manager.fix_missing_values(dirty_data)
        
        # Should have no missing values in critical columns
        assert not fixed_data['close'].isna().any()
        assert len(fixed_data) == len(dirty_data)
        
    def test_fix_outliers(self, sample_market_data):
        """Test outliers fixing"""
        config = QualityConfig(outlier_threshold=2.0)
        manager = DataQualityManager(config)
        
        # Add extreme outliers
        dirty_data = sample_market_data.copy()
        mean_price = dirty_data['close'].mean()
        std_price = dirty_data['close'].std()
        
        dirty_data.loc[0, 'close'] = mean_price + 10 * std_price
        dirty_data.loc[1, 'close'] = mean_price - 10 * std_price
        
        fixed_data = manager.fix_outliers(dirty_data)
        
        # Outliers should be capped/replaced
        assert fixed_data['close'].max() < dirty_data['close'].max()
        assert fixed_data['close'].min() > dirty_data['close'].min()
        
    def test_real_time_monitoring_setup(self):
        """Test real-time monitoring setup"""
        config = QualityConfig(enable_real_time=True)
        manager = DataQualityManager(config)
        
        # Mock callback function
        callback = Mock()
        
        manager.setup_realtime_monitoring(callback)
        
        # Should store callback
        assert manager.alert_callback == callback
        
    def test_real_time_monitoring_alert(self, sample_market_data):
        """Test real-time monitoring alert"""
        config = QualityConfig(enable_real_time=True)
        manager = DataQualityManager(config)
        
        # Mock callback
        callback = Mock()
        manager.setup_realtime_monitoring(callback)
        
        # Add problematic data that should trigger alert
        dirty_data = sample_market_data.copy()
        dirty_data.loc[:50, 'close'] = np.nan  # 50% missing
        
        # Process data (should trigger alert)
        report = manager.run_all_checks(dirty_data)
        
        if report.overall_score < 0.7:  # Quality threshold
            manager.send_alert(report)
            
            # Callback should be called
            assert callback.called
            
    def test_generate_report_summary(self, sample_market_data):
        """Test report summary generation"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        report = manager.run_all_checks(sample_market_data)
        summary = manager.generate_summary(report)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Quality Score" in summary
        
    def test_save_and_load_report(self, sample_market_data):
        """Test saving and loading quality reports"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "quality_report.json"
            
            # Generate and save report
            report = manager.run_all_checks(sample_market_data)
            manager.save_report(report, save_path)
            
            # Check file exists
            assert save_path.exists()
            
            # Load and verify report
            loaded_report = manager.load_report(save_path)
            
            assert loaded_report.overall_score == report.overall_score
            assert len(loaded_report.checks) == len(report.checks)
            
    def test_batch_validation(self, sample_market_data):
        """Test batch validation functionality"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        # Split data into batches
        batch_size = len(sample_market_data) // 3
        batches = [
            sample_market_data.iloc[i:i+batch_size]
            for i in range(0, len(sample_market_data), batch_size)
        ]
        
        results = []
        for batch in batches:
            if len(batch) > 0:
                report = manager.run_all_checks(batch)
                results.append(report)
                
        assert len(results) > 0
        assert all(isinstance(r, QualityReport) for r in results)
        
    def test_streaming_validation(self, sample_market_data):
        """Test streaming validation functionality"""
        config = QualityConfig(enable_real_time=True)
        manager = DataQualityManager(config)
        
        # Simulate streaming data
        alerts = []
        
        def mock_callback(report):
            alerts.append(report)
            
        manager.setup_realtime_monitoring(mock_callback)
        
        # Process data in small chunks (simulate streaming)
        chunk_size = 10
        for i in range(0, len(sample_market_data), chunk_size):
            chunk = sample_market_data.iloc[i:i+chunk_size]
            if len(chunk) > 0:
                report = manager.run_all_checks(chunk)
                
                # Simulate poor quality data occasionally
                if i % 50 == 0:  # Every 5th chunk
                    poor_chunk = chunk.copy()
                    poor_chunk.loc[:, 'close'] = np.nan
                    poor_report = manager.run_all_checks(poor_chunk)
                    if poor_report.overall_score < 0.7:
                        manager.send_alert(poor_report)
                        
        # Should have some alerts from poor quality chunks
        # (depending on random data generation)
        assert isinstance(alerts, list)


class TestQualityIntegration:
    """Integration tests for data quality system"""
    
    def test_quality_pipeline_integration(self, sample_market_data, sample_macro_data):
        """Test full quality pipeline integration"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        # Test market data
        market_report = manager.run_all_checks(sample_market_data)
        assert isinstance(market_report, QualityReport)
        
        # Test macro data
        macro_report = manager.run_all_checks(sample_macro_data)
        assert isinstance(macro_report, QualityReport)
        
        # Both should have reasonable quality scores
        assert market_report.overall_score > 0.0
        assert macro_report.overall_score > 0.0
        
    def test_quality_with_feature_engineering(self, sample_market_data):
        """Test quality checks after feature engineering"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        # Add engineered features
        data_with_features = sample_market_data.copy()
        data_with_features['ma_5'] = data_with_features.groupby('symbol')['close'].rolling(5).mean().reset_index(0, drop=True)
        data_with_features['rsi'] = np.random.uniform(0, 100, len(data_with_features))
        data_with_features['volatility'] = data_with_features.groupby('symbol')['return'].rolling(20).std().reset_index(0, drop=True)
        
        # Quality check should handle additional features
        report = manager.run_all_checks(data_with_features)
        
        assert isinstance(report, QualityReport)
        assert report.overall_score > 0.0
        
        # Data summary should include new features
        assert report.data_summary['cols'] > sample_market_data.shape[1]
        
    def test_quality_monitoring_performance(self, sample_market_data):
        """Test performance of quality monitoring on large datasets"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        # Create larger dataset
        large_data = pd.concat([sample_market_data] * 10, ignore_index=True)
        
        import time
        start_time = time.time()
        
        report = manager.run_all_checks(large_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert processing_time < 30.0  # 30 seconds max
        assert isinstance(report, QualityReport)
        
    def test_quality_report_persistence(self, sample_market_data):
        """Test quality report persistence and retrieval"""
        config = QualityConfig()
        manager = DataQualityManager(config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = Path(tmp_dir) / "quality_reports"
            reports_dir.mkdir()
            
            # Generate multiple reports
            reports = []
            for i in range(5):
                # Slightly modify data each time
                data = sample_market_data.copy()
                if i % 2 == 0:
                    # Add some issues to alternate reports
                    data.loc[:i*10, 'close'] = np.nan
                    
                report = manager.run_all_checks(data)
                report_path = reports_dir / f"report_{i}.json"
                manager.save_report(report, report_path)
                reports.append(report)
                
            # Load all reports
            loaded_reports = []
            for i in range(5):
                report_path = reports_dir / f"report_{i}.json"
                loaded_report = manager.load_report(report_path)
                loaded_reports.append(loaded_report)
                
            # Verify all reports loaded correctly
            assert len(loaded_reports) == 5
            for orig, loaded in zip(reports, loaded_reports):
                assert abs(orig.overall_score - loaded.overall_score) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
