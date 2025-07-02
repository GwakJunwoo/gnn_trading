"""
gnn_trading.data_pipeline.quality
==================================
Data Quality Management System

Advanced data validation, cleaning, and monitoring for production-grade trading systems.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
import yaml

logger = logging.getLogger(__name__)


class DataQualityConfig:
    """Configuration for data quality checks"""
    
    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        missing_threshold: float = 0.1,
        price_change_threshold: float = 0.2,
        volume_threshold: float = 5.0,
        correlation_threshold: float = 0.95,
        lookback_days: int = 30
    ):
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold
        self.price_change_threshold = price_change_threshold
        self.volume_threshold = volume_threshold
        self.correlation_threshold = correlation_threshold
        self.lookback_days = lookback_days


class DataQualityValidator:
    """Comprehensive data quality validation and cleaning"""
    
    def __init__(self, config: DataQualityConfig):
        self.config = config
        self.quality_reports = []
        
    def validate_market_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Comprehensive market data validation"""
        logger.info("Starting market data quality validation")
        
        report = {
            'timestamp': datetime.now(),
            'total_rows': len(df),
            'issues': [],
            'cleaned_rows': 0,
            'quality_score': 0.0
        }
        
        original_size = len(df)
        
        # 1. Check for missing values
        df, missing_report = self._handle_missing_values(df)
        report['issues'].append(missing_report)
        
        # 2. Detect and handle outliers
        df, outlier_report = self._handle_outliers(df)
        report['issues'].append(outlier_report)
        
        # 3. Validate price consistency
        df, price_report = self._validate_price_consistency(df)
        report['issues'].append(price_report)
        
        # 4. Check volume anomalies
        df, volume_report = self._validate_volume_data(df)
        report['issues'].append(volume_report)
        
        # 5. Temporal consistency checks
        df, temporal_report = self._validate_temporal_consistency(df)
        report['issues'].append(temporal_report)
        
        # 6. Cross-asset correlation checks
        correlation_report = self._validate_correlations(df)
        report['issues'].append(correlation_report)
        
        # Calculate quality metrics
        report['cleaned_rows'] = original_size - len(df)
        report['quality_score'] = self._calculate_quality_score(report)
        
        self.quality_reports.append(report)
        logger.info(f"Data quality validation complete. Score: {report['quality_score']:.2f}")
        
        return df, report
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values with intelligent imputation"""
        missing_info = df.isnull().sum()
        total_missing = missing_info.sum()
        
        if total_missing == 0:
            return df, {'type': 'missing_values', 'count': 0, 'action': 'none'}
        
        # Forward fill for OHLC data (common in financial time series)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                # Group by symbol and forward fill
                df[col] = df.groupby('symbol')[col].fillna(method='ffill')
                
                # Backward fill for remaining NaN at the beginning
                df[col] = df.groupby('symbol')[col].fillna(method='bfill')
        
        # Handle volume with median imputation
        if 'volume' in df.columns:
            df['volume'] = df.groupby('symbol')['volume'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Remove rows with excessive missing data
        missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
        df = df[missing_ratio <= self.config.missing_threshold]
        
        return df, {
            'type': 'missing_values',
            'count': total_missing,
            'action': 'imputed_and_filtered',
            'columns_affected': missing_info[missing_info > 0].to_dict()
        }
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle outliers using multiple methods"""
        outlier_counts = {}
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            outliers = self._detect_outliers(df, col)
            outlier_counts[col] = outliers.sum()
            
            if outliers.sum() > 0:
                if col in ['open', 'high', 'low', 'close']:
                    # For price data, cap outliers instead of removing
                    df.loc[outliers, col] = self._cap_outliers(df[col], outliers)
                else:
                    # For volume, we can be more aggressive
                    df = df[~outliers]
        
        return df, {
            'type': 'outliers',
            'count': sum(outlier_counts.values()),
            'action': 'capped_and_filtered',
            'columns_affected': outlier_counts
        }
    
    def _detect_outliers(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Detect outliers using configurable method"""
        if self.config.outlier_method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif self.config.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            return z_scores > self.config.outlier_threshold
        
        elif self.config.outlier_method == "modified_zscore":
            median = df[column].median()
            mad = np.median(np.abs(df[column] - median))
            modified_z_scores = 0.6745 * (df[column] - median) / mad
            return np.abs(modified_z_scores) > self.config.outlier_threshold
        
        else:
            raise ValueError(f"Unknown outlier method: {self.config.outlier_method}")
    
    def _cap_outliers(self, series: pd.Series, outliers: pd.Series) -> pd.Series:
        """Cap outliers at reasonable percentiles"""
        lower_cap = series.quantile(0.01)
        upper_cap = series.quantile(0.99)
        
        capped_series = series.copy()
        capped_series[outliers & (series < lower_cap)] = lower_cap
        capped_series[outliers & (series > upper_cap)] = upper_cap
        
        return capped_series
    
    def _validate_price_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate OHLC price relationships and detect price jumps"""
        issues = []
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.sum() > 0:
            issues.append(f"Invalid OHLC relationships: {invalid_ohlc.sum()} rows")
            df = df[~invalid_ohlc]
        
        # Detect price jumps
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('datetime')
            
            if len(symbol_data) < 2:
                continue
                
            price_changes = symbol_data['close'].pct_change().abs()
            large_jumps = price_changes > self.config.price_change_threshold
            
            if large_jumps.sum() > 0:
                issues.append(f"{symbol}: {large_jumps.sum()} large price jumps detected")
                # Log but don't remove - might be legitimate market moves
        
        return df, {
            'type': 'price_consistency',
            'count': len(issues),
            'action': 'validated_and_logged',
            'issues': issues
        }
    
    def _validate_volume_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate volume data for anomalies"""
        if 'volume' not in df.columns:
            return df, {'type': 'volume_validation', 'count': 0, 'action': 'skipped'}
        
        issues = []
        
        # Remove zero or negative volume
        invalid_volume = df['volume'] <= 0
        if invalid_volume.sum() > 0:
            issues.append(f"Invalid volume (<=0): {invalid_volume.sum()} rows")
            df = df[~invalid_volume]
        
        # Detect volume spikes
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('datetime')
            
            if len(symbol_data) < 10:  # Need sufficient data
                continue
                
            # Calculate rolling volume statistics
            rolling_median = symbol_data['volume'].rolling(window=20, min_periods=5).median()
            volume_ratio = symbol_data['volume'] / rolling_median
            
            volume_spikes = volume_ratio > self.config.volume_threshold
            if volume_spikes.sum() > 0:
                issues.append(f"{symbol}: {volume_spikes.sum()} volume spikes detected")
        
        return df, {
            'type': 'volume_validation',
            'count': len(issues),
            'action': 'validated_and_logged',
            'issues': issues
        }
    
    def _validate_temporal_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate temporal consistency and detect gaps"""
        if 'datetime' not in df.columns:
            return df, {'type': 'temporal_validation', 'count': 0, 'action': 'skipped'}
        
        issues = []
        
        # Check for duplicate timestamps per symbol
        duplicates = df.duplicated(subset=['symbol', 'datetime'])
        if duplicates.sum() > 0:
            issues.append(f"Duplicate timestamps: {duplicates.sum()} rows")
            df = df[~duplicates]
        
        # Detect temporal gaps
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('datetime')
            
            if len(symbol_data) < 2:
                continue
                
            time_diffs = symbol_data['datetime'].diff()
            expected_freq = pd.Timedelta(minutes=1)  # Assuming 1-minute data
            
            large_gaps = time_diffs > expected_freq * 10  # Gaps > 10 minutes
            if large_gaps.sum() > 0:
                issues.append(f"{symbol}: {large_gaps.sum()} temporal gaps detected")
        
        return df, {
            'type': 'temporal_validation',
            'count': len(issues),
            'action': 'validated_and_logged',
            'issues': issues
        }
    
    def _validate_correlations(self, df: pd.DataFrame) -> Dict:
        """Check for suspicious correlations between assets"""
        if len(df['symbol'].unique()) < 2:
            return {'type': 'correlation_validation', 'count': 0, 'action': 'insufficient_data'}
        
        # Pivot to get returns matrix
        returns_df = df.pivot_table(
            index='datetime', 
            columns='symbol', 
            values='close'
        ).pct_change().dropna()
        
        if len(returns_df) < 10:
            return {'type': 'correlation_validation', 'count': 0, 'action': 'insufficient_data'}
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Find highly correlated pairs (excluding self-correlation)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > self.config.correlation_threshold:
                    high_corr_pairs.append({
                        'asset1': corr_matrix.columns[i],
                        'asset2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'type': 'correlation_validation',
            'count': len(high_corr_pairs),
            'action': 'validated',
            'high_correlations': high_corr_pairs
        }
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        base_score = 100.0
        
        # Deduct points for each issue type
        for issue in report['issues']:
            if issue['count'] > 0:
                if issue['type'] == 'missing_values':
                    base_score -= min(issue['count'] * 0.01, 20)
                elif issue['type'] == 'outliers':
                    base_score -= min(issue['count'] * 0.005, 15)
                elif issue['type'] == 'price_consistency':
                    base_score -= min(issue['count'] * 2, 25)
                elif issue['type'] == 'volume_validation':
                    base_score -= min(issue['count'] * 0.5, 10)
                elif issue['type'] == 'temporal_validation':
                    base_score -= min(issue['count'] * 1, 15)
        
        return max(base_score, 0.0)
    
    def generate_quality_report(self, output_path: Optional[Path] = None) -> str:
        """Generate comprehensive quality report"""
        if not self.quality_reports:
            return "No quality reports available"
        
        latest_report = self.quality_reports[-1]
        
        report_text = f"""
Data Quality Report
==================
Generated: {latest_report['timestamp']}
Overall Quality Score: {latest_report['quality_score']:.2f}/100

Summary:
- Total rows processed: {latest_report['total_rows']:,}
- Rows cleaned/removed: {latest_report['cleaned_rows']:,}
- Data retention rate: {(1 - latest_report['cleaned_rows']/latest_report['total_rows'])*100:.1f}%

Detailed Issues:
"""
        
        for issue in latest_report['issues']:
            report_text += f"\n{issue['type'].upper()}:\n"
            report_text += f"  Count: {issue['count']}\n"
            report_text += f"  Action: {issue['action']}\n"
            
            if 'issues' in issue:
                for detail in issue['issues']:
                    report_text += f"  - {detail}\n"
            
            if 'columns_affected' in issue:
                for col, count in issue['columns_affected'].items():
                    report_text += f"  - {col}: {count} issues\n"
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text


class RealTimeDataMonitor:
    """Real-time data quality monitoring"""
    
    def __init__(self, config: DataQualityConfig):
        self.config = config
        self.validator = DataQualityValidator(config)
        self.alert_thresholds = {
            'quality_score': 80.0,
            'missing_ratio': 0.05,
            'outlier_ratio': 0.02
        }
        
    def monitor_stream(self, new_data: pd.DataFrame) -> Dict:
        """Monitor streaming data quality"""
        _, report = self.validator.validate_market_data(new_data)
        
        # Check alert conditions
        alerts = []
        
        if report['quality_score'] < self.alert_thresholds['quality_score']:
            alerts.append(f"Low quality score: {report['quality_score']:.2f}")
        
        # Calculate ratios for alerting
        total_rows = report['total_rows']
        if total_rows > 0:
            for issue in report['issues']:
                if issue['type'] == 'missing_values' and issue['count'] > 0:
                    missing_ratio = issue['count'] / total_rows
                    if missing_ratio > self.alert_thresholds['missing_ratio']:
                        alerts.append(f"High missing data ratio: {missing_ratio:.3f}")
                
                elif issue['type'] == 'outliers' and issue['count'] > 0:
                    outlier_ratio = issue['count'] / total_rows
                    if outlier_ratio > self.alert_thresholds['outlier_ratio']:
                        alerts.append(f"High outlier ratio: {outlier_ratio:.3f}")
        
        return {
            'quality_report': report,
            'alerts': alerts,
            'monitoring_timestamp': datetime.now()
        }


# CLI for data quality validation
def main():
    import argparse
    
    parser = argparse.ArgumentParser("Data Quality Validator")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", help="Output cleaned file")
    parser.add_argument("--report", help="Quality report output path")
    parser.add_argument("--config", help="Quality config YAML file")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = DataQualityConfig(**config_dict)
    else:
        config = DataQualityConfig()
    
    # Validate data
    validator = DataQualityValidator(config)
    df = pd.read_parquet(args.input)
    
    cleaned_df, report = validator.validate_market_data(df)
    
    # Save outputs
    if args.output:
        cleaned_df.to_parquet(args.output, index=False)
        print(f"Cleaned data saved to {args.output}")
    
    if args.report:
        report_text = validator.generate_quality_report(Path(args.report))
        print(f"Quality report saved to {args.report}")
    
    print(f"Quality Score: {report['quality_score']:.2f}/100")


if __name__ == "__main__":
    main()
