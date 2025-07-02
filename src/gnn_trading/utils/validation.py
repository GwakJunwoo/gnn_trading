"""
gnn_trading.utils.validation
============================
Configuration validation utilities
"""

from pathlib import Path
from typing import Dict, Any, List
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validate configuration files"""
    
    @staticmethod
    def validate_source_config(config_path: Path) -> bool:
        """Validate source configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['market', 'macro', 'store']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section: {section}")
                    return False
            
            # Validate market section
            market = config['market']
            if 'base_url' not in market or 'asset_list' not in market:
                logger.error("Market section missing required fields")
                return False
            
            # Validate macro section
            macro = config['macro']
            if 'base_url' not in macro or 'indicator_list' not in macro:
                logger.error("Macro section missing required fields")
                return False
            
            # Validate store section
            store = config['store']
            if 'root' not in store:
                logger.error("Store section missing required fields")
                return False
            
            logger.info("Source config validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Source config validation failed: {e}")
            return False
    
    @staticmethod
    def validate_graph_config(config_path: Path) -> bool:
        """Validate graph configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            required_fields = ['symbols', 'indicators', 'edge_method']
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate edge method
            if config['edge_method'] not in ['corr', 'granger']:
                logger.error("Invalid edge_method. Must be 'corr' or 'granger'")
                return False
            
            # Validate thresholds
            if 'corr_threshold' in config and not (0 <= config['corr_threshold'] <= 1):
                logger.error("corr_threshold must be between 0 and 1")
                return False
            
            logger.info("Graph config validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Graph config validation failed: {e}")
            return False
    
    @staticmethod
    def validate_all_configs(config_dir: Path) -> bool:
        """Validate all configuration files"""
        config_files = {
            'source_config.yaml': ConfigValidator.validate_source_config,
            'graph_config.yaml': ConfigValidator.validate_graph_config,
            # Add more validators as needed
        }
        
        all_valid = True
        for filename, validator in config_files.items():
            config_path = config_dir / filename
            if config_path.exists():
                if not validator(config_path):
                    all_valid = False
                    logger.error(f"Validation failed for {filename}")
            else:
                logger.warning(f"Config file not found: {filename}")
        
        return all_valid


class DataValidator:
    """Validate data integrity"""
    
    @staticmethod
    def validate_parquet_file(file_path: Path, required_columns: List[str]) -> bool:
        """Validate parquet file structure"""
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            
            # Check required columns
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                logger.error(f"Missing columns in {file_path}: {missing_cols}")
                return False
            
            # Check for empty dataframe
            if df.empty:
                logger.warning(f"Empty dataframe: {file_path}")
                return False
            
            logger.info(f"Data validation passed for {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed for {file_path}: {e}")
            return False
    
    @staticmethod
    def validate_feature_store(feature_store_path: Path) -> bool:
        """Validate feature store integrity"""
        try:
            # Check raw data
            raw_files = {
                'market_intraday_*.parquet': ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
                'macro_indicators.parquet': ['date', 'indicator', 'value']
            }
            
            # Check processed data
            processed_path = feature_store_path / 'processed'
            if processed_path.exists():
                processed_files = {
                    'market_*.parquet': ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'return'],
                    'macro_daily.parquet': ['date', 'indicator', 'value']
                }
                
                for pattern, required_cols in processed_files.items():
                    files = list(processed_path.glob(pattern))
                    for file_path in files:
                        if not DataValidator.validate_parquet_file(file_path, required_cols):
                            return False
            
            logger.info("Feature store validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Feature store validation failed: {e}")
            return False


def run_full_validation(project_root: Path) -> bool:
    """Run complete project validation"""
    logger.info("Starting full project validation...")
    
    # Validate configs
    config_dir = project_root / 'configs'
    if not ConfigValidator.validate_all_configs(config_dir):
        logger.error("Configuration validation failed")
        return False
    
    # Validate feature store
    feature_store = project_root / 'feature_store'
    if feature_store.exists():
        if not DataValidator.validate_feature_store(feature_store):
            logger.error("Feature store validation failed")
            return False
    
    logger.info("Full project validation passed")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Project Validation")
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    success = run_full_validation(args.project_root)
    exit(0 if success else 1)
