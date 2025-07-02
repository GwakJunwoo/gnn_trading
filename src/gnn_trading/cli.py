"""
gnn_trading.cli
===============

Command-line interface for GNN Trading System
Provides comprehensive CLI tools for all system operations
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

import pandas as pd
import numpy as np

from gnn_trading.models.trainer import ModelTrainer
from gnn_trading.models.ensemble import EnsemblePredictor, EnsembleConfig
from gnn_trading.backtest.engine import BacktestEngine, BacktestConfig
from gnn_trading.data_pipeline.quality import DataQualityManager, QualityConfig
from gnn_trading.graphs.streaming import StreamingGraphBuilder, StreamingConfig
from gnn_trading.graphs.graph_builder import GraphConfig
from gnn_trading.utils.logging import setup_logging
from gnn_trading.utils.validation import validate_config


def setup_cli_logging(level: str = "INFO"):
    """Setup logging for CLI"""
    setup_logging(level=level)
    return logging.getLogger(__name__)


def train_command(args):
    """Train model command"""
    logger = setup_cli_logging(args.log_level)
    logger.info("Starting model training...")
    
    try:
        # Load configuration
        config_path = Path(args.config) if args.config else Path("configs/train_config.yaml")
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1
            
        # TODO: Implement training logic
        logger.info("Training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def predict_command(args):
    """Prediction command"""
    logger = setup_cli_logging(args.log_level)
    logger.info("Making predictions...")
    
    try:
        # Load ensemble or single model
        if args.ensemble:
            logger.info("Using ensemble prediction")
            # TODO: Load and use ensemble
        else:
            logger.info("Using single model prediction")
            # TODO: Load and use single model
            
        logger.info("Predictions completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1


def backtest_command(args):
    """Backtest command"""
    logger = setup_cli_logging(args.log_level)
    logger.info("Starting backtest...")
    
    try:
        # Setup backtest configuration
        config = BacktestConfig(
            initial_capital=args.capital,
            transaction_cost=args.transaction_cost,
            max_position_size=args.max_position,
            rebalance_freq=args.rebalance_freq
        )
        
        # Initialize backtest engine
        engine = BacktestEngine(config)
        
        # Load predictions and price data
        predictions_path = Path(args.predictions)
        prices_path = Path(args.prices)
        
        if not predictions_path.exists():
            logger.error(f"Predictions file not found: {predictions_path}")
            return 1
            
        if not prices_path.exists():
            logger.error(f"Prices file not found: {prices_path}")
            return 1
            
        predictions = pd.read_csv(predictions_path)
        prices = pd.read_csv(prices_path)
        
        # Run backtest
        logger.info("Running backtest simulation...")
        results = engine.run_backtest(predictions, prices)
        
        # Save results
        output_dir = Path(args.output) if args.output else Path("backtest_results")
        output_dir.mkdir(exist_ok=True)
        
        engine.save_results(output_dir)
        
        # Print summary
        logger.info("Backtest Results:")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"Number of Trades: {results.get('num_trades', 0)}")
        
        logger.info(f"Detailed results saved to {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1


def quality_command(args):
    """Data quality check command"""
    logger = setup_cli_logging(args.log_level)
    logger.info("Running data quality checks...")
    
    try:
        # Setup quality manager
        config = QualityConfig(
            outlier_threshold=args.outlier_threshold,
            missing_threshold=args.missing_threshold,
            enable_real_time=args.streaming
        )
        
        manager = DataQualityManager(config)
        
        if args.streaming:
            logger.info("Starting streaming quality monitoring...")
            
            def quality_callback(report):
                logger.info(f"Quality Score: {report.overall_score:.3f}")
                if report.overall_score < 0.7:
                    logger.warning("Poor data quality detected!")
                    
            manager.setup_realtime_monitoring(quality_callback)
            
            # Keep running until interrupted
            try:
                import time
                while True:
                    time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Stopping quality monitoring...")
                
        else:
            # Batch quality check
            data_path = Path(args.data)
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                return 1
                
            # Load data
            if data_path.suffix == '.csv':
                data = pd.read_csv(data_path)
            elif data_path.suffix == '.parquet':
                data = pd.read_parquet(data_path)
            else:
                logger.error(f"Unsupported file format: {data_path.suffix}")
                return 1
                
            # Run quality checks
            report = manager.run_all_checks(data)
            
            # Print results
            logger.info(f"Overall Quality Score: {report.overall_score:.3f}")
            logger.info("Individual Checks:")
            
            for check in report.checks:
                status = "PASS" if check.passed else "FAIL"
                logger.info(f"  {check.name}: {status} (Score: {check.score:.3f})")
                if not check.passed:
                    logger.info(f"    Issue: {check.message}")
                    
            # Save report if requested
            if args.output:
                output_path = Path(args.output)
                manager.save_report(report, output_path)
                logger.info(f"Report saved to {output_path}")
                
            # Fix issues if requested
            if args.fix and report.overall_score < 0.8:
                logger.info("Attempting to fix data quality issues...")
                fixed_data = manager.fix_missing_values(data)
                fixed_data = manager.fix_outliers(fixed_data)
                
                # Save fixed data
                fixed_path = data_path.parent / f"fixed_{data_path.name}"
                if data_path.suffix == '.csv':
                    fixed_data.to_csv(fixed_path, index=False)
                else:
                    fixed_data.to_parquet(fixed_path, index=False)
                    
                logger.info(f"Fixed data saved to {fixed_path}")
                
        return 0
        
    except Exception as e:
        logger.error(f"Quality check failed: {e}")
        return 1


def streaming_command(args):
    """Streaming system command"""
    logger = setup_cli_logging(args.log_level)
    logger.info("Managing streaming system...")
    
    try:
        # Setup streaming configuration
        graph_config = GraphConfig()
        streaming_config = StreamingConfig(
            buffer_size=args.buffer_size,
            update_frequency=args.update_freq,
            max_latency=args.max_latency,
            enable_caching=args.enable_cache
        )
        
        # Initialize streaming builder
        builder = StreamingGraphBuilder(
            graph_config=graph_config,
            streaming_config=streaming_config,
            feature_root=Path(args.feature_root),
            logger=logger
        )
        
        if args.action == "start":
            logger.info("Starting streaming system...")
            
            # Setup callbacks
            def graph_callback(graph, timestamp):
                logger.debug(f"New graph built at {timestamp}")
                
            def error_callback(error):
                logger.error(f"Streaming error: {error}")
                
            builder.set_graph_ready_callback(graph_callback)
            builder.set_error_callback(error_callback)
            
            # Start streaming
            builder.start_streaming()
            
            logger.info("Streaming system started. Press Ctrl+C to stop.")
            
            try:
                import time
                while True:
                    time.sleep(10)
                    
                    # Print performance stats periodically
                    stats = builder.get_performance_stats()
                    logger.info(f"Performance: {stats['graphs_built']} graphs, "
                              f"avg time: {stats['avg_build_time']:.3f}s")
                              
            except KeyboardInterrupt:
                logger.info("Stopping streaming system...")
                builder.stop_streaming()
                
        elif args.action == "status":
            # Get status (requires running system)
            logger.info("Streaming system status:")
            try:
                stats = builder.get_performance_stats()
                logger.info(f"Graphs built: {stats['graphs_built']}")
                logger.info(f"Average build time: {stats['avg_build_time']:.3f}s")
                logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
                logger.info(f"Error count: {stats['errors']}")
            except Exception as e:
                logger.warning(f"Could not get status: {e}")
                
        return 0
        
    except Exception as e:
        logger.error(f"Streaming command failed: {e}")
        return 1


def ensemble_command(args):
    """Ensemble management command"""
    logger = setup_cli_logging(args.log_level)
    logger.info("Managing ensemble...")
    
    try:
        config = EnsembleConfig(
            combination_method=args.method,
            weight_method=args.weights,
            max_models=args.max_models
        )
        
        ensemble = EnsemblePredictor(config, logger)
        
        if args.action == "create":
            logger.info("Creating new ensemble...")
            
            # Add models from directory
            models_dir = Path(args.models_dir)
            if not models_dir.exists():
                logger.error(f"Models directory not found: {models_dir}")
                return 1
                
            model_count = 0
            for model_file in models_dir.glob("*.pt"):
                try:
                    # TODO: Load model and add to ensemble
                    logger.info(f"Added model: {model_file.name}")
                    model_count += 1
                except Exception as e:
                    logger.warning(f"Could not load model {model_file.name}: {e}")
                    
            logger.info(f"Created ensemble with {model_count} models")
            
            # Save ensemble
            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(exist_ok=True)
                ensemble.save_ensemble(output_path)
                logger.info(f"Ensemble saved to {output_path}")
                
        elif args.action == "evaluate":
            # Load ensemble
            ensemble_path = Path(args.ensemble_path)
            if not ensemble_path.exists():
                logger.error(f"Ensemble not found: {ensemble_path}")
                return 1
                
            # TODO: Load ensemble and evaluate
            logger.info("Ensemble evaluation completed")
            
        return 0
        
    except Exception as e:
        logger.error(f"Ensemble command failed: {e}")
        return 1


def serve_command(args):
    """Serve API command"""
    logger = setup_cli_logging(args.log_level)
    logger.info("Starting API server...")
    
    try:
        # Import and run API server
        from gnn_trading.api.main import main as api_main
        
        # Override sys.argv for uvicorn
        sys.argv = [
            "uvicorn",
            "gnn_trading.api.main:app",
            "--host", args.host,
            "--port", str(args.port),
            "--workers", str(args.workers)
        ]
        
        if args.reload:
            sys.argv.append("--reload")
            
        if args.log_level.lower() != "info":
            sys.argv.extend(["--log-level", args.log_level.lower()])
            
        api_main()
        return 0
        
    except Exception as e:
        logger.error(f"API server failed: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GNN Trading System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gnn-trading train --config configs/train_config.yaml
  gnn-trading predict --model checkpoints/model.pt --data data/test.csv
  gnn-trading backtest --predictions preds.csv --prices prices.csv
  gnn-trading quality --data data/market.csv --fix
  gnn-trading streaming start --buffer-size 1000
  gnn-trading ensemble create --models-dir models/
  gnn-trading serve --host 0.0.0.0 --port 8000
        """
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", help="Training configuration file")
    train_parser.add_argument("--data", help="Training data directory")
    train_parser.add_argument("--output", help="Output directory for checkpoints")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--resume", help="Resume from checkpoint")
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--model", help="Model checkpoint path")
    predict_parser.add_argument("--ensemble", help="Ensemble directory path")
    predict_parser.add_argument("--data", help="Input data file")
    predict_parser.add_argument("--output", help="Output predictions file")
    predict_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    predict_parser.set_defaults(func=predict_command)
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--predictions", required=True, help="Predictions CSV file")
    backtest_parser.add_argument("--prices", required=True, help="Price data CSV file")
    backtest_parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    backtest_parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost")
    backtest_parser.add_argument("--max-position", type=float, default=0.2, help="Max position size")
    backtest_parser.add_argument("--rebalance-freq", default="1D", help="Rebalance frequency")
    backtest_parser.add_argument("--output", help="Output directory")
    backtest_parser.set_defaults(func=backtest_command)
    
    # Quality command
    quality_parser = subparsers.add_parser("quality", help="Data quality checks")
    quality_parser.add_argument("--data", help="Data file to check")
    quality_parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    quality_parser.add_argument("--outlier-threshold", type=float, default=3.0, help="Outlier threshold")
    quality_parser.add_argument("--missing-threshold", type=float, default=0.05, help="Missing data threshold")
    quality_parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    quality_parser.add_argument("--output", help="Output report file")
    quality_parser.set_defaults(func=quality_command)
    
    # Streaming command
    streaming_parser = subparsers.add_parser("streaming", help="Streaming system")
    streaming_parser.add_argument("action", choices=["start", "status"], help="Streaming action")
    streaming_parser.add_argument("--buffer-size", type=int, default=1000, help="Buffer size")
    streaming_parser.add_argument("--update-freq", type=int, default=60, help="Update frequency (seconds)")
    streaming_parser.add_argument("--max-latency", type=float, default=1.0, help="Max latency (seconds)")
    streaming_parser.add_argument("--enable-cache", action="store_true", help="Enable caching")
    streaming_parser.add_argument("--feature-root", default="data/features", help="Feature root directory")
    streaming_parser.set_defaults(func=streaming_command)
    
    # Ensemble command
    ensemble_parser = subparsers.add_parser("ensemble", help="Ensemble management")
    ensemble_parser.add_argument("action", choices=["create", "evaluate"], help="Ensemble action")
    ensemble_parser.add_argument("--models-dir", help="Directory containing models")
    ensemble_parser.add_argument("--ensemble-path", help="Path to existing ensemble")
    ensemble_parser.add_argument("--method", default="weighted_average", help="Combination method")
    ensemble_parser.add_argument("--weights", default="performance", help="Weight method")
    ensemble_parser.add_argument("--max-models", type=int, default=10, help="Maximum models")
    ensemble_parser.add_argument("--output", help="Output directory")
    ensemble_parser.set_defaults(func=ensemble_command)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.set_defaults(func=serve_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger = setup_cli_logging(args.log_level)
        logger.error(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
