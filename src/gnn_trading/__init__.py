"""
GNN Trading System
==================
A comprehensive Graph Neural Network-based trading system for Korean financial markets.

This package provides an end-to-end pipeline for:
- Data ingestion from KRX/ECOS APIs
- Feature engineering and graph construction
- TGAT model training and inference
- Backtesting and risk analysis
- Real-time prediction API
"""

__version__ = "1.0.0"
__author__ = "GNN Trading Team"

# Import main components for easier access
from gnn_trading.models.tgat import TGATModel
from gnn_trading.graphs.graph_builder import GraphSnapshotBuilder, GraphDataset
from gnn_trading.backtest.engine import BacktestEngine
from gnn_trading.utils import setup_logging, get_logger

__all__ = [
    "TGATModel",
    "GraphSnapshotBuilder", 
    "GraphDataset",
    "BacktestEngine",
    "setup_logging",
    "get_logger"
]
