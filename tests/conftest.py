"""
pytest configuration file for GNN Trading System
"""

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary test data directory"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
    symbols = ["KS200", "KQ150", "KTB3F"]
    
    data = []
    for symbol in symbols:
        base_price = 100 if symbol.startswith("K") else 1000
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.5)
        volumes = np.random.randint(1000, 10000, len(dates))
        
        df = pd.DataFrame({
            "datetime": dates,
            "symbol": symbol,
            "open": prices + np.random.randn(len(dates)) * 0.1,
            "high": prices + np.abs(np.random.randn(len(dates)) * 0.2),
            "low": prices - np.abs(np.random.randn(len(dates)) * 0.2),
            "close": prices,
            "volume": volumes,
            "return": np.random.randn(len(dates)) * 0.01
        })
        data.append(df)
    
    return pd.concat(data, ignore_index=True)


@pytest.fixture
def sample_macro_data():
    """Generate sample macro economic data for testing"""
    dates = pd.date_range("2023-01-01", periods=365, freq="1D")
    indicators = ["CPI", "BaseRate", "USD_KRW", "KOSPI_VOL"]
    
    data = []
    for indicator in indicators:
        base_value = 100 if indicator in ["CPI", "KOSPI_VOL"] else 1300 if indicator == "USD_KRW" else 3.5
        values = base_value + np.cumsum(np.random.randn(len(dates)) * 0.01)
        
        df = pd.DataFrame({
            "date": dates,
            "indicator": indicator,
            "value": values
        })
        data.append(df)
    
    return pd.concat(data, ignore_index=True)


@pytest.fixture
def sample_graph_data():
    """Generate sample graph data for testing"""
    from torch_geometric.data import Data
    
    # Create sample graph with 10 nodes
    num_nodes = 10
    node_features = torch.randn(num_nodes, 5)
    
    # Create edges (fully connected for simplicity)
    edge_indices = []
    edge_attrs = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edge_indices.extend([[i, j], [j, i]])  # Undirected
            edge_attrs.extend([torch.randn(3), torch.randn(3)])
    
    edge_index = torch.tensor(edge_indices).t().contiguous()
    edge_attr = torch.stack(edge_attrs)
    
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "data": {
            "symbols": ["KS200", "KQ150", "KTB3F"],
            "indicators": ["CPI", "BaseRate", "USD_KRW", "KOSPI_VOL"],
            "lookback": 60,
            "prediction_horizon": 1
        },
        "model": {
            "in_dim": 5,
            "hidden_dim": 32,
            "num_layers": 2,
            "heads": 2,
            "dropout": 0.1
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.001,
            "epochs": 10,
            "patience": 5
        },
        "graph": {
            "snapshot_freq": "1D",
            "corr_window": 60,
            "corr_threshold": 0.3
        }
    }


@pytest.fixture
def device():
    """Get device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pytest_configure(config):
    """Pytest configuration"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
