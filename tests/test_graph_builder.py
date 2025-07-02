"""
Tests for graph builder functionality
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from gnn_trading.graphs.graph_builder import GraphConfig, GraphSnapshotBuilder


@pytest.fixture
def sample_config():
    return GraphConfig(
        snapshot_freq="1D",
        corr_window=60,
        corr_threshold=0.3,
        symbols=["KS200", "KTB3F"],
        indicators=["CPI", "BaseRate"]
    )


@pytest.fixture
def sample_market_data():
    dates = pd.date_range("2025-01-01", periods=100, freq="1min")
    return pd.DataFrame({
        "datetime": dates,
        "symbol": ["KS200"] * 100,
        "close": 100 + torch.randn(100).numpy(),
        "return": torch.randn(100).numpy() * 0.01
    })


@pytest.fixture
def sample_macro_data():
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10),
        "indicator": ["CPI"] * 10,
        "value": 100 + torch.randn(10).numpy()
    })


def test_graph_config_creation(sample_config):
    assert sample_config.snapshot_freq == "1D"
    assert len(sample_config.symbols) == 2
    assert "KS200" in sample_config.symbols


@patch('pandas.read_parquet')
def test_graph_builder_initialization(mock_read_parquet, sample_config, sample_market_data, sample_macro_data, tmp_path):
    mock_read_parquet.side_effect = [sample_market_data, sample_macro_data]
    
    builder = GraphSnapshotBuilder(
        cfg=sample_config,
        feature_root=tmp_path,
        out_dir=tmp_path / "output"
    )
    
    assert builder.cfg == sample_config
    assert len(builder.node_list) == 4  # 2 symbols + 2 indicators


def test_edge_creation_correlation():
    # Test correlation-based edge creation
    ret_data = pd.DataFrame({
        'A': [0.01, -0.02, 0.015],
        'B': [0.02, -0.01, 0.01]
    })
    
    # Mock the builder and test edge creation logic
    # This would test the _edges_from_corr method
    pass


if __name__ == "__main__":
    pytest.main([__file__])
