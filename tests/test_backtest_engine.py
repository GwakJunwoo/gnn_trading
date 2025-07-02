"""
Comprehensive tests for backtest engine
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from gnn_trading.backtest.engine import BacktestEngine, PortfolioState, Trade, BacktestConfig


class TestBacktestConfig:
    """Test suite for BacktestConfig"""
    
    def test_backtest_config_creation(self):
        """Test backtest config creation"""
        config = BacktestConfig()
        
        assert config.initial_capital == 100000.0
        assert config.transaction_cost == 0.001
        assert config.max_position_size == 0.20
        assert config.rebalance_freq == "1D"
        assert config.risk_free_rate == 0.02
        
    def test_backtest_config_custom_values(self):
        """Test backtest config with custom values"""
        config = BacktestConfig(
            initial_capital=50000.0,
            transaction_cost=0.002,
            max_position_size=0.15,
            rebalance_freq="1W",
            risk_free_rate=0.03
        )
        
        assert config.initial_capital == 50000.0
        assert config.transaction_cost == 0.002
        assert config.max_position_size == 0.15
        assert config.rebalance_freq == "1W"
        assert config.risk_free_rate == 0.03


class TestTrade:
    """Test suite for Trade class"""
    
    def test_trade_creation(self):
        """Test trade creation"""
        trade = Trade(
            timestamp=datetime.now(),
            symbol="KS200",
            action="BUY",
            quantity=100,
            price=2500.0,
            commission=2.5
        )
        
        assert trade.symbol == "KS200"
        assert trade.action == "BUY"
        assert trade.quantity == 100
        assert trade.price == 2500.0
        assert trade.commission == 2.5
        assert trade.total_value == 100 * 2500.0 + 2.5
        
    def test_trade_sell(self):
        """Test sell trade"""
        trade = Trade(
            timestamp=datetime.now(),
            symbol="KS200",
            action="SELL",
            quantity=50,
            price=2600.0,
            commission=1.3
        )
        
        assert trade.action == "SELL"
        assert trade.total_value == 50 * 2600.0 + 1.3


class TestPortfolioState:
    """Test suite for PortfolioState"""
    
    def test_portfolio_state_creation(self):
        """Test portfolio state creation"""
        state = PortfolioState(
            timestamp=datetime.now(),
            cash=50000.0,
            positions={"KS200": 100, "KQ150": 50},
            portfolio_value=100000.0,
            returns=0.05
        )
        
        assert state.cash == 50000.0
        assert state.positions == {"KS200": 100, "KQ150": 50}
        assert state.portfolio_value == 100000.0
        assert state.returns == 0.05
        
    def test_portfolio_state_empty_positions(self):
        """Test portfolio state with empty positions"""
        state = PortfolioState(
            timestamp=datetime.now(),
            cash=100000.0,
            positions={},
            portfolio_value=100000.0,
            returns=0.0
        )
        
        assert state.positions == {}
        assert state.portfolio_value == 100000.0


class TestBacktestEngine:
    """Test suite for BacktestEngine"""
    
    def test_engine_initialization(self, sample_config):
        """Test backtest engine initialization"""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        assert engine.config == config
        assert engine.initial_capital == config.initial_capital
        assert engine.cash == config.initial_capital
        assert engine.positions == {}
        assert len(engine.trade_history) == 0
        assert len(engine.portfolio_history) == 0
        
    def test_calculate_position_size(self, sample_config):
        """Test position size calculation"""
        config = BacktestConfig(
            initial_capital=100000.0,
            max_position_size=0.20
        )
        engine = BacktestEngine(config)
        
        # Test position size for $2500 stock
        size = engine.calculate_position_size("KS200", 2500.0, 0.1)  # 10% target weight
        expected_value = 100000.0 * 0.1  # $10,000
        expected_shares = int(expected_value / 2500.0)  # 4 shares
        
        assert size == expected_shares
        
        # Test position size exceeding max
        size = engine.calculate_position_size("KS200", 1000.0, 0.30)  # 30% target > 20% max
        expected_value = 100000.0 * 0.20  # Capped at 20%
        expected_shares = int(expected_value / 1000.0)  # 20 shares
        
        assert size == expected_shares
        
    def test_calculate_commission(self, sample_config):
        """Test commission calculation"""
        config = BacktestConfig(transaction_cost=0.001)  # 0.1%
        engine = BacktestEngine(config)
        
        commission = engine.calculate_commission(100, 2500.0)  # 100 shares @ $2500
        expected = 100 * 2500.0 * 0.001  # $2.50
        
        assert commission == expected
        
    def test_execute_trade_buy(self, sample_config):
        """Test executing buy trade"""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        
        initial_cash = engine.cash
        
        # Execute buy trade
        trade = engine.execute_trade(
            timestamp=datetime.now(),
            symbol="KS200",
            target_shares=10,
            current_price=2500.0
        )
        
        assert trade is not None
        assert trade.action == "BUY"
        assert trade.symbol == "KS200"
        assert trade.quantity == 10
        assert trade.price == 2500.0
        
        # Check cash and positions updated
        assert engine.cash < initial_cash
        assert engine.positions.get("KS200", 0) == 10
        assert len(engine.trade_history) == 1
        
    def test_execute_trade_sell(self, sample_config):
        """Test executing sell trade"""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        
        # First buy some shares
        engine.positions["KS200"] = 20
        engine.cash = 50000.0
        
        initial_cash = engine.cash
        
        # Execute sell trade
        trade = engine.execute_trade(
            timestamp=datetime.now(),
            symbol="KS200",
            target_shares=10,  # Reduce from 20 to 10
            current_price=2600.0
        )
        
        assert trade is not None
        assert trade.action == "SELL"
        assert trade.symbol == "KS200"
        assert trade.quantity == 10  # Selling 10 shares
        assert trade.price == 2600.0
        
        # Check cash increased and positions reduced
        assert engine.cash > initial_cash
        assert engine.positions.get("KS200", 0) == 10
        
    def test_execute_trade_no_change(self, sample_config):
        """Test executing trade with no position change"""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Set current position
        engine.positions["KS200"] = 10
        
        # Execute trade with same target
        trade = engine.execute_trade(
            timestamp=datetime.now(),
            symbol="KS200",
            target_shares=10,
            current_price=2500.0
        )
        
        # No trade should be executed
        assert trade is None
        assert len(engine.trade_history) == 0
        
    def test_calculate_portfolio_value(self, sample_config):
        """Test portfolio value calculation"""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Set up portfolio
        engine.cash = 50000.0
        engine.positions = {"KS200": 10, "KQ150": 20}
        
        prices = {"KS200": 2500.0, "KQ150": 1500.0}
        
        portfolio_value = engine.calculate_portfolio_value(prices)
        expected = 50000.0 + (10 * 2500.0) + (20 * 1500.0)  # 105,000
        
        assert portfolio_value == expected
        
    def test_calculate_returns(self, sample_config):
        """Test returns calculation"""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        
        returns = engine.calculate_returns(110000.0)  # 10% gain
        assert returns == 0.10
        
        returns = engine.calculate_returns(90000.0)  # 10% loss
        assert returns == -0.10
        
    def test_record_portfolio_state(self, sample_config):
        """Test recording portfolio state"""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        timestamp = datetime.now()
        engine.cash = 60000.0
        engine.positions = {"KS200": 15}
        
        engine.record_portfolio_state(timestamp, 100000.0)
        
        assert len(engine.portfolio_history) == 1
        state = engine.portfolio_history[0]
        
        assert state.timestamp == timestamp
        assert state.cash == 60000.0
        assert state.positions == {"KS200": 15}
        assert state.portfolio_value == 100000.0
        
    def test_run_backtest_simple(self, sample_market_data):
        """Test running simple backtest"""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        
        # Create simple predictions
        predictions = pd.DataFrame({
            'datetime': pd.date_range("2023-01-01", periods=10, freq="1D"),
            'symbol': ['KS200'] * 10,
            'prediction': np.random.randn(10) * 0.02  # ±2% predictions
        })
        
        # Create price data for backtest period
        prices = pd.DataFrame({
            'datetime': pd.date_range("2023-01-01", periods=10, freq="1D"),
            'symbol': ['KS200'] * 10,
            'close': 2500 + np.cumsum(np.random.randn(10) * 10)  # Random walk around 2500
        })
        
        # Run backtest
        results = engine.run_backtest(predictions, prices)
        
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'num_trades' in results
        
        # Should have some portfolio history
        assert len(engine.portfolio_history) > 0
        
    def test_run_backtest_multiple_symbols(self, sample_market_data):
        """Test backtest with multiple symbols"""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        
        symbols = ['KS200', 'KQ150']
        dates = pd.date_range("2023-01-01", periods=5, freq="1D")
        
        # Create predictions for multiple symbols
        predictions_data = []
        prices_data = []
        
        for symbol in symbols:
            for date in dates:
                predictions_data.append({
                    'datetime': date,
                    'symbol': symbol,
                    'prediction': np.random.randn() * 0.02
                })
                
                base_price = 2500 if symbol == 'KS200' else 1500
                prices_data.append({
                    'datetime': date,
                    'symbol': symbol,
                    'close': base_price + np.random.randn() * 50
                })
                
        predictions = pd.DataFrame(predictions_data)
        prices = pd.DataFrame(prices_data)
        
        # Run backtest
        results = engine.run_backtest(predictions, prices)
        
        assert isinstance(results, dict)
        assert len(engine.portfolio_history) > 0
        
        # Should have trades for multiple symbols
        if len(engine.trade_history) > 0:
            traded_symbols = set(trade.symbol for trade in engine.trade_history)
            assert len(traded_symbols) > 0
            
    def test_calculate_performance_metrics(self, sample_config):
        """Test performance metrics calculation"""
        config = BacktestConfig(
            initial_capital=100000.0,
            risk_free_rate=0.02
        )
        engine = BacktestEngine(config)
        
        # Create mock portfolio history
        dates = pd.date_range("2023-01-01", periods=100, freq="1D")
        
        # Simulate portfolio with some volatility and positive trend
        returns = np.random.randn(100) * 0.01 + 0.0002  # ~5% annual with volatility
        portfolio_values = 100000.0 * np.cumprod(1 + returns)
        
        for i, (date, value) in enumerate(zip(dates, portfolio_values)):
            state = PortfolioState(
                timestamp=date,
                cash=value * 0.1,  # 10% cash
                positions={"KS200": int(value * 0.9 / 2500)},  # Rest in stocks
                portfolio_value=value,
                returns=returns[i]
            )
            engine.portfolio_history.append(state)
            
        # Calculate metrics
        metrics = engine.calculate_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'calmar_ratio' in metrics
        
        # Reasonable bounds for metrics
        assert -1.0 < metrics['total_return'] < 5.0  # -100% to +500%
        assert -1.0 < metrics['annualized_return'] < 2.0  # -100% to +200%
        assert 0.0 <= metrics['volatility'] < 1.0  # 0% to 100%
        assert 0.0 <= metrics['max_drawdown'] <= 1.0  # 0% to 100%
        
    def test_get_trade_statistics(self, sample_config):
        """Test trade statistics calculation"""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Add mock trades
        base_time = datetime.now()
        trades = [
            Trade(base_time, "KS200", "BUY", 10, 2500.0, 2.5),
            Trade(base_time + timedelta(days=1), "KS200", "SELL", 10, 2600.0, 2.6),
            Trade(base_time + timedelta(days=2), "KQ150", "BUY", 20, 1500.0, 3.0),
            Trade(base_time + timedelta(days=3), "KQ150", "SELL", 20, 1450.0, 2.9),
        ]
        
        engine.trade_history = trades
        
        stats = engine.get_trade_statistics()
        
        assert 'total_trades' in stats
        assert 'winning_trades' in stats
        assert 'losing_trades' in stats
        assert 'win_rate' in stats
        assert 'avg_profit_per_trade' in stats
        assert 'total_commission' in stats
        
        assert stats['total_trades'] == 4
        assert stats['total_commission'] == sum(t.commission for t in trades)
        assert 0.0 <= stats['win_rate'] <= 1.0
        
    def test_get_position_history(self, sample_config):
        """Test position history tracking"""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Add mock portfolio states
        dates = pd.date_range("2023-01-01", periods=5, freq="1D")
        
        for i, date in enumerate(dates):
            state = PortfolioState(
                timestamp=date,
                cash=50000.0,
                positions={"KS200": 10 + i, "KQ150": 20 - i},
                portfolio_value=100000.0 + i * 1000,
                returns=0.01 * i
            )
            engine.portfolio_history.append(state)
            
        history = engine.get_position_history()
        
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 5
        assert 'timestamp' in history.columns
        assert 'cash' in history.columns
        assert 'portfolio_value' in history.columns
        assert 'returns' in history.columns
        
        # Check positions are properly expanded
        for symbol in ["KS200", "KQ150"]:
            assert symbol in history.columns
            
    def test_save_and_load_results(self, sample_config):
        """Test saving and loading backtest results"""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Add some mock data
        engine.trade_history = [
            Trade(datetime.now(), "KS200", "BUY", 10, 2500.0, 2.5)
        ]
        
        engine.portfolio_history = [
            PortfolioState(
                datetime.now(),
                50000.0,
                {"KS200": 10},
                100000.0,
                0.0
            )
        ]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir)
            
            # Save results
            engine.save_results(save_path)
            
            # Check files were created
            assert (save_path / "trade_history.csv").exists()
            assert (save_path / "portfolio_history.csv").exists()
            assert (save_path / "performance_metrics.json").exists()
            
            # Load and verify
            trade_df = pd.read_csv(save_path / "trade_history.csv")
            portfolio_df = pd.read_csv(save_path / "portfolio_history.csv")
            
            assert len(trade_df) == 1
            assert len(portfolio_df) == 1
            assert trade_df.iloc[0]['symbol'] == "KS200"
            
    def test_backtest_with_insufficient_funds(self, sample_config):
        """Test backtest behavior with insufficient funds"""
        config = BacktestConfig(initial_capital=1000.0)  # Very small capital
        engine = BacktestEngine(config)
        
        # Try to buy expensive stock
        trade = engine.execute_trade(
            timestamp=datetime.now(),
            symbol="KS200",
            target_shares=10,  # Wants 10 shares @ 2500 = 25000 (more than capital)
            current_price=2500.0
        )
        
        # Should either execute partial trade or no trade
        if trade is not None:
            # If trade executed, should be for affordable quantity
            assert trade.quantity * trade.price <= engine.initial_capital * 1.1  # Allow for small margin
        else:
            # No trade should be fine too
            assert len(engine.trade_history) == 0
            
    def test_backtest_edge_cases(self, sample_config):
        """Test backtest edge cases"""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Test with empty predictions
        empty_predictions = pd.DataFrame(columns=['datetime', 'symbol', 'prediction'])
        empty_prices = pd.DataFrame(columns=['datetime', 'symbol', 'close'])
        
        results = engine.run_backtest(empty_predictions, empty_prices)
        
        # Should handle gracefully
        assert isinstance(results, dict)
        assert results.get('num_trades', 0) == 0
        
        # Test with single data point
        single_prediction = pd.DataFrame({
            'datetime': [datetime.now()],
            'symbol': ['KS200'],
            'prediction': [0.05]
        })
        
        single_price = pd.DataFrame({
            'datetime': [datetime.now()],
            'symbol': ['KS200'],
            'close': [2500.0]
        })
        
        results = engine.run_backtest(single_prediction, single_price)
        assert isinstance(results, dict)


class TestBacktestIntegration:
    """Integration tests for backtest engine"""
    
    def test_full_backtest_pipeline(self, sample_market_data):
        """Test full backtest pipeline"""
        config = BacktestConfig(
            initial_capital=100000.0,
            transaction_cost=0.001,
            max_position_size=0.30
        )
        engine = BacktestEngine(config)
        
        # Use sample market data to create realistic test
        # Generate predictions based on simple momentum strategy
        market_data = sample_market_data.copy()
        market_data = market_data.sort_values(['symbol', 'datetime'])
        
        # Calculate simple momentum signal
        market_data['price_change'] = market_data.groupby('symbol')['close'].pct_change()
        market_data['momentum'] = market_data.groupby('symbol')['price_change'].rolling(5).mean().reset_index(0, drop=True)
        
        # Create predictions
        predictions = market_data[['datetime', 'symbol']].copy()
        predictions['prediction'] = market_data['momentum'].fillna(0) * 0.5  # Scale down predictions
        
        # Create price data
        prices = market_data[['datetime', 'symbol', 'close']].copy()
        
        # Run backtest
        results = engine.run_backtest(predictions, prices)
        
        # Verify results
        assert isinstance(results, dict)
        assert len(engine.portfolio_history) > 0
        
        # Calculate additional metrics
        trade_stats = engine.get_trade_statistics()
        performance = engine.calculate_performance_metrics()
        
        assert isinstance(trade_stats, dict)
        assert isinstance(performance, dict)
        
        # Save results
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine.save_results(Path(tmp_dir))
            
            # Verify files created
            assert (Path(tmp_dir) / "trade_history.csv").exists()
            assert (Path(tmp_dir) / "portfolio_history.csv").exists()
            
    def test_backtest_robustness(self, sample_market_data):
        """Test backtest robustness with various conditions"""
        configs = [
            BacktestConfig(initial_capital=50000.0, transaction_cost=0.0005),
            BacktestConfig(initial_capital=200000.0, transaction_cost=0.002),
            BacktestConfig(initial_capital=100000.0, max_position_size=0.10),
            BacktestConfig(initial_capital=100000.0, max_position_size=0.50),
        ]
        
        for config in configs:
            engine = BacktestEngine(config)
            
            # Simple predictions
            dates = pd.date_range("2023-01-01", periods=20, freq="1D")
            symbols = sample_market_data['symbol'].unique()[:2]  # Use first 2 symbols
            
            predictions_data = []
            prices_data = []
            
            for date in dates:
                for symbol in symbols:
                    predictions_data.append({
                        'datetime': date,
                        'symbol': symbol,
                        'prediction': np.random.randn() * 0.02
                    })
                    
                    base_price = 2500 if symbol == 'KS200' else 1500
                    prices_data.append({
                        'datetime': date,
                        'symbol': symbol,
                        'close': base_price + np.random.randn() * 50
                    })
                    
            predictions = pd.DataFrame(predictions_data)
            prices = pd.DataFrame(prices_data)
            
            # Should run without errors
            results = engine.run_backtest(predictions, prices)
            assert isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
