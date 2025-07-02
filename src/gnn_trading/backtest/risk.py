"""
gnn_trading.backtest.risk
-------------------------
Compute Sharpe, Max Drawdown, CAGR, Sortino Ratio, and Value at Risk
"""
import pandas as pd
import numpy as np

class RiskManager:
    @staticmethod
    def sharpe(pnl_series: pd.Series, freq: int = 252) -> float:
        if pnl_series.std() == 0:
            return 0.0
        return (pnl_series.mean() * freq) / (pnl_series.std() * np.sqrt(freq))

    @staticmethod
    def max_drawdown(cum_pnl: pd.Series) -> float:
        roll_max = cum_pnl.cummax()
        drawdown = cum_pnl - roll_max
        return drawdown.min()
    
    @staticmethod
    def cagr(cum_pnl: pd.Series, years: float) -> float:
        """Compound Annual Growth Rate"""
        if years <= 0:
            return 0.0
        return (cum_pnl.iloc[-1] / cum_pnl.iloc[0]) ** (1/years) - 1
    
    @staticmethod
    def sortino_ratio(pnl_series: pd.Series, freq: int = 252) -> float:
        """Sortino ratio using downside deviation"""
        downside_returns = pnl_series[pnl_series < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        return (pnl_series.mean() * freq) / (downside_std * np.sqrt(freq))
    
    @staticmethod
    def value_at_risk(pnl_series: pd.Series, confidence: float = 0.05) -> float:
        """Value at Risk at given confidence level"""
        return np.percentile(pnl_series, confidence * 100)
