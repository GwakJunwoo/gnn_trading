
"""
gnn_trading.backtest.risk
-------------------------
Compute Sharpe, Max Drawdown, CAGR
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
