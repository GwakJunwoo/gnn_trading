"""
gnn_trading.backtest.engine
===========================
Phase-4 : **Vectorised Backtest Engine**

기능
----
1. SignalGenerator(TGAT 모델) -> 일별/주간 예측 수익률 → 포지션 결정
2. 수수료, 슬리피지 적용 후 PnL 계산
3. RiskManager 로 Sharpe, MaxDD 등 메트릭 리턴

CLI
----
```bash
poetry run python -m gnn_trading.backtest.engine --start 2025-07-01 --end 2025-07-31
```

설정은 `configs/backtest_config.yaml` 참고.
"""

from __future__ import annotations
import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from gnn_trading.models.tgat import TGATModel
from gnn_trading.graphs.graph_builder import GraphDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class BacktestConfig:
    snapshot_dir: Path = Path("graph_snapshots")
    ckpt_path: Path = Path("checkpoints/tgat.ckpt")
    symbol: str = "KS200"
    fee_rate: float = 0.0005
    slip: float = 0.0002

    @classmethod
    def from_yaml(cls, path: Path) -> "BacktestConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(
            snapshot_dir=Path(cfg.get("snapshot_dir", "graph_snapshots")),
            ckpt_path=Path(cfg.get("ckpt_path", "checkpoints/tgat.ckpt")),
            symbol=cfg.get("symbol", "KS200"),
            fee_rate=cfg.get("fee_rate", 0.0005),
            slip=cfg.get("slip", 0.0002),
        )


class BacktestEngine:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.dataset = GraphDataset(cfg.snapshot_dir)
        self.model = TGATModel()
        state = torch.load(cfg.ckpt_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"] if isinstance(state, dict) else state)
        self.model.eval()
        
        # Performance tracking
        self.trades = []
        self.positions_history = []

    def run(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Run backtest with optional date filtering"""
        pnl_rows = []
        prev_price = None
        position = 0  # 1 long, -1 short, 0 flat
        
        for i, snap in enumerate(self.dataset):
            snap_time = pd.to_datetime(snap.snapshot_ts.item(), unit="ns")
            
            # Apply date filtering if specified
            if start_date and snap_time < start_date:
                continue
            if end_date and snap_time > end_date:
                break
                
            with torch.no_grad():
                pred = self.model([snap]).squeeze()  # [N]
            
            # Choose symbol idx (improve this mapping)
            sym_idx = self._get_symbol_index(snap)
            signal = pred[sym_idx].item()
            price = snap.x[sym_idx, 0].item()
            
            # Calculate PnL
            if prev_price is not None:
                ret = (price - prev_price) / prev_price
                pnl = position * ret - abs(position) * (self.cfg.fee_rate + self.cfg.slip)
            else:
                pnl = 0.0
            
            # Position sizing based on signal strength
            new_position = self._calculate_position(signal)
            
            # Track trade if position changed
            if new_position != position:
                self.trades.append({
                    'timestamp': snap_time,
                    'old_position': position,
                    'new_position': new_position,
                    'price': price,
                    'signal': signal
                })
            
            position = new_position
            self.positions_history.append({
                'timestamp': snap_time,
                'position': position,
                'price': price,
                'signal': signal
            })
            
            pnl_rows.append({
                "ts": snap_time, 
                "pnl": pnl, 
                "pos": position,
                "signal": signal,
                "price": price
            })
            prev_price = price
            
        df = pd.DataFrame(pnl_rows)
        df["cum_pnl"] = df["pnl"].cumsum()
        return df
    
    def _get_symbol_index(self, snap: torch.Tensor) -> int:
        """Get symbol index from snapshot - improve this logic"""
        # For now, assume target symbol is the last asset node
        # TODO: Create proper symbol-to-index mapping
        return snap.x.size(0) - 1
    
    def _calculate_position(self, signal: float, threshold: float = 0.001) -> int:
        """Calculate position based on signal strength"""
        if signal > threshold:
            return 1  # Long
        elif signal < -threshold:
            return -1  # Short
        else:
            return 0  # Flat
    
    def get_trade_statistics(self) -> dict:
        """Calculate detailed trade statistics"""
        if not self.trades:
            return {}
            
        trades_df = pd.DataFrame(self.trades)
        
        # Count trades by type
        long_trades = trades_df[trades_df['new_position'] == 1]
        short_trades = trades_df[trades_df['new_position'] == -1]
        exit_trades = trades_df[trades_df['new_position'] == 0]
        
        return {
            'total_trades': len(self.trades),
            'long_entries': len(long_trades),
            'short_entries': len(short_trades),
            'exits': len(exit_trades),
            'avg_holding_period': self._calculate_avg_holding_period(),
            'position_changes_per_day': len(self.trades) / max(1, (trades_df['timestamp'].max() - trades_df['timestamp'].min()).days)
        }
    
    def _calculate_avg_holding_period(self) -> float:
        """Calculate average holding period in days"""
        if len(self.trades) < 2:
            return 0.0
            
        holding_periods = []
        for i in range(len(self.trades) - 1):
            period = (self.trades[i+1]['timestamp'] - self.trades[i]['timestamp']).total_seconds() / (24 * 3600)
            holding_periods.append(period)
        
        return sum(holding_periods) / len(holding_periods) if holding_periods else 0.0

if __name__ == "__main__":
    import argparse, yaml, logging
    import pandas as pd
    from gnn_trading.backtest.risk import RiskManager

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser("Backtest CLI")
    parser.add_argument("--config", default="configs/backtest_config.yaml")
    args = parser.parse_args()

    cfg = BacktestConfig.from_yaml(Path(args.config))
    engine = BacktestEngine(cfg)
    pnl_df = engine.run()
    pnl_df.to_csv("backtest_result.csv", index=False)
    sharpe = RiskManager.sharpe(pnl_df['pnl'])
    mdd = RiskManager.max_drawdown(pnl_df['cum_pnl'])
    print("Backtest completed  Sharpe:", round(sharpe, 3), " MaxDD:", round(mdd, 3))
