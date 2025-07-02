
"""
gnn_trading.graphs.graph_builder
=================================
Phase-2 : **Graph Construction Layer**

▣ 목적
──────────────────────────────────────────────────────────────
1. `feature_store/processed/` 의 시장·거시 Parquet 파일을 읽어
   **시점별 동적 그래프 스냅샷**(torch_geometric Data)으로 변환.
2. 노드 : 자산(symbol) + 거시(indicator) → feature 벡터
   엣지 : 상관계수 기반 또는 Granger Causality 기반
   - corr  |corr| > threshold  → undirected
   - granger p<alpha → directed (src→dst) edge
3. 결과는 `graph_snapshots/` 경로에 `<timestamp>.pt` 파일 저장.
   이후 `GraphDataset` 이 이 폴더를 메모리 로드한다.

▣ 설정 파일 (`configs/graph_config.yaml`)
──────────────────────────────────────────────────────────────
```yaml
snapshot_freq: '1D'           # 그래프 스냅샷 주기 (pandas offset)
corr_window:  60              # 수익률 기준 rolling window(분)
corr_threshold: 0.3
granger_lag: 5
edge_method: 'corr'           # 'corr' | 'granger'
symbols: ['KS200', 'KTB3F']
indicators: ['CPI', 'BaseRate']
```
▣ CLI
```bash
poetry run python -m gnn_trading.graphs.graph_builder \
    --feature_root feature_store/processed \
    --config configs/graph_config.yaml
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data
import yaml
import numpy as np
# Optional: statsmodels for granger
try:
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:  # pragma: no cover
    grangercausalitytests = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

@dataclass
class GraphConfig:
    snapshot_freq: str = "1D"
    corr_window: int = 60
    corr_threshold: float = 0.3
    granger_lag: int = 5
    edge_method: str = "corr"  # 'corr' | 'granger'
    symbols: List[str] = None
    indicators: List[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "GraphConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)

# -------------------------------------------------------------------------
# GraphSnapshotBuilder
# -------------------------------------------------------------------------

class GraphSnapshotBuilder:
    def __init__(self, cfg: GraphConfig, feature_root: Path, out_dir: Path):
        self.cfg = cfg
        self.feature_root = feature_root
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Load feature tables once
        self.mkt_df = pd.read_parquet(feature_root / f"market_1min.parquet")
        self.macro_df = pd.read_parquet(feature_root / "macro_daily.parquet")

        # Pre-compute node list
        self.node_list = cfg.symbols + cfg.indicators
        self.node_idx = {n: i for i, n in enumerate(self.node_list)}

    # --------------- public ---------------
    def build_all(self) -> None:
        logger.info("Building graph snapshots...")
        # Determine snapshot boundaries
        start_ts = self.mkt_df["datetime"].min().floor(self.cfg.snapshot_freq)
        end_ts = self.mkt_df["datetime"].max().ceil(self.cfg.snapshot_freq)
        ts_range = pd.date_range(start_ts, end_ts, freq=self.cfg.snapshot_freq)

        for ts in ts_range:
            try:
                data = self._build_single(ts)
                torch.save(data, self.out_dir / f"{ts.isoformat()}.pt")
            except ValueError as e:
                logger.warning("Skip %s : %s", ts, e)
        logger.info("Graph snapshots saved → %s", self.out_dir)

    # --------------- internal ---------------
    def _build_single(self, ts: pd.Timestamp) -> Data:
        """Build one snapshot ending at time ts (inclusive)."""
        window_start = ts - pd.Timedelta(minutes=self.cfg.corr_window)
        win_df = self.mkt_df[(self.mkt_df["datetime"] > window_start) & (self.mkt_df["datetime"] <= ts)]

        if win_df.empty:
            raise ValueError("No market data in window")

        # Pivot to returns matrix symbol x time
        ret = (
            win_df.pivot_table(index="datetime", columns="symbol", values="return")
            .fillna(0.0)
        )
        # Node features: latest close price normalized
        latest = (
            win_df.sort_values("datetime")
            .groupby("symbol")
            .tail(1)
            .set_index("symbol")["close"]
        )
        node_feat = torch.zeros((len(self.node_list), 1), dtype=torch.float32)
        for sym, price in latest.items():
            node_feat[self.node_idx[sym], 0] = price

        # Add macro node feature: latest value
        for ind in self.cfg.indicators:
            val = (
                self.macro_df[self.macro_df["indicator"] == ind]
                .sort_values("date")
                .tail(1)["value"]
                .values[0]
            )
            node_feat[self.node_idx[ind], 0] = val

        # Edge index & attr
        if self.cfg.edge_method == "corr":
            edge_index, edge_attr = self._edges_from_corr(ret)
        else:
            edge_index, edge_attr = self._edges_from_granger(ret)

        data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_attr, y=None, snapshot_ts=torch.tensor([ts.value], dtype=torch.int64))
        return data

    def _edges_from_corr(self, ret: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        corr = ret.corr()
        edges: List[Tuple[int, int, float]] = []
        for i in corr.columns:
            for j in corr.columns:
                if i >= j:
                    continue
                c = corr.loc[i, j]
                if abs(c) >= self.cfg.corr_threshold:
                    edges.append((self.node_idx[i], self.node_idx[j], c))
                    edges.append((self.node_idx[j], self.node_idx[i], c))
        if not edges:
            raise ValueError("No edges after corr threshold")
        idx = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
        attr = torch.tensor([e[2] for e in edges], dtype=torch.float32).unsqueeze(1)
        return idx, attr

    def _edges_from_granger(self, ret: pd.DataFrame):
        if grangercausalitytests is None:
            raise ImportError("statsmodels required for granger method")
        edges = []
        symbols = ret.columns.tolist()
        for i, src in enumerate(symbols):
            for j, dst in enumerate(symbols):
                if src == dst:
                    continue
                try:
                    test_res = grangercausalitytests(ret[[dst, src]].dropna(), maxlag=self.cfg.granger_lag, verbose=False)
                    pval = test_res[self.cfg.granger_lag][0]['ssr_ftest'][1]
                    if pval < 0.05:
                        edges.append((self.node_idx[src], self.node_idx[dst], pval))
                except Exception:
                    continue
        if not edges:
            raise ValueError("No edges from granger")
        idx = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
        attr = torch.tensor([e[2] for e in edges], dtype=torch.float32).unsqueeze(1)
        return idx, attr

# -------------------------------------------------------------------------
# GraphDataset
# -------------------------------------------------------------------------

from torch.utils.data import Dataset

class GraphDataset(Dataset):
    """Iterates over saved .pt snapshots"""

    def __init__(self, snapshot_dir: Path):
        self.files = sorted(snapshot_dir.glob("*.pt"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        return torch.load(self.files[idx])

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser("Graph Builder CLI")
    parser.add_argument("--feature_root", required=True, help="feature_store/processed path")
    parser.add_argument("--config", default="configs/graph_config.yaml")
    parser.add_argument("--out_dir", default="graph_snapshots")
    args = parser.parse_args()

    cfg = GraphConfig.from_yaml(Path(args.config))
    builder = GraphSnapshotBuilder(cfg, Path(args.feature_root), Path(args.out_dir))
    builder.build_all()
