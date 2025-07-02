
"""
gnn_trading.data_pipeline.feature_builder
=========================================
Phase‑1 : **Feature Engineering Layer**

■ 목적
──────────────────────────────────────────────────────────
1. *Raw Parquet* (`feature_store/*`)에 저장된 인트라데이 시장·거시 데이터를
   **모델 학습에 바로 쓰일 Feature 세트**로 변환한다.
2. 리샘플 주기(1min·5min·EOD 등)와 거시 Forward‑fill 방식(또는 MIDAS)을
   설정 파일(`configs/feature_config.yaml`)로 자유롭게 제어한다.
3. 변환 결과는 `feature_store/processed/` 디렉터리에 저장하며,
   **모델·TGAT Dataset**이 바로 이 경로를 참조한다.

■ 사용 방법
──────────────────────────────────────────────────────────
0. Phase‑0 인제스트가 선행되어 `feature_store/`에
   *market_intraday_YYYYMM.parquet*, *macro_indicators.parquet*
   가 존재해야 한다.
1. 설정 파일 예시 (`configs/feature_config.yaml`)
   ```yaml
   feature_root: "feature_store/processed"
   ohlc_freq: "1min"        # pandas offset alias
   macro_fill_method: "ffill"
   ```
2. CLI 실행
   ```bash
   poetry run python -m gnn_trading.data_pipeline.feature_builder \
       --mkt_table market_intraday_202506 \
       --macro_table macro_indicators \
       --freq 1min
   ```
   ▶ 결과:
   * `feature_store/processed/market_1min.parquet`
   * `feature_store/processed/macro_daily.parquet`
3. 향후 MIDAS 변환:
   macro_fill_method: "midas" 로 지정하고, `fill_macro_daily`
   내부 TODO 섹션에 MIDAS 변환을 구현해주면 된다.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------------
# Config dataclass
# -------------------------------------------------------------------------

@dataclass
class BuilderConfig:
    """Feature Builder 설정"""

    feature_root: Path = Path("feature_store/processed")
    ohlc_freq: str = "1min"            # 리샘플 주기
    macro_fill_method: str = "ffill"   # 'ffill' | 'midas'

    @classmethod
    def from_yaml(cls, path: Path) -> "BuilderConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            feature_root=Path(data["feature_root"]),
            ohlc_freq=data.get("ohlc_freq", "1min"),
            macro_fill_method=data.get("macro_fill_method", "ffill"),
        )

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """인트라데이 OHLCV → 주기별 캔들 + 수익률"""
    df = df.set_index("datetime").sort_index()
    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    res = (
        df.groupby("symbol")
        .resample(freq)
        .agg(ohlc_dict)
        .dropna(subset=["open"])
        .reset_index()
    )
    res["return"] = res.groupby("symbol")["close"].pct_change()
    return res


def fill_macro_daily(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """월·분기 거시 → 일단위 변환 (ffill / midas)"""
    df = df.sort_values("date").set_index("date")
    if method == "ffill":
        df = (
            df.groupby("indicator")
            .apply(lambda x: x.resample("1D").ffill())
            .reset_index(level=0, drop=True)
            .reset_index()
        )
        return df
    elif method == "midas":
        # TODO: 실제 Almon‑Midas 가중 변환 구현
        raise NotImplementedError("MIDAS 변환은 구현 필요")
    else:
        raise ValueError(f"Unknown macro_fill_method: {method}")

# -------------------------------------------------------------------------
# FeatureBuilder 클래스
# -------------------------------------------------------------------------

class FeatureBuilder:
    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        self.cfg.feature_root.mkdir(parents=True, exist_ok=True)

    # ---------- MARKET ----------
    def build_market(self, mkt_df: pd.DataFrame) -> Path:
        res = resample_ohlc(mkt_df, self.cfg.ohlc_freq)
        out = self.cfg.feature_root / f"market_{self.cfg.ohlc_freq}.parquet"
        res.to_parquet(out, index=False)
        logger.info("✔ Market features saved → %s  (%d rows)", out, len(res))
        return out

    # ---------- MACRO ----------
    def build_macro(self, macro_df: pd.DataFrame) -> Path:
        daily = fill_macro_daily(macro_df, self.cfg.macro_fill_method)
        out = self.cfg.feature_root / "macro_daily.parquet"
        daily.to_parquet(out, index=False)
        logger.info("✔ Macro features saved → %s  (%d rows)", out, len(daily))
        return out

    # ---------- Orchestrator ----------
    def run(self, market_path: Path, macro_path: Path) -> None:
        mkt_df = pd.read_parquet(market_path)
        macro_df = pd.read_parquet(macro_path)
        self.build_market(mkt_df)
        self.build_macro(macro_df)

# -------------------------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser("Feature Builder CLI")
    parser.add_argument("--mkt_table", required=True, help="market_intraday_YYYYMM")
    parser.add_argument("--macro_table", required=True, help="macro_indicators")
    parser.add_argument("--freq", default="1min")
    parser.add_argument("--config", default="configs/feature_config.yaml")
    args = parser.parse_args()

    cfg = BuilderConfig.from_yaml(Path(args.config))
    cfg.ohlc_freq = args.freq  # CLI 우선 적용

    builder = FeatureBuilder(cfg)

    market_path = Path("feature_store") / f"{args.mkt_table}.parquet"
    macro_path = Path("feature_store") / f"{args.macro_table}.parquet"

    if not market_path.exists() or not macro_path.exists():
        logger.error(
            "❌ Raw parquet not found. Run ingestion first: %s  %s",
            market_path,
            macro_path,
        )
        sys.exit(1)

    builder.run(market_path, macro_path)
