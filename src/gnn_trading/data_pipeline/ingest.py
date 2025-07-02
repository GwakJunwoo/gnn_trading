
"""
gnn_trading.data_pipeline.ingest
===================================
데이터 인제스트 레이어: KRX·선물 인트라데이와 ECOS/KOSIS 거시지표를
수집하여 Feature Store(PARQUET)에 저장합니다.

⚙️ **사용 방법 요약**
---------------------
1. `/configs/source_config.yaml` 에 API 엔드포인트·인증키·자산/지표 목록을 작성합니다.
   예시:
   ```yaml
   market:
     base_url: "https://krx.example/api"
     api_key: null
     asset_list: ["KS200", "KTB3F"]
     interval: "1min"

   macro:
     base_url: "https://ecos.bok.or.kr/openapi/service/rest/StatisticSearchJSON"
     api_key: "YOUR_BOK_APIKEY"
     indicator_list: ["901Y001", "CPI"]

   store:
     root: "feature_store"
     format: "parquet"
   ```
2. 프로젝트 루트에서:
   ```bash
   poetry run python -m gnn_trading.data_pipeline.ingest_cli \
       --start 2025-06-01 --end 2025-06-30
   ```
   → 해당 기간의 시장·거시 데이터를 수집해 `feature_store/` 폴더에
     `market_intraday_YYYYMM.parquet`, `macro_indicators.parquet` 로 저장합니다.

3. Feature Store 위치는 `FeatureStoreConfig.root` 로 지정하며,
   향후 모델 학습·예측 단계에서 동일 경로를 참조합니다.

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------------
# Config dataclasses
# -------------------------------------------------------------------------

@dataclass
class SourceConfig:
    """API 엔드포인트·키·추가 파라미터"""

    base_url: str
    api_key: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None

    def to_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(self.extra_params or {})
        if self.api_key:
            params["apikey"] = self.api_key
        return params


@dataclass
class FeatureStoreConfig:
    """Feature Store 저장 경로·포맷"""

    root: Path = Path("feature_store")
    format: str = "parquet"
    partition_cols: Optional[List[str]] = None

    def target_path(self, table: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / f"{table}.{self.format}"


# -------------------------------------------------------------------------
# Abstract Ingestor
# -------------------------------------------------------------------------

class DataIngestor(ABC):
    """데이터 인제스트 추상 클래스"""

    def __init__(self, source_cfg: SourceConfig, store_cfg: FeatureStoreConfig):
        self.source_cfg = source_cfg
        self.store_cfg = store_cfg

    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """원천 API 호출"""

    @abstractmethod
    def normalize(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """컬럼 정제·타입 변환"""

    def ingest(self, start: datetime, end: datetime, table: str) -> Path:
        """Fetch → Normalize → 저장"""
        logger.info("[Ingest] %s — %s (%s)", start.date(), end.date(), table)
        raw = self.fetch(start, end)
        if raw.empty:
            logger.warning("No data fetched: %s", table)
            return self.store_cfg.target_path(table)
        df = self.normalize(raw)
        self._write(df, self.store_cfg.target_path(table))
        logger.info("Saved %d rows → %s", len(df), table)
        return self.store_cfg.target_path(table)

    # internal
    def _write(self, df: pd.DataFrame, target: Path) -> None:
        if self.store_cfg.format == "parquet":
            df.to_parquet(target, index=False)
        else:
            df.to_csv(target, index=False)


# -------------------------------------------------------------------------
# MarketDataIngestor (KRX/선물)
# -------------------------------------------------------------------------

class MarketDataIngestor(DataIngestor):
    """KRX·선물 인트라데이 수집"""

    def __init__(
        self,
        asset_list: List[str],
        source_cfg: SourceConfig,
        store_cfg: FeatureStoreConfig,
        interval: str = "1min",
    ):
        super().__init__(source_cfg, store_cfg)
        self.asset_list = asset_list
        self.interval = interval

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for symbol in self.asset_list:
            params = self.source_cfg.to_params() | {
                "symbol": symbol,
                "interval": self.interval,
                "start": int(start.timestamp()),
                "end": int(end.timestamp()),
            }
            resp = requests.get(self.source_cfg.base_url, params=params, timeout=20)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json())
            df["symbol"] = symbol
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def normalize(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "t": "datetime",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "symbol": "symbol",
        }
        df = raw_df.rename(columns=col_map)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
        df = df.astype(
            {
                "open": "float32",
                "high": "float32",
                "low": "float32",
                "close": "float32",
                "volume": "float32",
            }
        )
        return df[[*col_map.values()]]


# -------------------------------------------------------------------------
# MacroDataIngestor (ECOS/KOSIS)
# -------------------------------------------------------------------------

class MacroDataIngestor(DataIngestor):
    """거시경제 지표 수집"""

    def __init__(
        self,
        indicator_list: List[str],
        source_cfg: SourceConfig,
        store_cfg: FeatureStoreConfig,
    ):
        super().__init__(source_cfg, store_cfg)
        self.indicator_list = indicator_list

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for ind in self.indicator_list:
            params = self.source_cfg.to_params() | {
                "statCode": ind,
                "start": start.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
            }
            resp = requests.get(self.source_cfg.base_url, params=params, timeout=20)
            resp.raise_for_status()
            js = resp.json()
            for item in js.get("data", []):
                rows.append(
                    {
                        "date": pd.to_datetime(item["TIME"]),
                        "value": float(item["DATA_VALUE"]),
                        "indicator": ind,
                    }
                )
        return pd.DataFrame(rows)

    def normalize(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.astype({"value": "float32"})
        return df[["date", "indicator", "value"]]


# -------------------------------------------------------------------------
# CLI helper
# -------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser("Data Ingest CLI")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/source_config.yaml")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    cfg = load_yaml(Path(args.config))

    store_cfg = FeatureStoreConfig(root=Path(cfg["store"]["root"]))

    # market
    mkt_ing = MarketDataIngestor(
        asset_list=cfg["market"]["asset_list"],
        source_cfg=SourceConfig(
            base_url=cfg["market"]["base_url"],
            api_key=cfg["market"].get("api_key"),
        ),
        store_cfg=store_cfg,
        interval=cfg["market"].get("interval", "1min"),
    )
    mkt_table = f"market_intraday_{start.strftime('%Y%m')}"
    mkt_ing.ingest(start, end, mkt_table)

    # macro
    macro_ing = MacroDataIngestor(
        indicator_list=cfg["macro"]["indicator_list"],
        source_cfg=SourceConfig(
            base_url=cfg["macro"]["base_url"],
            api_key=cfg["macro"].get("api_key"),
        ),
        store_cfg=store_cfg,
    )
    macro_ing.ingest(start, end, "macro_indicators")

if __name__ == "__main__":
    main()
