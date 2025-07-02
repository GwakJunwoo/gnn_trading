
# gnn-trading · Data Ingestion Layer

> 생성 일시: 2025-07-02T03:12:45

이 디렉터리는 **KRX/선물 인트라데이** 및 **ECOS·KOSIS 거시지표**를
수집해 `/mnt/data/gnn_trading/configs/source_config.yaml` 에 정의된 Feature Store 위치
(`feature_store/`)로 **Parquet** 파일을 저장합니다.

## 빠른 시작

```bash
cd gnn-trading
poetry install
poetry run python -m gnn_trading.data_pipeline.ingest \
    --start 2025-06-01 --end 2025-06-30
```

## 중요 파일
- `src/gnn_trading/data_pipeline/ingest.py` : 인제스트 로직 (CLI 포함)
- `configs/source_config.yaml` : API·자산·지표·저장경로 설정
