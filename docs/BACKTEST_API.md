
# Phase 4 · Backtest & API Deployment

⏱ 생성 : 2025-07-02T03:17:34

## Backtest
```bash
poetry run python -m gnn_trading.backtest.engine --config configs/backtest_config.yaml
```
결과 : `backtest_result.csv`, CLI 출력에 Sharpe/MaxDD

## API
```bash
poetry run uvicorn gnn_trading.api.main:app --reload --port 8000
# POST /predict  { "snapshot_path": "graph_snapshots/2025-07-02T00:00:00.pt" }
```
