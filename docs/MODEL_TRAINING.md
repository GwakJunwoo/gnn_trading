
# Model Training Layer (Phase‑3)

> 생성 : 2025-07-02T03:15:59

## 구성
- `models/tgat.py`        : GATConv+GRU 간이 TGAT 네트워크
- `models/trainer.py`     : LightningModule + DataModule + CLI

## 학습 예시
```bash
poetry run python -m gnn_trading.models.trainer \
    --snapshot_dir graph_snapshots \
    --epochs 10 --batch_size 1 --seq_len 5
```
