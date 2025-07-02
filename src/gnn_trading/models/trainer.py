
"""
gnn_trading.models.trainer
==========================
PyTorch‑Lightning Trainer wrapper
* DataLoader : GraphDataset(Phase‑2) → sequence batching
* Loss       : MSE(예측 수익률, 실제 t+1 수익률)
"""

from __future__ import annotations
from pathlib import Path
import logging
from typing import List
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gnn_trading.models.tgat import TGATModel
from gnn_trading.graphs.graph_builder import GraphDataset

logger = logging.getLogger(__name__)

# ---- Sequence collate ----
def collate_seq(batch: List, seq_len: int = 5):
    """Return sequences of length `seq_len` ending at each idx"""
    data_seq, target_seq = [], []
    for i in range(len(batch) - seq_len - 1):
        data_seq.append(batch[i : i + seq_len])
        target_seq.append(batch[i + seq_len])  # predict next snapshot
    return data_seq, target_seq

class TGATLightningModule(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TGATModel()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, batch_data):
        return self.model(batch_data)

    def training_step(self, batch, batch_idx):
        x_seq, y_snap = batch
        preds = self(x_seq)
        # y_snap.x : using node returns as ground truth
        target = y_snap.x[:, 0:1]  # assume first column is return
        loss = self.loss_fn(preds, target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ----------------- Datamodule -----------------
class GraphDataModule(pl.LightningDataModule):
    def __init__(self, snapshot_dir: Path, batch_size: int = 1, seq_len: int = 5):
        super().__init__()
        self.snapshot_dir = snapshot_dir
        self.batch_size = batch_size
        self.seq_len = seq_len

    def setup(self, stage=None):
        self.dataset = GraphDataset(self.snapshot_dir)

        # Generate sequence indices
        self.samples = []
        for i in range(len(self.dataset) - self.seq_len - 1):
            seq = [self.dataset[j] for j in range(i, i + self.seq_len + 1)]
            self.samples.append(seq)

    def __len__(self):
        return len(self.samples)

    def _collate(self, batch):
        xs = [item[:-1] for item in batch]  # seq_len list
        ys = [item[-1] for item in batch]
        # simplify: batch size 1 to avoid complex pad
        return xs[0], ys[0]

    def train_dataloader(self):
        return DataLoader(self.samples, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate)


# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse, yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser("TGAT Training CLI")
    parser.add_argument("--snapshot_dir", default="graph_snapshots")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    args = parser.parse_args()

    dm = GraphDataModule(Path(args.snapshot_dir), args.batch_size, args.seq_len)
    model = TGATLightningModule(lr=args.lr)
    trainer = pl.Trainer(max_epochs=args.epochs, default_root_dir=args.ckpt_dir, log_every_n_steps=1)
    trainer.fit(model, dm)
    trainer.save_checkpoint(Path(args.ckpt_dir) / "tgat.ckpt")
