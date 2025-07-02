
"""
gnn_trading.models.tgat
=======================
Phase‑3 : **TGATModel & Trainer**

* 간단한 Graph‑Attention 기반 시계열 예측용 네트워크
  (실제 논문 TGAT의 모든 기능은 아니며, torch_geometric 의 GATConv + GRU 를
  조합한 경량 버전입니다.)
* 입력  : Data(x, edge_index, edge_attr, snapshot_ts)
* 출력  : 각 자산(sym) 노드에 대해 1‑step ahead 수익률 예측 (regression)

Hyper‑params 및 학습 config 는 `configs/train_config.yaml` 로 제어합니다.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class TGATModel(nn.Module):
    """GATConv + GRU 로 TIME‑AWARE 특징을 쌓는 간이 TGAT"""

    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    # ---- helper to apply GAT to one snapshot ----
    def _encode_graph(self, data: Data) -> torch.Tensor:
        x = data.x
        for gat in self.gat_layers:
            x = F.elu(gat(x, data.edge_index))
        return x  # [num_nodes, hidden]

    def forward(self, batch_data: list[Data]) -> torch.Tensor:
        """batch_data: list length B (snapshot sequence)
        -> returns tensor [B, num_nodes, 1]
        """
        encodings = []
        for d in batch_data:
            enc = self._encode_graph(d)  # [N,H]
            encodings.append(enc.unsqueeze(0))  # [1,N,H]
        seq = torch.cat(encodings, dim=0)  # [T,N,H]
        # GRU expects [B,T,H]; we transpose N and T by flattening N into batch dim
        seq = seq.permute(1, 0, 2)  # [N,T,H]
        out, _ = self.gru(seq)
        pred = self.fc_out(out[:, -1, :])  # [N,1]
        return pred  # prediction per node
