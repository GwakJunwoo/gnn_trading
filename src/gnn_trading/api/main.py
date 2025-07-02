
"""
gnn_trading.api.main
====================
FastAPI inference server
* POST /predict  {"snapshot_path": "graph_snapshots/<ts>.pt"}
  -> returns predicted returns per node
"""
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from gnn_trading.models.tgat import TGATModel
from torch_geometric.data import Data
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

model = TGATModel()
try:
    state = torch.load(Path("checkpoints/tgat.ckpt"), map_location="cpu")
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) else state)
    model.eval()
except FileNotFoundError:
    logger.warning("Model checkpoint not found. API will not work until trained.")

class PredictRequest(BaseModel):
    snapshot_path: str

@app.post("/predict")
def predict(req: PredictRequest):
    path = Path(req.snapshot_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    snap: Data = torch.load(path)
    with torch.no_grad():
        preds = model([snap]).squeeze().tolist()
    return {"predictions": preds}
