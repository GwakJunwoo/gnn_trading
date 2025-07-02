"""
gnn_trading.api.main
====================
FastAPI inference server
* POST /predict  {"snapshot_path": "graph_snapshots/<ts>.pt"}
  -> returns predicted returns per node
* GET /health - health check
* POST /batch_predict - batch prediction
* GET /model_info - model information
"""
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import json
from datetime import datetime

from gnn_trading.models.tgat import TGATModel
from torch_geometric.data import Data

app = FastAPI(
    title="GNN Trading API",
    description="Graph Neural Network based trading predictions",
    version="1.0.0"
)
logger = logging.getLogger(__name__)

# Global model instance
model = TGATModel()
model_loaded = False

try:
    state = torch.load(Path("checkpoints/tgat.ckpt"), map_location="cpu")
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) else state)
    model.eval()
    model_loaded = True
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.warning("Model checkpoint not found. API will not work until trained.")

# Request/Response models
class PredictRequest(BaseModel):
    snapshot_path: str = Field(..., description="Path to graph snapshot file")

class BatchPredictRequest(BaseModel):
    snapshot_paths: List[str] = Field(..., description="List of paths to graph snapshot files")
    
class PredictResponse(BaseModel):
    predictions: List[float] = Field(..., description="Predicted returns per node")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")

class BatchPredictResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Batch prediction results")
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime

class ModelInfoResponse(BaseModel):
    model_name: str
    parameters: Dict[str, Any]
    last_updated: Optional[datetime]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        timestamp=datetime.now()
    )

@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="TGAT",
        parameters={
            "in_dim": model.in_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "heads": model.heads,
            "dropout": model.dropout
        },
        last_updated=datetime.now()  # TODO: Get actual model timestamp
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Single prediction endpoint"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    path = Path(req.snapshot_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    
    try:
        snap: Data = torch.load(path, map_location="cpu")
        with torch.no_grad():
            preds = model([snap]).squeeze().tolist()
        
        return PredictResponse(
            predictions=preds,
            timestamp=datetime.now(),
            model_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(req: BatchPredictRequest):
    """Batch prediction endpoint"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for snapshot_path in req.snapshot_paths:
        path = Path(snapshot_path)
        try:
            if not path.exists():
                results.append({
                    "snapshot_path": snapshot_path,
                    "status": "error",
                    "error": "File not found"
                })
                continue
                
            snap: Data = torch.load(path, map_location="cpu")
            with torch.no_grad():
                preds = model([snap]).squeeze().tolist()
            
            results.append({
                "snapshot_path": snapshot_path,
                "status": "success",
                "predictions": preds,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            results.append({
                "snapshot_path": snapshot_path,
                "status": "error",
                "error": str(e)
            })
    
    return BatchPredictResponse(results=results)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GNN Trading API",
        "version": "1.0.0",
        "status": "healthy" if model_loaded else "model_not_loaded",
        "endpoints": {
            "POST /predict": "Single prediction",
            "POST /batch_predict": "Batch predictions", 
            "GET /health": "Health check",
            "GET /model_info": "Model information",
            "GET /docs": "API documentation"
        }
    }
