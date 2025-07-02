"""
gnn_trading.api.main
====================
Production-grade FastAPI server for GNN Trading System
Features:
* Real-time predictions with ensemble models
* Streaming graph processing
* Batch prediction support
* Model monitoring and health checks
* Data quality validation
* Performance metrics tracking
"""
import asyncio
import logging
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import threading

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Local imports
from gnn_trading.models.tgat import TGATModel
from gnn_trading.models.ensemble import EnsemblePredictor, EnsembleConfig
from gnn_trading.graphs.streaming import StreamingGraphBuilder, StreamingConfig
from gnn_trading.graphs.graph_builder import GraphConfig
from gnn_trading.data_pipeline.quality import DataQualityManager, QualityConfig
from gnn_trading.utils.logging import setup_logging
from gnn_trading.utils.validation import validate_config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI app with enhanced configuration
app = FastAPI(
    title="GNN Trading System API",
    description="Production-grade Graph Neural Network based trading system with ensemble models and real-time processing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
ensemble_predictor: Optional[EnsemblePredictor] = None
streaming_builder: Optional[StreamingGraphBuilder] = None
quality_manager: Optional[DataQualityManager] = None
model_loaded = False
api_stats = defaultdict(int)
performance_metrics = {
    "request_count": 0,
    "avg_response_time": 0.0,
    "error_count": 0,
    "last_prediction_time": None,
    "uptime_start": datetime.now()
}

# Initialize components
def initialize_components():
    """Initialize all system components"""
    global ensemble_predictor, streaming_builder, quality_manager, model_loaded
    
    try:
        # Load configurations
        config_path = Path("configs")
        
        # Initialize data quality manager
        quality_config = QualityConfig()
        quality_manager = DataQualityManager(quality_config)
        logger.info("Data quality manager initialized")
        
        # Initialize ensemble predictor
        ensemble_config = EnsembleConfig(
            combination_method="weighted_average",
            weight_method="performance",
            max_models=5,
            enable_uncertainty_estimation=True
        )
        ensemble_predictor = EnsemblePredictor(ensemble_config, logger)
        
        # Load base model
        try:
            base_model = TGATModel()
            checkpoint_path = Path("checkpoints/tgat.ckpt")
            
            if checkpoint_path.exists():
                import torch
                state = torch.load(checkpoint_path, map_location="cpu")
                base_model.load_state_dict(state.get("state_dict", state))
                base_model.eval()
                
                # Add to ensemble
                ensemble_predictor.add_model("tgat_base", base_model, {
                    "type": "TGAT",
                    "version": "1.0",
                    "loaded_at": datetime.now().isoformat()
                })
                
                model_loaded = True
                logger.info("Base TGAT model loaded and added to ensemble")
            else:
                logger.warning("No model checkpoint found")
                
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            
        # Initialize streaming builder
        try:
            graph_config = GraphConfig()
            streaming_config = StreamingConfig(
                buffer_size=1000,
                update_frequency=60,
                max_latency=1.0,
                enable_caching=True
            )
            
            streaming_builder = StreamingGraphBuilder(
                graph_config=graph_config,
                streaming_config=streaming_config,
                feature_root=Path("data/features"),
                logger=logger
            )
            
            # Set up callbacks
            streaming_builder.set_graph_ready_callback(on_graph_ready)
            streaming_builder.set_error_callback(on_streaming_error)
            
            logger.info("Streaming graph builder initialized")
            
        except Exception as e:
            logger.error(f"Error initializing streaming builder: {e}")
            
    except Exception as e:
        logger.error(f"Error initializing components: {e}")

def on_graph_ready(graph, timestamp):
    """Callback for when new graph is ready"""
    try:
        logger.debug(f"New graph ready at {timestamp}")
        # Could trigger predictions here
    except Exception as e:
        logger.error(f"Error in graph ready callback: {e}")

def on_streaming_error(error):
    """Callback for streaming errors"""
    logger.error(f"Streaming error: {error}")
    performance_metrics["error_count"] += 1

# Request/Response models
class PredictRequest(BaseModel):
    """Request model for single prediction"""
    data: Optional[Dict[str, Any]] = Field(None, description="Market data for prediction")
    snapshot_path: Optional[str] = Field(None, description="Path to graph snapshot file")
    use_ensemble: bool = Field(True, description="Use ensemble prediction")
    return_uncertainty: bool = Field(False, description="Return prediction uncertainty")

class BatchPredictRequest(BaseModel):
    """Request model for batch prediction"""
    data_batch: Optional[List[Dict[str, Any]]] = Field(None, description="Batch of market data")
    snapshot_paths: Optional[List[str]] = Field(None, description="List of paths to graph snapshot files")
    use_ensemble: bool = Field(True, description="Use ensemble prediction")
    return_uncertainty: bool = Field(False, description="Return prediction uncertainty")

class MarketDataRequest(BaseModel):
    """Request model for market data ingestion"""
    data: List[Dict[str, Any]] = Field(..., description="Market data points")
    validate_quality: bool = Field(True, description="Run data quality checks")

class PredictResponse(BaseModel):
    """Response model for predictions"""
    prediction: float = Field(..., description="Predicted value")
    uncertainty: Optional[float] = Field(None, description="Prediction uncertainty")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Prediction metadata")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictResponse] = Field(..., description="List of predictions")
    batch_metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    ensemble_status: Dict[str, Any] = Field(default_factory=dict, description="Ensemble status")
    streaming_status: Dict[str, Any] = Field(default_factory=dict, description="Streaming status")
    quality_status: Dict[str, Any] = Field(default_factory=dict, description="Data quality status")

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    ensemble_info: Dict[str, Any] = Field(default_factory=dict, description="Ensemble information")
    model_count: int = Field(..., description="Number of models in ensemble")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    streaming_stats: Dict[str, Any] = Field(default_factory=dict, description="Streaming statistics")

class QualityReportResponse(BaseModel):
    """Response model for data quality report"""
    overall_score: float = Field(..., description="Overall quality score")
    checks: List[Dict[str, Any]] = Field(..., description="Individual quality checks")
    data_summary: Dict[str, Any] = Field(..., description="Data summary statistics")
    timestamp: str = Field(..., description="Report timestamp")

# Dependency functions
def get_ensemble_predictor():
    """Get ensemble predictor dependency"""
    if ensemble_predictor is None:
        raise HTTPException(status_code=503, detail="Ensemble predictor not available")
    return ensemble_predictor

def get_quality_manager():
    """Get quality manager dependency"""
    if quality_manager is None:
        raise HTTPException(status_code=503, detail="Quality manager not available")
    return quality_manager

def get_streaming_builder():
    """Get streaming builder dependency"""
    if streaming_builder is None:
        raise HTTPException(status_code=503, detail="Streaming builder not available")
    return streaming_builder

# Utility functions
def track_request(endpoint: str):
    """Track API request metrics"""
    performance_metrics["request_count"] += 1
    api_stats[endpoint] += 1

def calculate_response_time(start_time: float) -> float:
    """Calculate and update response time metrics"""
    response_time = time.time() - start_time
    
    # Update average response time
    count = performance_metrics["request_count"]
    current_avg = performance_metrics["avg_response_time"]
    performance_metrics["avg_response_time"] = (
        (current_avg * (count - 1) + response_time) / count
    )
    
    return response_time

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting GNN Trading API...")
    initialize_components()
    
    # Start streaming builder if available
    if streaming_builder:
        try:
            streaming_builder.start_streaming()
            logger.info("Streaming builder started")
        except Exception as e:
            logger.error(f"Error starting streaming builder: {e}")
    
    logger.info("GNN Trading API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down GNN Trading API...")
    
    # Stop streaming builder
    if streaming_builder:
        try:
            streaming_builder.stop_streaming()
            logger.info("Streaming builder stopped")
        except Exception as e:
            logger.error(f"Error stopping streaming builder: {e}")
    
    logger.info("GNN Trading API shutdown complete")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    track_request("health")
    
    ensemble_status = {}
    streaming_status = {}
    quality_status = {}
    
    # Check ensemble status
    if ensemble_predictor:
        try:
            ensemble_status = {
                "available": True,
                "model_count": len(ensemble_predictor.models),
                "is_fitted": ensemble_predictor.is_fitted,
                "prediction_count": ensemble_predictor.prediction_count
            }
        except Exception as e:
            ensemble_status = {"available": False, "error": str(e)}
    else:
        ensemble_status = {"available": False}
    
    # Check streaming status
    if streaming_builder:
        try:
            streaming_status = {
                "available": True,
                "stats": streaming_builder.get_performance_stats()
            }
        except Exception as e:
            streaming_status = {"available": False, "error": str(e)}
    else:
        streaming_status = {"available": False}
    
    # Check quality manager status
    if quality_manager:
        quality_status = {"available": True}
    else:
        quality_status = {"available": False}
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        ensemble_status=ensemble_status,
        streaming_status=streaming_status,
        quality_status=quality_status
    )

@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info(predictor: EnsemblePredictor = Depends(get_ensemble_predictor)):
    """Get model information"""
    track_request("model_info")
    
    ensemble_info = predictor.get_ensemble_stats()
    streaming_stats = {}
    
    if streaming_builder:
        try:
            streaming_stats = streaming_builder.get_performance_stats()
        except Exception as e:
            streaming_stats = {"error": str(e)}
    
    return ModelInfoResponse(
        ensemble_info=ensemble_info,
        model_count=len(predictor.models),
        performance_metrics=performance_metrics,
        streaming_stats=streaming_stats
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    predictor: EnsemblePredictor = Depends(get_ensemble_predictor)
):
    """Make single prediction"""
    start_time = time.time()
    track_request("predict")
    
    try:
        # Handle different input types
        if request.data:
            # Convert data to appropriate format for model
            data = request.data
        elif request.snapshot_path:
            # Load graph snapshot
            try:
                import torch
                from torch_geometric.data import Data
                path = Path(request.snapshot_path)
                if not path.exists():
                    raise HTTPException(status_code=404, detail=f"Snapshot not found: {request.snapshot_path}")
                data = torch.load(path, map_location="cpu")
            except ImportError:
                raise HTTPException(status_code=503, detail="PyTorch not available")
        else:
            raise HTTPException(status_code=400, detail="Either 'data' or 'snapshot_path' must be provided")
        
        # Make prediction
        if request.return_uncertainty:
            prediction, uncertainty = predictor.predict_with_uncertainty(data)
        else:
            prediction, metadata = predictor.predict(data)
            uncertainty = metadata.get("uncertainty")
        
        # Update performance tracking
        performance_metrics["last_prediction_time"] = datetime.now().isoformat()
        response_time = calculate_response_time(start_time)
        
        return PredictResponse(
            prediction=float(prediction),
            uncertainty=float(uncertainty) if uncertainty is not None else None,
            metadata={
                "response_time": response_time,
                "use_ensemble": request.use_ensemble,
                **(metadata if 'metadata' in locals() else {})
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        performance_metrics["error_count"] += 1
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(
    request: BatchPredictRequest,
    predictor: EnsemblePredictor = Depends(get_ensemble_predictor)
):
    """Make batch predictions"""
    start_time = time.time()
    track_request("batch_predict")
    
    try:
        predictions = []
        data_sources = request.data_batch or request.snapshot_paths or []
        
        if not data_sources:
            raise HTTPException(status_code=400, detail="No data provided for batch prediction")
        
        for i, data_source in enumerate(data_sources):
            try:
                # Handle different data types
                if request.data_batch:
                    data = data_source
                else:
                    # Load from snapshot path
                    import torch
                    from torch_geometric.data import Data
                    path = Path(data_source)
                    if not path.exists():
                        logger.warning(f"Snapshot not found: {data_source}")
                        continue
                    data = torch.load(path, map_location="cpu")
                
                # Make prediction
                if request.return_uncertainty:
                    prediction, uncertainty = predictor.predict_with_uncertainty(data)
                    metadata = {"uncertainty": uncertainty}
                else:
                    prediction, metadata = predictor.predict(data)
                    uncertainty = metadata.get("uncertainty")
                
                predictions.append(PredictResponse(
                    prediction=float(prediction),
                    uncertainty=float(uncertainty) if uncertainty is not None else None,
                    metadata=metadata,
                    timestamp=datetime.now().isoformat()
                ))
                
            except Exception as e:
                logger.error(f"Error in batch prediction {i}: {e}")
                # Continue with other predictions
                continue
        
        response_time = calculate_response_time(start_time)
        
        return BatchPredictResponse(
            predictions=predictions,
            batch_metadata={
                "total_requested": len(data_sources),
                "successful_predictions": len(predictions),
                "batch_response_time": response_time,
                "avg_prediction_time": response_time / len(predictions) if predictions else 0
            }
        )
        
    except Exception as e:
        performance_metrics["error_count"] += 1
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/data/ingest")
async def ingest_market_data(
    request: MarketDataRequest,
    builder: StreamingGraphBuilder = Depends(get_streaming_builder)
):
    """Ingest real-time market data"""
    track_request("data_ingest")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Validate data quality if requested
        quality_report = None
        if request.validate_quality and quality_manager:
            quality_report = quality_manager.run_all_checks(df)
            
            # Reject if quality is too poor
            if quality_report.overall_score < 0.5:
                raise HTTPException(
                    status_code=400,
                    detail=f"Data quality too poor: {quality_report.overall_score:.3f}"
                )
        
        # Add to streaming builder
        builder.add_market_data(df)
        
        return {
            "status": "success",
            "message": f"Ingested {len(df)} data points",
            "quality_score": quality_report.overall_score if quality_report else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise HTTPException(status_code=500, detail=f"Data ingestion error: {str(e)}")

@app.get("/data/quality", response_model=QualityReportResponse)
async def get_data_quality(
    manager: DataQualityManager = Depends(get_quality_manager)
):
    """Get latest data quality report"""
    track_request("data_quality")
    
    try:
        # Get recent data from streaming builder if available
        if streaming_builder:
            recent_data = streaming_builder.data_buffer.get_recent_data(300)  # Last 5 minutes
            
            if not recent_data.empty:
                report = manager.run_all_checks(recent_data)
                
                return QualityReportResponse(
                    overall_score=report.overall_score,
                    checks=[check.__dict__ for check in report.checks],
                    data_summary=report.data_summary,
                    timestamp=report.timestamp.isoformat()
                )
        
        # No recent data available
        return QualityReportResponse(
            overall_score=0.0,
            checks=[],
            data_summary={"message": "No recent data available"},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting quality report: {e}")
        raise HTTPException(status_code=500, detail=f"Quality report error: {str(e)}")

@app.get("/streaming/status")
async def get_streaming_status(
    builder: StreamingGraphBuilder = Depends(get_streaming_builder)
):
    """Get streaming system status"""
    track_request("streaming_status")
    
    try:
        stats = builder.get_performance_stats()
        
        return {
            "status": "active",
            "performance_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming status error: {str(e)}")

@app.post("/streaming/control")
async def control_streaming(
    action: str,
    builder: StreamingGraphBuilder = Depends(get_streaming_builder)
):
    """Control streaming system"""
    track_request("streaming_control")
    
    try:
        if action == "start":
            builder.start_streaming()
            message = "Streaming started"
        elif action == "stop":
            builder.stop_streaming()
            message = "Streaming stopped"
        elif action == "clear_cache":
            builder.clear_cache()
            message = "Cache cleared"
        elif action == "optimize":
            builder.optimize_memory()
            message = "Memory optimized"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
        
        return {
            "status": "success",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error controlling streaming: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming control error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    track_request("metrics")
    
    uptime = datetime.now() - performance_metrics["uptime_start"]
    
    return {
        "performance_metrics": performance_metrics,
        "api_stats": dict(api_stats),
        "uptime_seconds": uptime.total_seconds(),
        "ensemble_stats": ensemble_predictor.get_ensemble_stats() if ensemble_predictor else {},
        "streaming_stats": streaming_builder.get_performance_stats() if streaming_builder else {},
        "timestamp": datetime.now().isoformat()
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws/predictions")
async def websocket_predictions(websocket):
    """WebSocket endpoint for real-time predictions"""
    try:
        from fastapi import WebSocket
        await websocket.accept()
        
        # Setup real-time prediction callback
        if streaming_builder:
            def prediction_callback(graph, timestamp):
                try:
                    if ensemble_predictor and ensemble_predictor.is_fitted:
                        prediction, metadata = ensemble_predictor.predict(graph)
                        
                        asyncio.create_task(websocket.send_json({
                            "type": "prediction",
                            "prediction": float(prediction),
                            "timestamp": timestamp.isoformat(),
                            "metadata": metadata
                        }))
                except Exception as e:
                    logger.error(f"Error in WebSocket prediction callback: {e}")
            
            streaming_builder.set_graph_ready_callback(prediction_callback)
        
        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except Exception:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    performance_metrics["error_count"] += 1
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    performance_metrics["error_count"] += 1
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Main function for running the server
def main():
    """Main function to run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GNN Trading API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Run server
    try:
        uvicorn.run(
            "gnn_trading.api.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level
        )
    except ImportError:
        logger.error("uvicorn not available. Install with: pip install uvicorn")
        print("Starting with basic server...")
        # Fallback to basic server if uvicorn not available
        import threading
        import http.server
        import socketserver
        
        class APIHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "GNN Trading API is running"}')
                
        with socketserver.TCPServer(("", args.port), APIHandler) as httpd:
            print(f"Basic server running on port {args.port}")
            httpd.serve_forever()

if __name__ == "__main__":
    main()
