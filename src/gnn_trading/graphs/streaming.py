"""
gnn_trading.graphs.streaming
===========================

Real-time streaming graph builder for live trading systems.
Optimized for low-latency graph construction and incremental updates.
"""

from __future__ import annotations
import asyncio
import logging
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import queue
import threading
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .graph_builder import GraphConfig, GraphSnapshotBuilder


@dataclass
class StreamingConfig:
    """Configuration for streaming graph builder"""
    buffer_size: int = 1000
    update_frequency: int = 60  # seconds
    max_latency: float = 1.0  # seconds
    batch_size: int = 32
    num_workers: int = 4
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    enable_compression: bool = True
    memory_limit: int = 500  # MB
    
    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.update_frequency <= 0:
            raise ValueError("update_frequency must be positive")
        if self.max_latency <= 0:
            raise ValueError("max_latency must be positive")


class RealTimeDataBuffer:
    """Thread-safe circular buffer for real-time data"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._last_update = time.time()
        
    def add_data(self, data: pd.DataFrame) -> None:
        """Add new data to buffer"""
        with self._lock:
            timestamp = time.time()
            self._buffer.append((timestamp, data))
            self._last_update = timestamp
            
    def get_recent_data(self, window_seconds: int = 300) -> pd.DataFrame:
        """Get recent data within time window"""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            recent_data = []
            for timestamp, data in self._buffer:
                if timestamp >= cutoff_time:
                    recent_data.append(data)
                    
            if recent_data:
                return pd.concat(recent_data, ignore_index=True)
            else:
                return pd.DataFrame()
                
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            return {
                "size": len(self._buffer),
                "max_size": self.max_size,
                "last_update": self._last_update,
                "age_seconds": time.time() - self._last_update if self._buffer else 0
            }
            
    def clear(self) -> None:
        """Clear buffer"""
        with self._lock:
            self._buffer.clear()


class GraphCache:
    """LRU cache for graph objects with TTL"""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[float, Data]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Data]:
        """Get cached graph"""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                timestamp, graph = self._cache[key]
                
                # Check TTL
                if current_time - timestamp <= self.ttl:
                    self._access_times[key] = current_time
                    return graph
                else:
                    # Expired
                    del self._cache[key]
                    del self._access_times[key]
                    
            return None
            
    def put(self, key: str, graph: Data) -> None:
        """Cache graph"""
        with self._lock:
            current_time = time.time()
            
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._access_times.keys(), 
                               key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
                
            self._cache[key] = (current_time, graph)
            self._access_times[key] = current_time
            
    def clear(self) -> None:
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            current_time = time.time()
            valid_entries = sum(
                1 for timestamp, _ in self._cache.values()
                if current_time - timestamp <= self.ttl
            )
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "valid_entries": valid_entries,
                "hit_rate": getattr(self, '_hit_rate', 0.0)
            }


class StreamingGraphBuilder:
    """Real-time streaming graph builder"""
    
    def __init__(
        self,
        graph_config: GraphConfig,
        streaming_config: StreamingConfig,
        feature_root: Path,
        logger: Optional[logging.Logger] = None
    ):
        self.graph_config = graph_config
        self.streaming_config = streaming_config
        self.feature_root = Path(feature_root)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.data_buffer = RealTimeDataBuffer(streaming_config.buffer_size)
        self.graph_cache = GraphCache(
            max_size=streaming_config.buffer_size // 10,
            ttl=streaming_config.cache_ttl
        ) if streaming_config.enable_caching else None
        
        # Base graph builder for heavy operations
        self.base_builder = GraphSnapshotBuilder(
            cfg=graph_config,
            feature_root=feature_root,
            out_dir=feature_root / "streaming"
        )
        
        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=streaming_config.num_workers)
        self._update_thread = None
        self._stop_event = threading.Event()
        self._graph_queue = queue.Queue(maxsize=streaming_config.buffer_size)
        
        # Performance tracking
        self.performance_stats = {
            "graphs_built": 0,
            "avg_build_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }
        
        # Callbacks
        self.graph_ready_callback: Optional[Callable[[Data, datetime], None]] = None
        self.error_callback: Optional[Callable[[Exception], None]] = None
        
    def start_streaming(self) -> None:
        """Start streaming graph builder"""
        if self._update_thread and self._update_thread.is_alive():
            self.logger.warning("Streaming already started")
            return
            
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        
        self.logger.info("Streaming graph builder started")
        
    def stop_streaming(self) -> None:
        """Stop streaming graph builder"""
        if self._update_thread and self._update_thread.is_alive():
            self._stop_event.set()
            self._update_thread.join(timeout=5.0)
            
        self.executor.shutdown(wait=True)
        self.logger.info("Streaming graph builder stopped")
        
    def add_market_data(self, data: pd.DataFrame) -> None:
        """Add new market data for processing"""
        try:
            # Validate data
            required_cols = ['datetime', 'symbol', 'close', 'return']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Data must contain columns: {required_cols}")
                
            # Add to buffer
            self.data_buffer.add_data(data)
            
            # Log data ingestion
            self.logger.debug(f"Added {len(data)} data points to buffer")
            
        except Exception as e:
            self.logger.error(f"Error adding market data: {e}")
            if self.error_callback:
                self.error_callback(e)
                
    def build_graph_incremental(
        self,
        timestamp: datetime,
        force_rebuild: bool = False
    ) -> Optional[Data]:
        """Build graph incrementally with caching"""
        try:
            cache_key = f"{timestamp.isoformat()}_{hash(str(self.graph_config))}"
            
            # Check cache first
            if self.graph_cache and not force_rebuild:
                cached_graph = self.graph_cache.get(cache_key)
                if cached_graph is not None:
                    self.performance_stats["cache_hits"] += 1
                    return cached_graph
                    
            # Build new graph
            start_time = time.time()
            
            # Get recent data
            recent_data = self.data_buffer.get_recent_data(
                window_seconds=self.streaming_config.update_frequency * 2
            )
            
            if recent_data.empty:
                self.logger.warning("No recent data available for graph building")
                return None
                
            # Build graph using optimized method
            graph = self._build_graph_optimized(recent_data, timestamp)
            
            if graph is not None:
                # Cache result
                if self.graph_cache:
                    self.graph_cache.put(cache_key, graph)
                    
                # Update performance stats
                build_time = time.time() - start_time
                self._update_performance_stats(build_time)
                
                self.logger.debug(f"Built graph in {build_time:.3f}s")
                
            return graph
            
        except Exception as e:
            self.logger.error(f"Error building incremental graph: {e}")
            self.performance_stats["errors"] += 1
            if self.error_callback:
                self.error_callback(e)
            return None
            
    def _build_graph_optimized(self, data: pd.DataFrame, timestamp: datetime) -> Optional[Data]:
        """Optimized graph building for streaming"""
        try:
            # Filter data to recent window
            end_time = timestamp
            start_time = end_time - timedelta(seconds=self.graph_config.corr_window * 60)
            
            recent_data = data[
                (data['datetime'] >= start_time) & (data['datetime'] <= end_time)
            ].copy()
            
            if len(recent_data) < 10:  # Minimum data points
                return None
                
            # Get unique symbols
            symbols = recent_data['symbol'].unique()
            if len(symbols) < 2:
                return None
                
            # Build node features (simplified for speed)
            node_features = []
            symbol_to_idx = {symbol: i for i, symbol in enumerate(symbols)}
            
            for symbol in symbols:
                symbol_data = recent_data[recent_data['symbol'] == symbol]
                
                if len(symbol_data) > 0:
                    # Basic features
                    recent_return = symbol_data['return'].iloc[-1] if len(symbol_data) > 0 else 0.0
                    avg_return = symbol_data['return'].mean()
                    vol = symbol_data['return'].std() if len(symbol_data) > 1 else 0.0
                    
                    features = [recent_return, avg_return, vol]
                else:
                    features = [0.0, 0.0, 0.0]
                    
                node_features.append(features)
                
            # Build edges based on correlation (simplified)
            edge_indices = []
            edge_attrs = []
            
            if len(symbols) > 1:
                # Calculate pairwise correlations
                pivot_data = recent_data.pivot_table(
                    index='datetime', 
                    columns='symbol', 
                    values='return'
                ).fillna(0)
                
                if len(pivot_data) > 1:
                    corr_matrix = pivot_data.corr().fillna(0)
                    
                    for i, sym1 in enumerate(symbols):
                        for j, sym2 in enumerate(symbols):
                            if i != j and sym1 in corr_matrix.index and sym2 in corr_matrix.columns:
                                corr = corr_matrix.loc[sym1, sym2]
                                
                                if abs(corr) > self.graph_config.corr_threshold:
                                    edge_indices.append([i, j])
                                    edge_attrs.append([corr, abs(corr), 1.0])
                                    
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float32)
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
            else:
                # Create self-loops if no edges
                edge_index = torch.tensor([[i, i] for i in range(len(symbols))], dtype=torch.long).t()
                edge_attr = torch.ones((len(symbols), 3), dtype=torch.float32)
                
            # Create graph
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                timestamp=timestamp,
                symbols=symbols
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Error in optimized graph building: {e}")
            return None
            
    def _update_loop(self) -> None:
        """Main update loop for streaming"""
        while not self._stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Build new graph
                graph = self.build_graph_incremental(current_time)
                
                if graph is not None:
                    # Add to queue for processing
                    try:
                        self._graph_queue.put_nowait((current_time, graph))
                    except queue.Full:
                        # Remove oldest if queue is full
                        try:
                            self._graph_queue.get_nowait()
                            self._graph_queue.put_nowait((current_time, graph))
                        except queue.Empty:
                            pass
                            
                    # Call callback if set
                    if self.graph_ready_callback:
                        self.graph_ready_callback(graph, current_time)
                        
                # Sleep until next update
                self._stop_event.wait(self.streaming_config.update_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                self.performance_stats["errors"] += 1
                if self.error_callback:
                    self.error_callback(e)
                    
                # Continue after error
                time.sleep(1.0)
                
    def _update_performance_stats(self, build_time: float) -> None:
        """Update performance statistics"""
        self.performance_stats["graphs_built"] += 1
        
        # Update average build time
        prev_avg = self.performance_stats["avg_build_time"]
        count = self.performance_stats["graphs_built"]
        
        self.performance_stats["avg_build_time"] = (
            (prev_avg * (count - 1) + build_time) / count
        )
        
    def get_latest_graph(self, timeout: float = 1.0) -> Optional[Tuple[datetime, Data]]:
        """Get latest graph from queue"""
        try:
            return self._graph_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add buffer and cache stats
        stats["buffer_stats"] = self.data_buffer.get_buffer_stats()
        
        if self.graph_cache:
            stats["cache_stats"] = self.graph_cache.get_stats()
            
        # Calculate cache hit rate
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_requests
        else:
            stats["cache_hit_rate"] = 0.0
            
        return stats
        
    def set_graph_ready_callback(self, callback: Callable[[Data, datetime], None]) -> None:
        """Set callback for when new graph is ready"""
        self.graph_ready_callback = callback
        
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for error handling"""
        self.error_callback = callback
        
    def clear_cache(self) -> None:
        """Clear all caches"""
        if self.graph_cache:
            self.graph_cache.clear()
        self.data_buffer.clear()
        
        # Clear queue
        while not self._graph_queue.empty():
            try:
                self._graph_queue.get_nowait()
            except queue.Empty:
                break
                
    def optimize_memory(self) -> None:
        """Optimize memory usage"""
        # Clear old cache entries
        if self.graph_cache:
            self.graph_cache.clear()
            
        # Reset performance stats if too large
        if self.performance_stats["graphs_built"] > 10000:
            self.performance_stats = {
                "graphs_built": 0,
                "avg_build_time": self.performance_stats["avg_build_time"],
                "cache_hits": 0,
                "cache_misses": 0,
                "errors": 0
            }
            
        self.logger.info("Memory optimization completed")


class AsyncStreamingGraphBuilder:
    """Async version of streaming graph builder"""
    
    def __init__(
        self,
        graph_config: GraphConfig,
        streaming_config: StreamingConfig,
        feature_root: Path,
        logger: Optional[logging.Logger] = None
    ):
        self.sync_builder = StreamingGraphBuilder(
            graph_config, streaming_config, feature_root, logger
        )
        self._tasks: List[asyncio.Task] = []
        
    async def start_streaming(self) -> None:
        """Start async streaming"""
        loop = asyncio.get_event_loop()
        
        # Run sync builder in thread pool
        await loop.run_in_executor(None, self.sync_builder.start_streaming)
        
        # Start async monitoring task
        self._tasks.append(asyncio.create_task(self._monitor_performance()))
        
    async def stop_streaming(self) -> None:
        """Stop async streaming"""
        # Cancel async tasks
        for task in self._tasks:
            task.cancel()
            
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except Exception:
            pass
            
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.sync_builder.stop_streaming)
        
    async def add_market_data(self, data: pd.DataFrame) -> None:
        """Add market data asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.sync_builder.add_market_data, data)
        
    async def get_latest_graph(self, timeout: float = 1.0) -> Optional[Tuple[datetime, Data]]:
        """Get latest graph asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.sync_builder.get_latest_graph, timeout
        )
        
    async def _monitor_performance(self) -> None:
        """Monitor performance asynchronously"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                stats = self.sync_builder.get_performance_stats()
                
                # Log performance metrics
                if self.sync_builder.logger:
                    self.sync_builder.logger.info(f"Performance stats: {stats}")
                    
                # Auto-optimize if needed
                if stats.get("avg_build_time", 0) > self.sync_builder.streaming_config.max_latency:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.sync_builder.optimize_memory
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.sync_builder.logger:
                    self.sync_builder.logger.error(f"Error in performance monitoring: {e}")
                    
    def __getattr__(self, name):
        """Delegate to sync builder"""
        return getattr(self.sync_builder, name)
