"""
gnn_trading.optimization.performance
===================================

Performance optimization utilities for production deployment
Includes caching, profiling, memory management, and batch processing
"""

import asyncio
import functools
import logging
import time
import threading
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import weakref
import gc
import psutil

import numpy as np
import torch
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    batch_size: int = 1
    throughput: float = 0.0


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
                
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
                
            self.cache[key] = value
            
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class MemoryPool:
    """Memory pool for tensor reuse"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.pools = defaultdict(list)  # shape -> list of tensors
        self.lock = threading.Lock()
        self.max_pool_size = 100
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool or create new one"""
        key = (shape, dtype)
        
        with self.lock:
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].pop()
                tensor.zero_()  # Clear data
                return tensor
                
        # Create new tensor if none available
        return torch.zeros(shape, dtype=dtype, device=self.device)
        
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool"""
        if tensor.device.type != self.device:
            return  # Don't pool tensors from different devices
            
        key = (tuple(tensor.shape), tensor.dtype)
        
        with self.lock:
            if len(self.pools[key]) < self.max_pool_size:
                self.pools[key].append(tensor.detach().clone())
                
    def clear(self):
        """Clear all pools"""
        with self.lock:
            self.pools.clear()
            
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            total_tensors = sum(len(pool) for pool in self.pools.values())
            return {
                'total_pools': len(self.pools),
                'total_tensors': total_tensors,
                'pool_sizes': {str(k): len(v) for k, v in self.pools.items()}
            }


class BatchProcessor:
    """Batch processing utility for efficient inference"""
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.processing = False
        
    async def process_batch(self, processor_func: Callable, *args, **kwargs) -> Any:
        """Add request to batch and wait for processing"""
        request_id = id(args)
        result_future = asyncio.Future()
        
        with self.condition:
            self.pending_requests.append((request_id, args, kwargs, result_future))
            
            if len(self.pending_requests) >= self.batch_size or not self.processing:
                self.condition.notify()
                
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batches(processor_func))
            
        return await result_future
        
    async def _process_batches(self, processor_func: Callable):
        """Process batches of requests"""
        self.processing = True
        
        try:
            while True:
                with self.condition:
                    # Wait for requests or timeout
                    self.condition.wait(timeout=self.max_wait_time)
                    
                    if not self.pending_requests:
                        break
                        
                    # Get batch of requests
                    batch = self.pending_requests[:self.batch_size]
                    self.pending_requests = self.pending_requests[self.batch_size:]
                    
                if not batch:
                    continue
                    
                # Process batch
                try:
                    # Combine batch inputs
                    batch_args = []
                    batch_kwargs = {}
                    request_futures = []
                    
                    for request_id, args, kwargs, future in batch:
                        batch_args.append(args)
                        request_futures.append(future)
                        
                        # Merge kwargs (assuming compatible)
                        for k, v in kwargs.items():
                            if k not in batch_kwargs:
                                batch_kwargs[k] = []
                            batch_kwargs[k].append(v)
                            
                    # Process batch
                    batch_results = await processor_func(batch_args, **batch_kwargs)
                    
                    # Distribute results
                    if isinstance(batch_results, (list, tuple)):
                        for future, result in zip(request_futures, batch_results):
                            future.set_result(result)
                    else:
                        # Single result for all
                        for future in request_futures:
                            future.set_result(batch_results)
                            
                except Exception as e:
                    # Set exception for all futures
                    for _, _, _, future in batch:
                        future.set_exception(e)
                        
        finally:
            self.processing = False


class PerformanceProfiler:
    """Performance profiling utility"""
    
    def __init__(self):
        self.profiles = {}
        self.lock = threading.Lock()
        
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling code blocks"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        # GPU memory if available
        gpu_memory_start = 0
        if torch.cuda.is_available():
            gpu_memory_start = torch.cuda.memory_allocated()
            
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            
            gpu_memory_end = 0
            if torch.cuda.is_available():
                gpu_memory_end = torch.cuda.memory_allocated()
                
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_usage = (start_cpu + end_cpu) / 2
            gpu_memory_delta = gpu_memory_end - gpu_memory_start
            
            # Store profile
            with self.lock:
                if name not in self.profiles:
                    self.profiles[name] = []
                    
                self.profiles[name].append({
                    'execution_time': execution_time,
                    'memory_delta': memory_delta,
                    'cpu_usage': cpu_usage,
                    'gpu_memory_delta': gpu_memory_delta,
                    'timestamp': time.time()
                })
                
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0.0
            
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling statistics"""
        with self.lock:
            if name:
                if name not in self.profiles:
                    return {}
                    
                data = self.profiles[name]
                return self._calculate_stats(data)
            else:
                stats = {}
                for profile_name, data in self.profiles.items():
                    stats[profile_name] = self._calculate_stats(data)
                return stats
                
    def _calculate_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from profile data"""
        if not data:
            return {}
            
        execution_times = [d['execution_time'] for d in data]
        memory_deltas = [d['memory_delta'] for d in data]
        cpu_usages = [d['cpu_usage'] for d in data]
        
        return {
            'count': len(data),
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'std_execution_time': np.std(execution_times),
            'avg_memory_delta': np.mean(memory_deltas),
            'avg_cpu_usage': np.mean(cpu_usages),
            'total_time': np.sum(execution_times)
        }
        
    def clear(self, name: Optional[str] = None):
        """Clear profiling data"""
        with self.lock:
            if name:
                if name in self.profiles:
                    del self.profiles[name]
            else:
                self.profiles.clear()


def memoize(max_size: int = 128, ttl: Optional[float] = None):
    """Memoization decorator with LRU cache and optional TTL"""
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size)
        cache_times = {} if ttl else None
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(hash((args, tuple(sorted(kwargs.items())))))
            
            # Check TTL if enabled
            if ttl and key in cache_times:
                if time.time() - cache_times[key] > ttl:
                    cache.cache.pop(key, None)
                    cache_times.pop(key, None)
                    
            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result
                
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            
            if ttl:
                cache_times[key] = time.time()
                
            return result
            
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        wrapper.cache_stats = cache.stats
        return wrapper
        
    return decorator


class MemoryManager:
    """Memory management utilities"""
    
    @staticmethod
    def cleanup_torch_cache():
        """Clean up PyTorch cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection"""
        gc.collect()
        
    @staticmethod
    def get_memory_stats() -> Dict[str, Any]:
        """Get detailed memory statistics"""
        stats = {}
        
        # System memory
        memory = psutil.virtual_memory()
        stats['system'] = {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        stats['process'] = {
            'rss': process_memory.rss,
            'vms': process_memory.vms,
            'percent': process.memory_percent()
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            stats['gpu'] = {}
            for i in range(torch.cuda.device_count()):
                stats['gpu'][f'device_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i),
                    'cached': torch.cuda.memory_reserved(i),
                    'max_allocated': torch.cuda.max_memory_allocated(i),
                    'max_cached': torch.cuda.max_memory_reserved(i)
                }
                
        return stats
        
    @contextmanager
    def memory_limit_context(self, max_memory_mb: int):
        """Context manager to monitor memory usage"""
        start_memory = self.get_memory_stats()['process']['rss']
        
        try:
            yield
        finally:
            current_memory = self.get_memory_stats()['process']['rss']
            memory_used_mb = (current_memory - start_memory) / 1024 / 1024
            
            if memory_used_mb > max_memory_mb:
                logger.warning(f"Memory usage exceeded limit: {memory_used_mb:.2f}MB > {max_memory_mb}MB")
                

class OptimizationConfig:
    """Configuration for performance optimizations"""
    
    def __init__(self):
        self.cache_enabled = True
        self.cache_size = 1000
        self.batch_processing_enabled = True
        self.batch_size = 32
        self.batch_timeout = 0.1
        self.memory_pool_enabled = True
        self.profiling_enabled = False
        self.auto_cleanup_enabled = True
        self.cleanup_interval = 300  # 5 minutes


class PerformanceOptimizer:
    """Main performance optimization manager"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = LRUCache(config.cache_size) if config.cache_enabled else None
        self.memory_pool = MemoryPool() if config.memory_pool_enabled else None
        self.profiler = PerformanceProfiler() if config.profiling_enabled else None
        self.batch_processor = BatchProcessor(config.batch_size, config.batch_timeout) if config.batch_processing_enabled else None
        self.memory_manager = MemoryManager()
        
        # Start auto cleanup if enabled
        if config.auto_cleanup_enabled:
            self._start_auto_cleanup()
            
    def _start_auto_cleanup(self):
        """Start automatic cleanup thread"""
        def cleanup_loop():
            while True:
                time.sleep(self.config.cleanup_interval)
                try:
                    self.cleanup()
                except Exception as e:
                    logger.error(f"Error in auto cleanup: {e}")
                    
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        
    def cleanup(self):
        """Perform cleanup operations"""
        logger.debug("Performing optimization cleanup")
        
        # Clear caches if they're getting large
        if self.cache and len(self.cache.cache) > self.config.cache_size * 0.9:
            # Clear oldest 25% of cache
            with self.cache.lock:
                items_to_remove = len(self.cache.cache) // 4
                for _ in range(items_to_remove):
                    if self.cache.cache:
                        self.cache.cache.popitem(last=False)
                        
        # Clear memory pool periodically
        if self.memory_pool:
            self.memory_pool.clear()
            
        # Force garbage collection
        self.memory_manager.force_garbage_collection()
        self.memory_manager.cleanup_torch_cache()
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'memory': self.memory_manager.get_memory_stats(),
            'timestamp': time.time()
        }
        
        if self.cache:
            stats['cache'] = self.cache.stats()
            
        if self.memory_pool:
            stats['memory_pool'] = self.memory_pool.stats()
            
        if self.profiler:
            stats['profiling'] = self.profiler.get_stats()
            
        return stats


# Global optimizer instance
_optimizer: Optional[PerformanceOptimizer] = None


def get_optimizer() -> Optional[PerformanceOptimizer]:
    """Get global performance optimizer"""
    return _optimizer


def initialize_optimizer(config: OptimizationConfig) -> PerformanceOptimizer:
    """Initialize global performance optimizer"""
    global _optimizer
    _optimizer = PerformanceOptimizer(config)
    return _optimizer
