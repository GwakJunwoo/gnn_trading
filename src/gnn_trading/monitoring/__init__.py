"""
gnn_trading.monitoring.alerts
=============================

Production monitoring and alerting system for GNN Trading System
Provides real-time monitoring, alerting, and health checks
"""

import asyncio
import logging
import smtplib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
import threading
import json

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Configuration for monitoring alerts"""
    name: str
    condition: str  # e.g., "prediction_latency > 1000"
    threshold: float
    window_minutes: int = 5
    min_samples: int = 10
    severity: str = "warning"  # warning, critical
    description: str = ""
    enabled: bool = True
    cooldown_minutes: int = 30
    
    
@dataclass 
class AlertConfig:
    """Configuration for alert system"""
    enabled: bool = True
    email_enabled: bool = False
    smtp_server: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_email: str = "alerts@gnn-trading.com"
    to_emails: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None


class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._setup_metrics()
        self.custom_metrics = {}
        self._lock = threading.Lock()
        
    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        # Prediction metrics
        self.prediction_counter = Counter(
            'gnn_predictions_total', 
            'Total number of predictions made',
            ['model_name', 'status'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'gnn_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'gnn_model_accuracy',
            'Current model accuracy',
            ['model_name'],
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'gnn_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'gnn_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'gnn_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Data quality metrics
        self.data_quality_score = Gauge(
            'gnn_data_quality_score',
            'Data quality score (0-1)',
            ['data_source'],
            registry=self.registry
        )
        
        self.data_processing_errors = Counter(
            'gnn_data_processing_errors_total',
            'Total data processing errors',
            ['error_type'],
            registry=self.registry
        )
        
    def record_prediction(self, model_name: str, latency: float, success: bool = True):
        """Record a prediction event"""
        status = "success" if success else "error"
        self.prediction_counter.labels(model_name=model_name, status=status).inc()
        if success:
            self.prediction_latency.labels(model_name=model_name).observe(latency)
            
    def update_model_accuracy(self, model_name: str, accuracy: float):
        """Update model accuracy metric"""
        self.model_accuracy.labels(model_name=model_name).set(accuracy)
        
    def update_system_metrics(self, connections: int, memory_bytes: int, cpu_percent: float):
        """Update system-level metrics"""
        self.active_connections.set(connections)
        self.memory_usage.set(memory_bytes)
        self.cpu_usage.set(cpu_percent)
        
    def update_data_quality(self, data_source: str, quality_score: float):
        """Update data quality metrics"""
        self.data_quality_score.labels(data_source=data_source).set(quality_score)
        
    def record_data_error(self, error_type: str):
        """Record a data processing error"""
        self.data_processing_errors.labels(error_type=error_type).inc()
        
    def add_custom_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add custom metric"""
        with self._lock:
            key = f"{name}_{str(labels) if labels else ''}"
            if key not in self.custom_metrics:
                self.custom_metrics[key] = Gauge(
                    f'gnn_custom_{name}',
                    f'Custom metric: {name}',
                    list(labels.keys()) if labels else [],
                    registry=self.registry
                )
            
            if labels:
                self.custom_metrics[key].labels(**labels).set(value)
            else:
                self.custom_metrics[key].set(value)
                
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')


class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.checks = {}
        self.last_results = {}
        self._lock = threading.Lock()
        
    def register_check(self, name: str, check_func: Callable[[], bool], 
                      description: str = "", timeout_seconds: int = 30):
        """Register a health check"""
        self.checks[name] = {
            'func': check_func,
            'description': description,
            'timeout': timeout_seconds
        }
        
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check"""
        if name not in self.checks:
            return {'status': 'error', 'message': f'Check {name} not found'}
            
        check = self.checks[name]
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = asyncio.wait_for(
                asyncio.to_thread(check['func']),
                timeout=check['timeout']
            )
            
            if asyncio.iscoroutine(result):
                result = await result
            else:
                result = check['func']()
            
            duration = time.time() - start_time
            
            status = 'healthy' if result else 'unhealthy'
            return {
                'status': status,
                'duration_ms': duration * 1000,
                'timestamp': datetime.utcnow().isoformat(),
                'description': check['description']
            }
            
        except asyncio.TimeoutError:
            return {
                'status': 'timeout',
                'duration_ms': check['timeout'] * 1000,
                'timestamp': datetime.utcnow().isoformat(),
                'message': f'Check timed out after {check["timeout"]}s'
            }
        except Exception as e:
            return {
                'status': 'error',
                'duration_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.utcnow().isoformat(),
                'message': str(e)
            }
            
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        overall_status = 'healthy'
        
        for name in self.checks:
            result = await self.run_check(name)
            results[name] = result
            
            if result['status'] in ['unhealthy', 'error', 'timeout']:
                overall_status = 'unhealthy'
                
        return {
            'overall_status': overall_status,
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def get_last_results(self) -> Dict[str, Any]:
        """Get last health check results"""
        with self._lock:
            return self.last_results.copy()


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, datetime] = {}
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
        
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            
    def add_metric_value(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add metric value for monitoring"""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        with self._lock:
            self.metrics_buffer[metric_name].append((timestamp, value))
            
    def check_alerts(self):
        """Check all alert rules against current metrics"""
        current_time = datetime.utcnow()
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
                
            # Check if we're in cooldown period
            if rule_name in self.active_alerts:
                cooldown_end = self.active_alerts[rule_name] + timedelta(minutes=rule.cooldown_minutes)
                if current_time < cooldown_end:
                    continue
                    
            # Evaluate alert condition
            if self._evaluate_rule(rule, current_time):
                self._trigger_alert(rule, current_time)
            elif rule_name in self.active_alerts:
                # Clear alert if condition no longer met
                self._clear_alert(rule_name, current_time)
                
    def _evaluate_rule(self, rule: AlertRule, current_time: datetime) -> bool:
        """Evaluate if alert rule condition is met"""
        # Extract metric name from condition
        metric_name = rule.condition.split()[0]
        
        if metric_name not in self.metrics_buffer:
            return False
            
        # Get recent values within window
        window_start = current_time - timedelta(minutes=rule.window_minutes)
        recent_values = [
            value for timestamp, value in self.metrics_buffer[metric_name]
            if timestamp >= window_start
        ]
        
        if len(recent_values) < rule.min_samples:
            return False
            
        # Evaluate condition
        try:
            if ">" in rule.condition:
                return np.mean(recent_values) > rule.threshold
            elif "<" in rule.condition:
                return np.mean(recent_values) < rule.threshold
            elif "==" in rule.condition:
                return np.mean(recent_values) == rule.threshold
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
            
        return False
        
    def _trigger_alert(self, rule: AlertRule, timestamp: datetime):
        """Trigger an alert"""
        alert_data = {
            'rule_name': rule.name,
            'severity': rule.severity,
            'description': rule.description,
            'condition': rule.condition,
            'threshold': rule.threshold,
            'timestamp': timestamp.isoformat(),
            'status': 'triggered'
        }
        
        # Record alert
        with self._lock:
            self.active_alerts[rule.name] = timestamp
            self.alert_history[rule.name].append(alert_data)
            
        # Send notifications
        self._send_notifications(alert_data)
        
        logger.warning(f"Alert triggered: {rule.name} - {rule.description}")
        
    def _clear_alert(self, rule_name: str, timestamp: datetime):
        """Clear an active alert"""
        alert_data = {
            'rule_name': rule_name,
            'timestamp': timestamp.isoformat(),
            'status': 'cleared'
        }
        
        with self._lock:
            if rule_name in self.active_alerts:
                del self.active_alerts[rule_name]
            self.alert_history[rule_name].append(alert_data)
            
        logger.info(f"Alert cleared: {rule_name}")
        
    def _send_notifications(self, alert_data: Dict[str, Any]):
        """Send alert notifications"""
        if not self.config.enabled:
            return
            
        # Send email notification
        if self.config.email_enabled and self.config.to_emails:
            self._send_email_alert(alert_data)
            
        # Send webhook notification
        if self.config.webhook_url:
            self._send_webhook_alert(alert_data)
            
        # Send Slack notification
        if self.config.slack_webhook:
            self._send_slack_alert(alert_data)
            
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send email alert notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.from_email
            msg['To'] = ', '.join(self.config.to_emails)
            msg['Subject'] = f"GNN Trading Alert: {alert_data['rule_name']}"
            
            body = f"""
            Alert Triggered: {alert_data['rule_name']}
            Severity: {alert_data['severity']}
            Condition: {alert_data['condition']}
            Threshold: {alert_data['threshold']}
            Time: {alert_data['timestamp']}
            Description: {alert_data['description']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            if self.config.smtp_user and self.config.smtp_password:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert_data['rule_name']}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            
    def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Send webhook alert notification"""
        try:
            import requests
            response = requests.post(
                self.config.webhook_url,
                json=alert_data,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Webhook alert sent for {alert_data['rule_name']}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            
    def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send Slack alert notification"""
        try:
            import requests
            
            color = "danger" if alert_data['severity'] == 'critical' else "warning"
            slack_data = {
                "attachments": [{
                    "color": color,
                    "title": f"GNN Trading Alert: {alert_data['rule_name']}",
                    "fields": [
                        {"title": "Severity", "value": alert_data['severity'], "short": True},
                        {"title": "Condition", "value": alert_data['condition'], "short": True},
                        {"title": "Description", "value": alert_data['description'], "short": False}
                    ],
                    "timestamp": alert_data['timestamp']
                }]
            }
            
            response = requests.post(
                self.config.slack_webhook,
                json=slack_data,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Slack alert sent for {alert_data['rule_name']}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        with self._lock:
            return [
                {
                    'rule_name': rule_name,
                    'triggered_at': timestamp.isoformat(),
                    'duration_minutes': (datetime.utcnow() - timestamp).total_seconds() / 60
                }
                for rule_name, timestamp in self.active_alerts.items()
            ]
            
    def get_alert_history(self, rule_name: Optional[str] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        with self._lock:
            if rule_name:
                if rule_name in self.alert_history:
                    return list(self.alert_history[rule_name])[-limit:]
                return []
            else:
                all_alerts = []
                for alerts in self.alert_history.values():
                    all_alerts.extend(alerts)
                # Sort by timestamp and return recent
                all_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
                return all_alerts[:limit]


class MonitoringSystem:
    """Complete monitoring system combining metrics, health checks, and alerts"""
    
    def __init__(self, alert_config: AlertConfig):
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager(alert_config)
        self.monitoring_thread = None
        self.running = False
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
    def _setup_default_health_checks(self):
        """Setup default system health checks"""
        def check_api_health():
            # Simple health check - always returns True for now
            # In production, this would check actual API endpoint
            return True
            
        def check_model_health():
            # Check if models are loaded and responding
            return True
            
        def check_data_pipeline_health():
            # Check data pipeline connectivity
            return True
            
        self.health_checker.register_check(
            "api", check_api_health, "API endpoint health"
        )
        self.health_checker.register_check(
            "models", check_model_health, "Model availability and health"
        )
        self.health_checker.register_check(
            "data_pipeline", check_data_pipeline_health, "Data pipeline connectivity"
        )
        
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        # High prediction latency alert
        self.alert_manager.add_rule(AlertRule(
            name="high_prediction_latency",
            condition="prediction_latency > 1.0",
            threshold=1.0,
            window_minutes=5,
            severity="warning",
            description="Prediction latency is too high"
        ))
        
        # Low model accuracy alert
        self.alert_manager.add_rule(AlertRule(
            name="low_model_accuracy",
            condition="model_accuracy < 0.7",
            threshold=0.7,
            window_minutes=10,
            severity="critical",
            description="Model accuracy has dropped below acceptable threshold"
        ))
        
        # High CPU usage alert
        self.alert_manager.add_rule(AlertRule(
            name="high_cpu_usage",
            condition="cpu_usage > 80",
            threshold=80,
            window_minutes=5,
            severity="warning",
            description="CPU usage is very high"
        ))
        
        # High memory usage alert
        self.alert_manager.add_rule(AlertRule(
            name="high_memory_usage",
            condition="memory_usage > 8589934592",  # 8GB in bytes
            threshold=8589934592,
            window_minutes=5,
            severity="critical",
            description="Memory usage is critically high"
        ))
        
    def start_monitoring(self, check_interval_seconds: int = 60):
        """Start background monitoring"""
        if self.running:
            logger.warning("Monitoring is already running")
            return
            
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Monitoring system started")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Monitoring system stopped")
        
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts()
                
                # Sleep until next check
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
                
    def _update_system_metrics(self):
        """Update system-level metrics"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Update metrics
            self.metrics.update_system_metrics(
                connections=0,  # TODO: Get actual connection count
                memory_bytes=memory.used,
                cpu_percent=cpu_percent
            )
            
            # Add to alert monitoring
            self.alert_manager.add_metric_value("cpu_usage", cpu_percent)
            self.alert_manager.add_metric_value("memory_usage", memory.used)
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            
    def record_prediction_metrics(self, model_name: str, latency: float, 
                                accuracy: Optional[float] = None, success: bool = True):
        """Record prediction metrics"""
        # Update Prometheus metrics
        self.metrics.record_prediction(model_name, latency, success)
        if accuracy is not None:
            self.metrics.update_model_accuracy(model_name, accuracy)
            
        # Add to alert monitoring
        self.alert_manager.add_metric_value("prediction_latency", latency)
        if accuracy is not None:
            self.alert_manager.add_metric_value("model_accuracy", accuracy)
            
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "health_status": self.health_checker.get_last_results(),
            "active_alerts": self.alert_manager.get_active_alerts(),
            "recent_alerts": self.alert_manager.get_alert_history(limit=50),
            "metrics_endpoint": "/metrics",  # Prometheus metrics endpoint
            "timestamp": datetime.utcnow().isoformat()
        }


# Global monitoring instance
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> Optional[MonitoringSystem]:
    """Get global monitoring system instance"""
    return _monitoring_system


def initialize_monitoring(alert_config: AlertConfig) -> MonitoringSystem:
    """Initialize global monitoring system"""
    global _monitoring_system
    _monitoring_system = MonitoringSystem(alert_config)
    return _monitoring_system
