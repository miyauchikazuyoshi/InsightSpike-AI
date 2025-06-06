#!/usr/bin/env python3
"""
InsightSpike-AI: Production Monitoring and Alerting System
Comprehensive monitoring for large-scale deployments

Features:
- Performance metrics collection
- Resource utilization monitoring  
- Error tracking and alerting
- Experiment progress monitoring
- GPU/CPU performance tracking
- Memory and storage monitoring
- Alert notifications (email, webhook, dashboard)
"""

import json
import logging
import psutil
import subprocess
import sys
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import warnings

# Optional imports with fallbacks
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("GPUtil not available - GPU monitoring disabled")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("requests not available - webhook alerts disabled")


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_metrics: Optional[Dict[str, Any]] = None
    process_metrics: Optional[Dict[str, Any]] = None
    custom_metrics: Optional[Dict[str, Any]] = None


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_path: str
    threshold: float
    operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    duration_seconds: int
    severity: str  # 'info', 'warning', 'error', 'critical'
    notification_channels: List[str]
    enabled: bool = True


class ProductionMonitor:
    """Comprehensive production monitoring system"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.metrics_history = deque(maxlen=self.config.get('history_size', 1000))
        self.alert_rules = []
        self.alert_states = {}
        self.running = False
        self.monitor_thread = None
        
        # Setup logging
        self._setup_logging()
        
        # Load alert rules
        self._load_alert_rules()
        
        # Initialize notification channels
        self._setup_notification_channels()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'monitor_interval': 10,  # seconds
            'history_size': 1000,
            'log_level': 'INFO',
            'enable_gpu_monitoring': True,
            'enable_process_monitoring': True,
            'metrics_export_interval': 300,  # 5 minutes
            'alert_evaluation_interval': 30,  # seconds
            'notification_channels': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'recipients': []
                },
                'webhook': {
                    'enabled': False,
                    'url': '',
                    'headers': {}
                },
                'file': {
                    'enabled': True,
                    'path': 'logs/alerts.log'
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging system"""
        log_level = getattr(logging, self.config['log_level'].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/production_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ProductionMonitor')
    
    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        # Default alert rules
        default_rules = [
            AlertRule(
                name="High CPU Usage",
                metric_path="cpu_percent",
                threshold=85.0,
                operator="gt",
                duration_seconds=300,
                severity="warning",
                notification_channels=["file", "webhook"]
            ),
            AlertRule(
                name="Critical CPU Usage",
                metric_path="cpu_percent", 
                threshold=95.0,
                operator="gt",
                duration_seconds=60,
                severity="critical",
                notification_channels=["file", "webhook", "email"]
            ),
            AlertRule(
                name="High Memory Usage",
                metric_path="memory_percent",
                threshold=90.0,
                operator="gt",
                duration_seconds=300,
                severity="warning",
                notification_channels=["file", "webhook"]
            ),
            AlertRule(
                name="Critical Memory Usage",
                metric_path="memory_percent",
                threshold=98.0,
                operator="gt",
                duration_seconds=60,
                severity="critical",
                notification_channels=["file", "webhook", "email"]
            ),
            AlertRule(
                name="High Disk Usage",
                metric_path="disk_percent",
                threshold=85.0,
                operator="gt", 
                duration_seconds=600,
                severity="warning",
                notification_channels=["file", "webhook"]
            ),
            AlertRule(
                name="GPU Memory High",
                metric_path="gpu_metrics.memory_percent",
                threshold=90.0,
                operator="gt",
                duration_seconds=300,
                severity="warning",
                notification_channels=["file", "webhook"]
            )
        ]
        
        # Load custom rules if available
        rules_file = Path('config/alert_rules.json')
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                custom_rules_data = json.load(f)
                custom_rules = [AlertRule(**rule) for rule in custom_rules_data]
                self.alert_rules = custom_rules
        else:
            self.alert_rules = default_rules
            
        self.logger.info(f"Loaded {len(self.alert_rules)} alert rules")
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        self.notification_channels = {}
        
        # File notification
        if self.config['notification_channels']['file']['enabled']:
            log_path = Path(self.config['notification_channels']['file']['path'])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.notification_channels['file'] = logging.FileHandler(log_path)
        
        # Email notification (placeholder)
        if self.config['notification_channels']['email']['enabled']:
            self.notification_channels['email'] = self._setup_email_notifications()
        
        # Webhook notification
        if self.config['notification_channels']['webhook']['enabled'] and REQUESTS_AVAILABLE:
            self.notification_channels['webhook'] = self._setup_webhook_notifications()
    
    def _setup_email_notifications(self):
        """Setup email notifications (requires SMTP configuration)"""
        # This would integrate with email libraries like smtplib
        # Placeholder for actual implementation
        self.logger.info("Email notifications configured")
        return lambda message: self.logger.info(f"EMAIL: {message}")
    
    def _setup_webhook_notifications(self):
        """Setup webhook notifications"""
        webhook_config = self.config['notification_channels']['webhook']
        
        def send_webhook(message):
            try:
                payload = {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'InsightSpike-AI-Monitor',
                    'message': message
                }
                response = requests.post(
                    webhook_config['url'],
                    json=payload,
                    headers=webhook_config.get('headers', {}),
                    timeout=10
                )
                response.raise_for_status()
                self.logger.info("Webhook notification sent successfully")
            except Exception as e:
                self.logger.error(f"Failed to send webhook notification: {e}")
        
        return send_webhook
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_metrics = None
        if GPU_AVAILABLE and self.config.get('enable_gpu_monitoring', True):
            gpu_metrics = self._collect_gpu_metrics()
        
        # Process metrics
        process_metrics = None
        if self.config.get('enable_process_monitoring', True):
            process_metrics = self._collect_process_metrics()
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent,
            disk_used_gb=disk.used / (1024**3),
            disk_total_gb=disk.total / (1024**3),
            gpu_metrics=gpu_metrics,
            process_metrics=process_metrics
        )
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU performance metrics"""
        try:
            gpus = GPUtil.getGPUs()
            gpu_data = []
            
            for gpu in gpus:
                gpu_info = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                }
                gpu_data.append(gpu_info)
            
            return {
                'gpu_count': len(gpus),
                'gpus': gpu_data,
                'total_memory_gb': sum(gpu.memoryTotal for gpu in gpus) / 1024,
                'total_memory_used_gb': sum(gpu.memoryUsed for gpu in gpus) / 1024
            }
        except Exception as e:
            self.logger.error(f"Failed to collect GPU metrics: {e}")
            return None
    
    def _collect_process_metrics(self) -> Dict[str, Any]:
        """Collect process-specific metrics"""
        try:
            current_process = psutil.Process()
            
            # Find Python processes (InsightSpike related)
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = proc.cmdline()
                        if any('insightspike' in arg.lower() for arg in cmdline):
                            python_processes.append({
                                'pid': proc.info['pid'],
                                'cpu_percent': proc.info['cpu_percent'],
                                'memory_mb': proc.info['memory_info'].rss / (1024*1024),
                                'cmdline': ' '.join(cmdline[:3])  # First 3 args
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'current_process': {
                    'pid': current_process.pid,
                    'cpu_percent': current_process.cpu_percent(),
                    'memory_mb': current_process.memory_info().rss / (1024*1024),
                    'threads': current_process.num_threads()
                },
                'insightspike_processes': python_processes,
                'total_processes': len(python_processes)
            }
        except Exception as e:
            self.logger.error(f"Failed to collect process metrics: {e}")
            return None
    
    def evaluate_alerts(self, metrics: PerformanceMetrics):
        """Evaluate alert rules against current metrics"""
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            # Extract metric value
            metric_value = self._get_metric_value(metrics, rule.metric_path)
            if metric_value is None:
                continue
            
            # Evaluate condition
            condition_met = self._evaluate_condition(metric_value, rule.threshold, rule.operator)
            
            # Track alert state
            rule_key = rule.name
            if rule_key not in self.alert_states:
                self.alert_states[rule_key] = {
                    'triggered_at': None,
                    'last_notified': None,
                    'notification_count': 0
                }
            
            state = self.alert_states[rule_key]
            
            if condition_met:
                if state['triggered_at'] is None:
                    # Alert just triggered
                    state['triggered_at'] = current_time
                else:
                    # Check if duration threshold met
                    duration = (current_time - state['triggered_at']).total_seconds()
                    if duration >= rule.duration_seconds:
                        # Send notification if not recently sent
                        should_notify = (
                            state['last_notified'] is None or
                            (current_time - state['last_notified']).total_seconds() > 300  # 5 min cooldown
                        )
                        
                        if should_notify:
                            self._send_alert(rule, metric_value, metrics)
                            state['last_notified'] = current_time
                            state['notification_count'] += 1
            else:
                # Condition not met, reset state
                if state['triggered_at'] is not None:
                    self.logger.info(f"Alert resolved: {rule.name}")
                    state['triggered_at'] = None
    
    def _get_metric_value(self, metrics: PerformanceMetrics, path: str) -> Optional[float]:
        """Extract metric value using dot notation path"""
        try:
            value = metrics
            for part in path.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return float(value) if value is not None else None
        except (AttributeError, ValueError, TypeError):
            return None
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate alert condition"""
        operators = {
            'gt': lambda v, t: v > t,
            'gte': lambda v, t: v >= t,
            'lt': lambda v, t: v < t,
            'lte': lambda v, t: v <= t,
            'eq': lambda v, t: abs(v - t) < 0.001
        }
        
        return operators.get(operator, lambda v, t: False)(value, threshold)
    
    def _send_alert(self, rule: AlertRule, value: float, metrics: PerformanceMetrics):
        """Send alert notification"""
        message = f"ALERT [{rule.severity.upper()}]: {rule.name} - Value: {value:.2f} (Threshold: {rule.threshold})"
        
        self.logger.warning(message)
        
        # Send to configured channels
        for channel in rule.notification_channels:
            if channel in self.notification_channels:
                try:
                    self.notification_channels[channel](message)
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel}: {e}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.running:
            self.logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("Production monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_export = datetime.now()
        last_alert_eval = datetime.now()
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Collect metrics
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Evaluate alerts
                if (current_time - last_alert_eval).total_seconds() >= self.config['alert_evaluation_interval']:
                    self.evaluate_alerts(metrics)
                    last_alert_eval = current_time
                
                # Export metrics
                if (current_time - last_export).total_seconds() >= self.config['metrics_export_interval']:
                    self._export_metrics()
                    last_export = current_time
                
                # Sleep until next interval
                time.sleep(self.config['monitor_interval'])
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config['monitor_interval'])
    
    def _export_metrics(self):
        """Export metrics to file"""
        try:
            metrics_file = Path('logs/performance_metrics.jsonl')
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Export recent metrics
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 entries
            
            with open(metrics_file, 'a') as f:
                for metrics in recent_metrics:
                    metrics_dict = asdict(metrics)
                    # Convert datetime to string
                    metrics_dict['timestamp'] = metrics.timestamp.isoformat()
                    f.write(json.dumps(metrics_dict) + '\n')
            
            self.logger.debug(f"Exported {len(recent_metrics)} metrics entries")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics collected yet"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate health score
        health_score = self._calculate_health_score(latest_metrics)
        
        # Count active alerts
        active_alerts = sum(
            1 for state in self.alert_states.values() 
            if state['triggered_at'] is not None
        )
        
        return {
            "status": "running" if self.running else "stopped",
            "timestamp": latest_metrics.timestamp.isoformat(),
            "health_score": health_score,
            "active_alerts": active_alerts,
            "metrics_collected": len(self.metrics_history),
            "latest_metrics": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "disk_percent": latest_metrics.disk_percent,
                "gpu_available": latest_metrics.gpu_metrics is not None
            }
        }
    
    def _calculate_health_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall system health score (0-100)"""
        scores = []
        
        # CPU health (inverted - lower usage is better)
        cpu_score = max(0, 100 - metrics.cpu_percent)
        scores.append(cpu_score)
        
        # Memory health
        memory_score = max(0, 100 - metrics.memory_percent)
        scores.append(memory_score)
        
        # Disk health
        disk_score = max(0, 100 - metrics.disk_percent)
        scores.append(disk_score)
        
        # GPU health (if available)
        if metrics.gpu_metrics:
            avg_gpu_memory = sum(
                gpu['memory_percent'] for gpu in metrics.gpu_metrics['gpus']
            ) / len(metrics.gpu_metrics['gpus'])
            gpu_score = max(0, 100 - avg_gpu_memory)
            scores.append(gpu_score)
        
        return sum(scores) / len(scores)


def main():
    """Main CLI interface for production monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='InsightSpike-AI Production Monitor')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--duration', type=int, default=300, help='Monitoring duration in seconds (default: 300)')
    parser.add_argument('--status', action='store_true', help='Show current system status')
    parser.add_argument('--alerts', action='store_true', help='Show active alerts')
    parser.add_argument('--metrics', action='store_true', help='Show latest metrics')
    parser.add_argument('--dashboard', action='store_true', help='Start web dashboard')
    
    args = parser.parse_args()
    
    try:
        # Initialize monitor
        config_file = args.config if args.config else None
        monitor = ProductionMonitor(config_file=config_file)
        
        if args.status:
            print("📊 InsightSpike-AI System Status")
            print("=" * 50)
            status = monitor.get_current_status()
            print(f"Health Score: {status.get('health_score', 0):.1f}/100")
            print(f"CPU Usage: {status.get('cpu_percent', 0):.1f}%")
            print(f"Memory Usage: {status.get('memory_percent', 0):.1f}%")
            print(f"Active Alerts: {status.get('active_alerts', 0)}")
            print(f"Uptime: {status.get('uptime', 'Unknown')}")
            return 0
            
        if args.alerts:
            print("🚨 Active System Alerts")
            print("=" * 50)
            alerts = monitor.get_active_alerts()
            if not alerts:
                print("✅ No active alerts")
            else:
                for alert in alerts:
                    print(f"[{alert.severity}] {alert.message}")
            return 0
            
        if args.metrics:
            print("📈 Latest Performance Metrics")
            print("=" * 50)
            metrics = monitor.get_latest_metrics()
            if metrics:
                print(f"Timestamp: {metrics.get('timestamp', 'Unknown')}")
                print(f"CPU: {metrics.get('cpu_percent', 0):.1f}%")
                print(f"Memory: {metrics.get('memory_percent', 0):.1f}%") 
                print(f"Disk: {metrics.get('disk_percent', 0):.1f}%")
                print(f"Network I/O: {metrics.get('network_io', 'N/A')}")
            else:
                print("No metrics available")
            return 0
            
        if args.dashboard:
            try:
                from monitoring.performance_dashboard import PerformanceDashboard
                dashboard = PerformanceDashboard()
                print("🚀 Starting production dashboard...")
                dashboard.run()
                return 0
            except ImportError:
                print("❌ Dashboard not available - missing dependencies")
                return 1
        
        # Default: Start monitoring
        print("🚀 Starting InsightSpike-AI Production Monitor")
        print(f"⏱️ Duration: {args.duration} seconds")
        print(f"📊 Monitor interval: {monitor.config['monitor_interval']} seconds")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='InsightSpike-AI Production Monitor')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--status', action='store_true', help='Show current system status')
    
    args = parser.parse_args()
    
    try:
        # Initialize monitor
        monitor = ProductionMonitor(config_file=args.config)
        
        if args.status:
            print("📊 InsightSpike-AI System Status")
            print("=" * 50)
            status = monitor.get_current_status()
            print(f"Health Score: {status.get('health_score', 0):.1f}/100")
            print(f"CPU Usage: {status.get('cpu_percent', 0):.1f}%")
            print(f"Memory Usage: {status.get('memory_percent', 0):.1f}%")
            print(f"Active Alerts: {status.get('active_alerts', 0)}")
            print("✅ System monitoring ready")
        else:
            print("🚀 InsightSpike-AI Production Monitor")
            print("Use --status to check system status")
    except Exception as e:
        print(f"❌ Monitor error: {e}")
        exit(1)
            
        if args.dashboard:
            try:
                from monitoring.performance_dashboard import PerformanceDashboard
                dashboard = PerformanceDashboard()
                print("🚀 Starting production dashboard...")
                dashboard.run()
                return 0
            except ImportError:
                print("❌ Dashboard not available - missing dependencies")
                return 1
        
        # Default: Start monitoring
        print("🚀 Starting InsightSpike-AI Production Monitor")
        print(f"⏱️ Duration: {args.duration} seconds")
        print(f"📊 Monitor interval: {monitor.config['monitor_interval']} seconds")
        print("Press Ctrl+C to stop monitoring\n")
        
        monitor.start_monitoring()
        
        # Monitor for specified duration
        time.sleep(args.duration)
        
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        return 1
    finally:
        try:
            monitor.stop_monitoring()
            
            # Final status
            status = monitor.get_current_status()
            print(f"\n📋 Final Status:")
            print(f"   Health Score: {status.get('health_score', 0):.1f}/100")
            print(f"   Active Alerts: {status.get('active_alerts', 0)}")
            print(f"   Metrics Collected: {status.get('metrics_collected', 0)}")
        except:
            pass
    
    return 0


if __name__ == "__main__":
    exit(main())
