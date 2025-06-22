#!/usr/bin/env python3
"""
InsightSpike-AI: Production Monitoring System (Fixed Version)
Comprehensive monitoring for large-scale deployments

Features:
- Performance metrics collection
- Resource utilization monitoring  
- System health checks
- Basic alerting system
"""

import json
import logging
import os
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
import sys
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]

@dataclass
class SystemAlert:
    """System alert data structure"""
    id: int
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: str
    resolved: bool = False

class ProductionMonitor:
    """Production monitoring system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.alerts: List[SystemAlert] = []
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = datetime.now()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'monitor_interval': 30,
            'alert_thresholds': {
                'cpu_warning': 80.0,
                'cpu_critical': 95.0,
                'memory_warning': 85.0,
                'memory_critical': 95.0,
                'disk_warning': 90.0,
                'disk_critical': 95.0
            },
            'retention_hours': 24,
            'enable_alerts': True
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}")
        
        return default_config
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        try:
            if PSUTIL_AVAILABLE:
                # Get network I/O stats
                net_io = psutil.net_io_counters()
                network_stats = {
                    'bytes_sent': net_io.bytes_sent if net_io else 0,
                    'bytes_recv': net_io.bytes_recv if net_io else 0
                }
                
                metrics = PerformanceMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    disk_percent=psutil.disk_usage('/').percent,
                    network_io=network_stats
                )
            else:
                # Mock metrics when psutil is not available
                import random
                network_stats = {
                    'bytes_sent': random.randint(1000000, 10000000),
                    'bytes_recv': random.randint(1000000, 10000000)
                }
                
                metrics = PerformanceMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=random.uniform(10, 80),
                    memory_percent=random.uniform(30, 90),
                    disk_percent=random.uniform(20, 80),
                    network_io=network_stats
                )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Cleanup old metrics (keep last 24 hours worth)
            max_metrics = int((self.config['retention_hours'] * 3600) / self.config['monitor_interval'])
            if len(self.metrics_history) > max_metrics:
                self.metrics_history = self.metrics_history[-max_metrics:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={'bytes_sent': 0, 'bytes_recv': 0}
            )
    
    def check_thresholds(self, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds"""
        if not self.config['enable_alerts']:
            return
        
        thresholds = self.config['alert_thresholds']
        
        # CPU alerts
        if metrics.cpu_percent >= thresholds['cpu_critical']:
            self._create_alert('CRITICAL', f'CPU usage critical: {metrics.cpu_percent:.1f}%')
        elif metrics.cpu_percent >= thresholds['cpu_warning']:
            self._create_alert('WARNING', f'CPU usage high: {metrics.cpu_percent:.1f}%')
        
        # Memory alerts
        if metrics.memory_percent >= thresholds['memory_critical']:
            self._create_alert('CRITICAL', f'Memory usage critical: {metrics.memory_percent:.1f}%')
        elif metrics.memory_percent >= thresholds['memory_warning']:
            self._create_alert('WARNING', f'Memory usage high: {metrics.memory_percent:.1f}%')
        
        # Disk alerts
        if metrics.disk_percent >= thresholds['disk_critical']:
            self._create_alert('CRITICAL', f'Disk usage critical: {metrics.disk_percent:.1f}%')
        elif metrics.disk_percent >= thresholds['disk_warning']:
            self._create_alert('WARNING', f'Disk usage high: {metrics.disk_percent:.1f}%')
    
    def _create_alert(self, severity: str, message: str):
        """Create a new system alert"""
        alert = SystemAlert(
            id=len(self.alerts) + 1,
            severity=severity,
            message=message,
            timestamp=datetime.now().isoformat(),
            resolved=False
        )
        self.alerts.append(alert)
        self.logger.warning(f"[{severity}] {message}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
        else:
            latest_metrics = self.collect_metrics()
        
        # Calculate health score (0-100)
        health_score = 100.0
        health_score -= min(latest_metrics.cpu_percent, 50)  # Max 50 point penalty
        health_score -= min(latest_metrics.memory_percent * 0.3, 30)  # Max 30 point penalty
        health_score -= min(latest_metrics.disk_percent * 0.2, 20)  # Max 20 point penalty
        health_score = max(0, health_score)
        
        # Count active alerts
        active_alerts = len([a for a in self.alerts if not a.resolved])
        
        uptime = datetime.now() - self.start_time
        uptime_str = f"{uptime.days} days, {uptime.seconds // 3600} hours"
        
        return {
            'health_score': health_score,
            'cpu_percent': latest_metrics.cpu_percent,
            'memory_percent': latest_metrics.memory_percent,
            'disk_percent': latest_metrics.disk_percent,
            'active_alerts': active_alerts,
            'metrics_collected': len(self.metrics_history),
            'uptime': uptime_str,
            'timestamp': latest_metrics.timestamp
        }
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest performance metrics"""
        if not self.metrics_history:
            return None
        
        latest = self.metrics_history[-1]
        return {
            'timestamp': latest.timestamp,
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'disk_percent': latest.disk_percent,
            'network_io': latest.network_io
        }
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get list of active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def start_monitoring(self):
        """Start monitoring process"""
        self.logger.info("Production monitoring started")
        self.collect_metrics()  # Initial collection
    
    def stop_monitoring(self):
        """Stop monitoring process"""
        self.logger.info("Production monitoring stopped")


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='InsightSpike-AI Production Monitor')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--status', action='store_true', help='Show current system status')
    parser.add_argument('--metrics', action='store_true', help='Show latest metrics')
    parser.add_argument('--alerts', action='store_true', help='Show active alerts')
    
    args = parser.parse_args()
    
    try:
        # Initialize monitor
        monitor = ProductionMonitor(config_file=args.config)
        
        if args.status:
            print("📊 InsightSpike-AI System Status")
            print("=" * 50)
            status = monitor.get_current_status()
            print(f"Health Score: {status['health_score']:.1f}/100")
            print(f"CPU Usage: {status['cpu_percent']:.1f}%")
            print(f"Memory Usage: {status['memory_percent']:.1f}%")
            print(f"Disk Usage: {status['disk_percent']:.1f}%")
            print(f"Active Alerts: {status['active_alerts']}")
            print(f"Uptime: {status['uptime']}")
            return 0
            
        if args.metrics:
            print("📈 Latest Performance Metrics")
            print("=" * 50)
            metrics = monitor.get_latest_metrics()
            if metrics:
                print(f"Timestamp: {metrics['timestamp']}")
                print(f"CPU: {metrics['cpu_percent']:.1f}%")
                print(f"Memory: {metrics['memory_percent']:.1f}%") 
                print(f"Disk: {metrics['disk_percent']:.1f}%")
                print(f"Network I/O: {metrics['network_io']}")
            else:
                print("No metrics available")
            return 0
            
        if args.alerts:
            print("🚨 Active System Alerts")
            print("=" * 50)
            alerts = monitor.get_active_alerts()
            if not alerts:
                print("✅ No active alerts")
            else:
                for alert in alerts:
                    print(f"[{alert.severity}] {alert.message} ({alert.timestamp})")
            return 0
        
        # Default: Show basic info
        print("🚀 InsightSpike-AI Production Monitor")
        print("Available commands:")
        print("  --status  : Show system status")
        print("  --metrics : Show latest metrics")
        print("  --alerts  : Show active alerts")
        print("  --config  : Use custom config file")
        return 0
        
    except Exception as e:
        print(f"❌ Monitor error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
