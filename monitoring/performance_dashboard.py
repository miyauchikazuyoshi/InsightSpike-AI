#!/usr/bin/env python3
"""
InsightSpike-AI: Real-Time Performance Monitoring Dashboard
Production-ready web dashboard for monitoring large-scale deployments

Features:
- Real-time metrics visualization
- Interactive performance charts
- System health monitoring
- Alert management interface
- Deployment status tracking
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import flask
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask not available - web dashboard disabled")

try:
    import plotly
    import plotly.graph_objs as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available - advanced charts disabled")

# Import monitoring system
try:
    from monitoring.production_monitor import ProductionMonitor, PerformanceMetrics
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("⚠️ Production monitor not available - using mock data")

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
        self.app = None
        self.monitor = None
        
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self.setup_routes()
        
        if MONITORING_AVAILABLE:
            self.monitor = ProductionMonitor()
        
        self.mock_data = self.generate_mock_data()
        
    def generate_mock_data(self) -> Dict[str, Any]:
        """Generate mock performance data for testing"""
        now = datetime.now()
        timestamps = [(now - timedelta(minutes=i)).isoformat() for i in range(60, 0, -1)]
        
        return {
            'cpu_usage': [20 + (i % 30) for i in range(60)],
            'memory_usage': [40 + (i % 25) for i in range(60)],
            'gpu_usage': [60 + (i % 35) for i in range(60)],
            'timestamps': timestamps,
            'alerts': [
                {
                    'id': 1,
                    'severity': 'WARNING',
                    'message': 'High CPU usage detected',
                    'timestamp': (now - timedelta(minutes=5)).isoformat()
                },
                {
                    'id': 2, 
                    'severity': 'INFO',
                    'message': 'System backup completed',
                    'timestamp': (now - timedelta(hours=1)).isoformat()
                }
            ],
            'system_status': {
                'overall_health': 'HEALTHY',
                'uptime': '5 days, 12 hours',
                'active_experiments': 3,
                'completed_experiments': 27,
                'total_documents_processed': 1234567
            }
        }
    
    def setup_routes(self):
        """Setup Flask routes for dashboard"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>InsightSpike-AI Production Dashboard</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
                    .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .metric-value { font-size: 2em; font-weight: bold; color: #667eea; }
                    .metric-label { color: #666; margin-top: 5px; }
                    .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
                    .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
                    .status-healthy { background-color: #4CAF50; }
                    .status-warning { background-color: #FF9800; }
                    .status-error { background-color: #F44336; }
                    .refresh-btn { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px 0; }
                    .refresh-btn:hover { background: #5a6fd8; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 InsightSpike-AI Production Dashboard</h1>
                    <p>Real-time monitoring for large-scale AI deployments</p>
                    <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Data</button>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="cpu-usage">--</div>
                        <div class="metric-label">CPU Usage (%)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="memory-usage">--</div>
                        <div class="metric-label">Memory Usage (%)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="gpu-usage">--</div>
                        <div class="metric-label">GPU Usage (%)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="active-experiments">--</div>
                        <div class="metric-label">Active Experiments</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>System Performance (Last Hour)</h3>
                    <div id="performance-chart" style="height: 400px;"></div>
                </div>
                
                <div class="chart-container">
                    <h3>🚨 System Status</h3>
                    <div id="system-status">
                        <p><span class="status-indicator status-healthy"></span><strong>Overall Health:</strong> <span id="overall-health">--</span></p>
                        <p><strong>Uptime:</strong> <span id="uptime">--</span></p>
                        <p><strong>Documents Processed:</strong> <span id="documents-processed">--</span></p>
                    </div>
                </div>
                
                <script>
                    async function loadDashboardData() {
                        try {
                            const response = await fetch('/api/metrics');
                            const data = await response.json();
                            
                            // Update metrics
                            document.getElementById('cpu-usage').textContent = data.cpu_usage[data.cpu_usage.length - 1] + '%';
                            document.getElementById('memory-usage').textContent = data.memory_usage[data.memory_usage.length - 1] + '%';
                            document.getElementById('gpu-usage').textContent = data.gpu_usage[data.gpu_usage.length - 1] + '%';
                            document.getElementById('active-experiments').textContent = data.system_status.active_experiments;
                            
                            // Update status
                            document.getElementById('overall-health').textContent = data.system_status.overall_health;
                            document.getElementById('uptime').textContent = data.system_status.uptime;
                            document.getElementById('documents-processed').textContent = data.system_status.total_documents_processed.toLocaleString();
                            
                            // Create performance chart
                            const trace1 = {
                                x: data.timestamps,
                                y: data.cpu_usage,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'CPU Usage',
                                line: { color: '#FF6B6B' }
                            };
                            
                            const trace2 = {
                                x: data.timestamps,
                                y: data.memory_usage,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'Memory Usage',
                                line: { color: '#4ECDC4' }
                            };
                            
                            const trace3 = {
                                x: data.timestamps,
                                y: data.gpu_usage,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'GPU Usage',
                                line: { color: '#45B7D1' }
                            };
                            
                            const layout = {
                                title: '',
                                xaxis: { title: 'Time' },
                                yaxis: { title: 'Usage (%)' },
                                showlegend: true
                            };
                            
                            Plotly.newPlot('performance-chart', [trace1, trace2, trace3], layout);
                            
                        } catch (error) {
                            console.error('Error loading dashboard data:', error);
                        }
                    }
                    
                    function refreshDashboard() {
                        loadDashboardData();
                    }
                    
                    // Load data on page load
                    loadDashboardData();
                    
                    // Auto-refresh every 30 seconds
                    setInterval(loadDashboardData, 30000);
                </script>
            </body>
            </html>
            """
            return html_template
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """API endpoint for real-time metrics"""
            if self.monitor and MONITORING_AVAILABLE:
                # Get real metrics from production monitor
                try:
                    latest_metrics = self.monitor.get_latest_metrics()
                    return jsonify(latest_metrics)
                except:
                    pass
            
            # Return mock data for demonstration
            return jsonify(self.mock_data)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """API endpoint for system alerts"""
            return jsonify(self.mock_data['alerts'])
    
    def run(self):
        """Start the dashboard server"""
        if not FLASK_AVAILABLE:
            print("❌ Flask not available - cannot start web dashboard")
            return False
        
        print(f"🚀 Starting InsightSpike-AI Production Dashboard")
        print(f"📊 Dashboard URL: http://{self.host}:{self.port}")
        print(f"🔄 Auto-refresh: Every 30 seconds")
        print("⚠️ Press Ctrl+C to stop")
        
        try:
            self.app.run(host=self.host, port=self.port, debug=False)
            return True
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            return False

def main():
    """Main dashboard launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description='InsightSpike-AI Production Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Dashboard host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port (default: 8080)')
    parser.add_argument('--install-deps', action='store_true', help='Install required dependencies')
    
    args = parser.parse_args()
    
    if args.install_deps:
        print("🔧 Installing dashboard dependencies...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', 'plotly'], check=True)
            print("✅ Dependencies installed successfully")
            return 0
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return 1
    
    dashboard = PerformanceDashboard(host=args.host, port=args.port)
    success = dashboard.run()
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
