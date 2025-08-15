"""
Real-time Trading Monitoring Dashboard
=====================================

Live monitoring dashboard for AI trading bot
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import threading
from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils
import warnings
warnings.filterwarnings('ignore')


class TradingDashboard:
    """
    Real-time trading dashboard
    """
    
    def __init__(self, trading_manager, port=5000):
        self.trading_manager = trading_manager
        self.port = port
        
        # Dashboard data
        self.dashboard_data = {
            'last_update': None,
            'portfolio_metrics': {},
            'positions': [],
            'signals': [],
            'risk_metrics': {},
            'performance_chart': {},
            'alerts': []
        }
        
        # Flask app for web dashboard
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Data update thread
        self.update_thread = None
        self.is_running = False
        
        print(f"Trading Dashboard initialized on port {port}")
    
    def _setup_routes(self):
        """Setup Flask routes for web dashboard"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/dashboard_data')
        def get_dashboard_data():
            return jsonify(self.dashboard_data)
        
        @self.app.route('/api/portfolio_metrics')
        def get_portfolio_metrics():
            return jsonify(self.dashboard_data.get('portfolio_metrics', {}))
        
        @self.app.route('/api/positions')
        def get_positions():
            return jsonify(self.dashboard_data.get('positions', []))
        
        @self.app.route('/api/risk_metrics')
        def get_risk_metrics():
            return jsonify(self.dashboard_data.get('risk_metrics', {}))
        
        @self.app.route('/api/performance_chart')
        def get_performance_chart():
            return jsonify(self.dashboard_data.get('performance_chart', {}))
        
        @self.app.route('/api/emergency_stop', methods=['POST'])
        def emergency_stop():
            try:
                if hasattr(self.trading_manager, 'risk_manager'):
                    self.trading_manager.risk_manager.emergency_stop()
                self.trading_manager.stop_trading()
                return jsonify({'status': 'success', 'message': 'Emergency stop activated'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/pause_trading', methods=['POST'])
        def pause_trading():
            try:
                self.trading_manager.is_trading_active = False
                return jsonify({'status': 'success', 'message': 'Trading paused'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/resume_trading', methods=['POST'])
        def resume_trading():
            try:
                self.trading_manager.is_trading_active = True
                return jsonify({'status': 'success', 'message': 'Trading resumed'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
    
    def update_dashboard_data(self):
        """Update dashboard data from trading manager"""
        try:
            current_time = datetime.now()
            
            # Portfolio metrics
            portfolio_metrics = {
                'current_capital': self.trading_manager.current_capital,
                'initial_capital': self.trading_manager.initial_capital,
                'total_pnl': self.trading_manager.strategy_performance['total_pnl'],
                'total_trades': self.trading_manager.strategy_performance['total_trades'],
                'winning_trades': self.trading_manager.strategy_performance['winning_trades'],
                'win_rate': (self.trading_manager.strategy_performance['winning_trades'] / 
                           max(1, self.trading_manager.strategy_performance['total_trades'])) * 100,
                'return_percent': ((self.trading_manager.current_capital - self.trading_manager.initial_capital) / 
                                 self.trading_manager.initial_capital) * 100,
                'active_positions': len(self.trading_manager.active_positions),
                'trading_status': 'ACTIVE' if self.trading_manager.is_trading_active else 'PAUSED'
            }
            
            # Current positions
            positions = []
            for order_id, position in self.trading_manager.active_positions.items():
                positions.append({
                    'order_id': order_id,
                    'symbol': position.get('symbol', 'Unknown'),
                    'side': position.get('side', 'Unknown'),
                    'quantity': position.get('quantity', 0),
                    'entry_price': position.get('entry_price', 0),
                    'current_price': position.get('current_price', 0),
                    'pnl': position.get('pnl', 0),
                    'pnl_percent': position.get('pnl_percent', 0),
                    'timestamp': position.get('timestamp', current_time).isoformat() if hasattr(position.get('timestamp', current_time), 'isoformat') else str(position.get('timestamp', current_time))
                })
            
            # AI signals
            signals = []
            if hasattr(self.trading_manager, 'ai_ensemble') and self.trading_manager.ai_ensemble.signals_history:
                for signal in self.trading_manager.ai_ensemble.signals_history[-10:]:  # Last 10 signals
                    signals.append({
                        'timestamp': signal['timestamp'].isoformat() if hasattr(signal['timestamp'], 'isoformat') else str(signal['timestamp']),
                        'signal': signal.get('final_signal', 'Unknown'),
                        'strength': signal.get('signal_strength', 0),
                        'confidence': signal.get('confidence', 0),
                        'ensemble_score': signal.get('ensemble_score', 0)
                    })
            
            # Risk metrics
            risk_metrics = {}
            if hasattr(self.trading_manager, 'risk_manager'):
                risk_dashboard = self.trading_manager.risk_manager.get_risk_dashboard()
                risk_metrics = {
                    'risk_status': risk_dashboard.get('risk_status', 'Unknown'),
                    'circuit_breakers': risk_dashboard.get('circuit_breakers', {}),
                    'position_limits': risk_dashboard.get('position_limits', {}),
                    'recent_alerts': risk_dashboard.get('recent_alerts', [])
                }
            
            # Performance chart data
            performance_chart = self._generate_performance_chart()
            
            # Update dashboard data
            self.dashboard_data.update({
                'last_update': current_time.isoformat(),
                'portfolio_metrics': portfolio_metrics,
                'positions': positions,
                'signals': signals,
                'risk_metrics': risk_metrics,
                'performance_chart': performance_chart
            })
            
        except Exception as e:
            print(f"Error updating dashboard data: {e}")
    
    def _generate_performance_chart(self):
        """Generate performance chart data"""
        try:
            # Get trade history for chart
            if not hasattr(self.trading_manager, 'trade_history') or not self.trading_manager.trade_history:
                return {'data': [], 'layout': {}}
            
            # Create cumulative P&L chart
            trades = self.trading_manager.trade_history
            
            cumulative_pnl = []
            timestamps = []
            running_pnl = 0
            
            for trade in trades:
                running_pnl += trade.get('final_pnl', 0)
                cumulative_pnl.append(running_pnl)
                timestamp = trade.get('exit_timestamp', datetime.now())
                timestamps.append(timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp))
            
            # Create Plotly chart
            trace = go.Scatter(
                x=timestamps,
                y=cumulative_pnl,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#2E86C1', width=2),
                marker=dict(size=6)
            )
            
            layout = go.Layout(
                title='Portfolio Performance',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Cumulative P&L (₹)'),
                hovermode='closest',
                showlegend=True,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            fig = go.Figure(data=[trace], layout=layout)
            
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
        except Exception as e:
            print(f"Error generating performance chart: {e}")
            return {'data': [], 'layout': {}}
    
    def start_data_updates(self, update_interval=5):
        """Start background data updates"""
        def update_loop():
            while self.is_running:
                self.update_dashboard_data()
                time.sleep(update_interval)
        
        self.is_running = True
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        print(f"Dashboard data updates started (interval: {update_interval}s)")
    
    def stop_data_updates(self):
        """Stop background data updates"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        print("Dashboard data updates stopped")
    
    def start_web_server(self, debug=False):
        """Start web server for dashboard"""
        print(f"Starting web dashboard on http://localhost:{self.port}")
        print("Dashboard features:")
        print("   - Real-time portfolio metrics")
        print("   - Live position monitoring")
        print("   - AI signal tracking")
        print("   - Risk management alerts")
        print("   - Performance charts")
        print("   - Emergency controls")
        
        # Start data updates
        self.start_data_updates()
        
        try:
            self.app.run(host='0.0.0.0', port=self.port, debug=debug, use_reloader=False)
        except KeyboardInterrupt:
            print("\nDashboard stopped by user")
        finally:
            self.stop_data_updates()
    
    def print_console_dashboard(self):
        """Print console-based dashboard"""
        try:
            self.update_dashboard_data()
            
            print(f"\n{'='*80}")
            print(f"LIVE TRADING DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            # Portfolio metrics
            metrics = self.dashboard_data['portfolio_metrics']
            print(f"Portfolio Value: Rs.{metrics.get('current_capital', 0):,.2f}")
            print(f"Total P&L: Rs.{metrics.get('total_pnl', 0):,.2f} ({metrics.get('return_percent', 0):+.2f}%)")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1f}% ({metrics.get('winning_trades', 0)}/{metrics.get('total_trades', 0)})")
            print(f"📊 Active Positions: {metrics.get('active_positions', 0)}")
            print(f"🔄 Trading Status: {metrics.get('trading_status', 'Unknown')}")
            
            # Risk status
            risk = self.dashboard_data['risk_metrics']
            if risk:
                print(f"⚠️  Risk Status: {risk.get('risk_status', 'Unknown')}")
                
                # Circuit breakers
                breakers = risk.get('circuit_breakers', {})
                active_breakers = [name for name, active in breakers.items() if active]
                if active_breakers:
                    print(f"🚨 Active Circuit Breakers: {', '.join(active_breakers)}")
            
            # Active positions
            positions = self.dashboard_data['positions']
            if positions:
                print(f"\n📋 Active Positions:")
                for pos in positions:
                    pnl_emoji = "🟢" if pos.get('pnl', 0) > 0 else "🔴" if pos.get('pnl', 0) < 0 else "⚪"
                    print(f"{pnl_emoji} {pos.get('side', 'Unknown')} {pos.get('quantity', 0)} {pos.get('symbol', 'Unknown')} @ ₹{pos.get('entry_price', 0):.2f} | P&L: ₹{pos.get('pnl', 0):.2f} ({pos.get('pnl_percent', 0):+.2f}%)")
            
            # Recent signals
            signals = self.dashboard_data['signals']
            if signals:
                print(f"\n🤖 Recent AI Signals:")
                for signal in signals[-3:]:  # Last 3 signals
                    confidence_emoji = "🟢" if signal.get('confidence', 0) > 0.7 else "🟡" if signal.get('confidence', 0) > 0.5 else "🔴"
                    print(f"{confidence_emoji} {signal.get('signal', 'Unknown')} | Strength: {signal.get('strength', 0):.2f} | Confidence: {signal.get('confidence', 0):.2f}")
            
            # Alerts
            risk_alerts = risk.get('recent_alerts', [])
            if risk_alerts:
                print(f"\n⚠️  Recent Risk Alerts:")
                for alert in risk_alerts[-3:]:  # Last 3 alerts
                    alert_emoji = "🚨" if alert.get('severity') == 'CRITICAL' else "⚠️"
                    print(f"{alert_emoji} {alert.get('message', 'Unknown alert')}")
            
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"Error printing console dashboard: {e}")


class ConsoleMonitor:
    """
    Console-based monitoring for trading bot
    """
    
    def __init__(self, trading_manager):
        self.trading_manager = trading_manager
        self.dashboard = TradingDashboard(trading_manager)
        self.is_monitoring = False
    
    def start_console_monitoring(self, update_interval=10):
        """Start console monitoring"""
        self.is_monitoring = True
        print(f"📊 Starting console monitoring (update every {update_interval}s)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while self.is_monitoring:
                self.dashboard.print_console_dashboard()
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\n📊 Console monitoring stopped")
        finally:
            self.is_monitoring = False
    
    def stop_monitoring(self):
        """Stop console monitoring"""
        self.is_monitoring = False


# Helper function to create dashboard HTML template
def create_dashboard_template():
    """Create HTML template for dashboard"""
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-title { font-size: 14px; color: #666; margin-bottom: 10px; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #34495e; }
        .section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .section-title { font-size: 18px; font-weight: bold; margin-bottom: 15px; }
        .position-item { display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }
        .signal-item { display: flex; justify-content: space-between; padding: 8px; margin-bottom: 5px; border-radius: 4px; }
        .signal-buy { background-color: #d5f4e6; }
        .signal-sell { background-color: #fce4e4; }
        .signal-hold { background-color: #f0f0f0; }
        .controls { text-align: center; margin: 20px 0; }
        .btn { padding: 10px 20px; margin: 0 10px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .btn-danger { background-color: #e74c3c; color: white; }
        .btn-warning { background-color: #f39c12; color: white; }
        .btn-success { background-color: #27ae60; color: white; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-active { background-color: #27ae60; }
        .status-paused { background-color: #f39c12; }
        .status-stopped { background-color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Trading Dashboard</h1>
            <div id="lastUpdate"></div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Portfolio Value</div>
                <div class="metric-value" id="portfolioValue">₹0</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Total P&L</div>
                <div class="metric-value" id="totalPnl">₹0</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Win Rate</div>
                <div class="metric-value" id="winRate">0%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Active Positions</div>
                <div class="metric-value" id="activePositions">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Trading Status</div>
                <div class="metric-value" id="tradingStatus">
                    <span class="status-indicator" id="statusIndicator"></span>
                    <span id="statusText">Unknown</span>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn btn-danger" onclick="emergencyStop()">🚨 Emergency Stop</button>
            <button class="btn btn-warning" onclick="pauseTrading()">⏸️ Pause Trading</button>
            <button class="btn btn-success" onclick="resumeTrading()">▶️ Resume Trading</button>
        </div>
        
        <div class="section">
            <div class="section-title">📈 Performance Chart</div>
            <div id="performanceChart"></div>
        </div>
        
        <div class="section">
            <div class="section-title">📋 Active Positions</div>
            <div id="positionsList"></div>
        </div>
        
        <div class="section">
            <div class="section-title">🤖 Recent AI Signals</div>
            <div id="signalsList"></div>
        </div>
        
        <div class="section">
            <div class="section-title">⚠️ Risk Alerts</div>
            <div id="riskAlerts"></div>
        </div>
    </div>

    <script>
        function updateDashboard() {
            fetch('/api/dashboard_data')
                .then(response => response.json())
                .then(data => {
                    // Update metrics
                    const metrics = data.portfolio_metrics || {};
                    document.getElementById('portfolioValue').textContent = '₹' + (metrics.current_capital || 0).toLocaleString();
                    
                    const pnl = metrics.total_pnl || 0;
                    const pnlElement = document.getElementById('totalPnl');
                    pnlElement.textContent = '₹' + pnl.toLocaleString();
                    pnlElement.className = 'metric-value ' + (pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral');
                    
                    document.getElementById('winRate').textContent = (metrics.win_rate || 0).toFixed(1) + '%';
                    document.getElementById('activePositions').textContent = metrics.active_positions || 0;
                    
                    const status = metrics.trading_status || 'Unknown';
                    document.getElementById('statusText').textContent = status;
                    const indicator = document.getElementById('statusIndicator');
                    indicator.className = 'status-indicator status-' + status.toLowerCase();
                    
                    // Update last update time
                    document.getElementById('lastUpdate').textContent = 'Last updated: ' + new Date(data.last_update).toLocaleTimeString();
                    
                    // Update performance chart
                    if (data.performance_chart && data.performance_chart.data) {
                        Plotly.newPlot('performanceChart', data.performance_chart.data, data.performance_chart.layout);
                    }
                    
                    // Update positions
                    updatePositions(data.positions || []);
                    
                    // Update signals
                    updateSignals(data.signals || []);
                    
                    // Update risk alerts
                    updateRiskAlerts(data.risk_metrics || {});
                })
                .catch(error => console.error('Error updating dashboard:', error));
        }
        
        function updatePositions(positions) {
            const container = document.getElementById('positionsList');
            if (positions.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666;">No active positions</div>';
                return;
            }
            
            container.innerHTML = positions.map(pos => `
                <div class="position-item">
                    <div>
                        <strong>${pos.symbol}</strong> ${pos.side} ${pos.quantity}
                        <br><small>Entry: ₹${pos.entry_price.toFixed(2)}</small>
                    </div>
                    <div style="text-align: right;">
                        <div class="${pos.pnl > 0 ? 'positive' : pos.pnl < 0 ? 'negative' : 'neutral'}">
                            ₹${pos.pnl.toFixed(2)}
                        </div>
                        <small>${pos.pnl_percent > 0 ? '+' : ''}${pos.pnl_percent.toFixed(2)}%</small>
                    </div>
                </div>
            `).join('');
        }
        
        function updateSignals(signals) {
            const container = document.getElementById('signalsList');
            if (signals.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666;">No recent signals</div>';
                return;
            }
            
            container.innerHTML = signals.slice(-5).map(signal => `
                <div class="signal-item signal-${signal.signal.toLowerCase()}">
                    <div>
                        <strong>${signal.signal}</strong>
                        <br><small>${new Date(signal.timestamp).toLocaleTimeString()}</small>
                    </div>
                    <div style="text-align: right;">
                        <div>Strength: ${signal.strength.toFixed(2)}</div>
                        <small>Confidence: ${signal.confidence.toFixed(2)}</small>
                    </div>
                </div>
            `).join('');
        }
        
        function updateRiskAlerts(riskMetrics) {
            const container = document.getElementById('riskAlerts');
            const alerts = riskMetrics.recent_alerts || [];
            
            if (alerts.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666;">No recent alerts</div>';
                return;
            }
            
            container.innerHTML = alerts.slice(-5).map(alert => `
                <div style="padding: 8px; margin-bottom: 5px; border-left: 4px solid ${alert.severity === 'CRITICAL' ? '#e74c3c' : '#f39c12'}; background-color: #f9f9f9;">
                    <strong>${alert.severity}</strong>: ${alert.message}
                    <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                </div>
            `).join('');
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to activate emergency stop? This will halt all trading immediately.')) {
                fetch('/api/emergency_stop', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => alert(data.message))
                    .catch(error => alert('Error: ' + error));
            }
        }
        
        function pauseTrading() {
            fetch('/api/pause_trading', {method: 'POST'})
                .then(response => response.json())
                .then(data => alert(data.message))
                .catch(error => alert('Error: ' + error));
        }
        
        function resumeTrading() {
            fetch('/api/resume_trading', {method: 'POST'})
                .then(response => response.json())
                .then(data => alert(data.message))
                .catch(error => alert('Error: ' + error));
        }
        
        // Update dashboard every 5 seconds
        setInterval(updateDashboard, 5000);
        updateDashboard(); // Initial load
    </script>
</body>
</html>
    '''
    
    # Create templates directory if it doesn't exist
    import os
    os.makedirs('templates', exist_ok=True)
    
    # Write template file
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_content)
    
    print("📄 Dashboard template created: templates/dashboard.html")