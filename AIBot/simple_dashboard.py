#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Dashboard for AI Trading Bot
==================================

Basic web dashboard showing trading performance and signals.
"""

from flask import Flask, render_template_string, jsonify
import json
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Simple data store
dashboard_data = {
    'portfolio_value': 10000,
    'total_pnl': 0,
    'signals_today': 0,
    'trades_today': 0,
    'last_signal': 'HOLD',
    'last_price': 0,
    'win_rate': 0,
    'last_update': datetime.now()
}

# Simple HTML template
SIMPLE_DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Trading Bot - Simple Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; margin-top: 5px; }
        .status { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #f39c12; }
        .update-time { text-align: center; margin-top: 20px; color: #7f8c8d; }
    </style>
    <script>
        function updateData() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('portfolio').textContent = 'Rs.' + data.portfolio_value.toLocaleString();
                    document.getElementById('pnl').textContent = 'Rs.' + (data.total_pnl > 0 ? '+' : '') + data.total_pnl.toLocaleString();
                    document.getElementById('pnl').className = 'metric-value ' + (data.total_pnl > 0 ? 'positive' : data.total_pnl < 0 ? 'negative' : 'neutral');
                    document.getElementById('signals').textContent = data.signals_today;
                    document.getElementById('trades').textContent = data.trades_today;
                    document.getElementById('last_signal').textContent = data.last_signal;
                    document.getElementById('last_signal').className = 'metric-value ' + (data.last_signal === 'BUY' ? 'positive' : data.last_signal === 'SELL' ? 'negative' : 'neutral');
                    document.getElementById('last_price').textContent = 'Rs.' + data.last_price.toFixed(2);
                    document.getElementById('win_rate').textContent = data.win_rate.toFixed(1) + '%';
                    document.getElementById('last_update').textContent = 'Last updated: ' + new Date(data.last_update).toLocaleTimeString();
                })
                .catch(error => console.error('Error:', error));
        }
        
        setInterval(updateData, 5000); // Update every 5 seconds
        window.onload = updateData;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Trading Bot - Simple Dashboard</h1>
            <p>Real-time paper trading performance</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div id="portfolio" class="metric-value">Rs.10,000</div>
                <div class="metric-label">Portfolio Value</div>
            </div>
            
            <div class="metric-card">
                <div id="pnl" class="metric-value neutral">Rs.0</div>
                <div class="metric-label">Total P&L</div>
            </div>
            
            <div class="metric-card">
                <div id="signals" class="metric-value">0</div>
                <div class="metric-label">Signals Today</div>
            </div>
            
            <div class="metric-card">
                <div id="trades" class="metric-value">0</div>
                <div class="metric-label">Trades Today</div>
            </div>
            
            <div class="metric-card">
                <div id="last_signal" class="metric-value neutral">HOLD</div>
                <div class="metric-label">Last Signal</div>
            </div>
            
            <div class="metric-card">
                <div id="last_price" class="metric-value">Rs.0.00</div>
                <div class="metric-label">Current Price</div>
            </div>
            
            <div class="metric-card">
                <div id="win_rate" class="metric-value">0%</div>
                <div class="metric-label">Win Rate</div>
            </div>
        </div>
        
        <div class="status">
            <h3>Status</h3>
            <p>Simple AI Trading Bot is running in paper trading mode.</p>
            <p>Monitoring NIFTY 50 with basic technical analysis.</p>
            <p>All trades are simulated - no real money involved.</p>
        </div>
        
        <div class="update-time">
            <span id="last_update">Loading...</span>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(SIMPLE_DASHBOARD_HTML)

@app.route('/api/data')
def get_data():
    return jsonify(dashboard_data)

@app.route('/api/update', methods=['POST'])
def update_data():
    # This would be called by the trading bot to update data
    global dashboard_data
    # For now, simulate some data updates
    dashboard_data['last_update'] = datetime.now().isoformat()
    return jsonify({'status': 'success'})

def simulate_data():
    """Simulate some data updates for demo"""
    import random
    global dashboard_data
    
    while True:
        time.sleep(10)  # Update every 10 seconds
        
        # Simulate some changes
        dashboard_data['last_price'] = 25000 + random.randint(-100, 100)
        dashboard_data['signals_today'] += random.choice([0, 0, 0, 1])  # Occasionally add signal
        
        if random.random() < 0.1:  # 10% chance to add trade
            dashboard_data['trades_today'] += 1
            pnl_change = random.randint(-500, 800)
            dashboard_data['total_pnl'] += pnl_change
            dashboard_data['portfolio_value'] = 10000 + dashboard_data['total_pnl']
            
            if dashboard_data['trades_today'] > 0:
                dashboard_data['win_rate'] = (random.randint(50, 80))
        
        dashboard_data['last_signal'] = random.choice(['HOLD', 'HOLD', 'HOLD', 'BUY', 'SELL'])
        dashboard_data['last_update'] = datetime.now().isoformat()

if __name__ == '__main__':
    print("Starting Simple Dashboard...")
    print("Dashboard available at: http://localhost:5001")
    print("Press Ctrl+C to stop")
    
    # Start data simulation in background
    data_thread = threading.Thread(target=simulate_data, daemon=True)
    data_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("\nDashboard stopped")