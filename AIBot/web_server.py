#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIBot Web Server
===============

Simple web server for AIBot on localhost:8080
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import threading
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from flask import Flask, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask not installed. Install with: pip install flask")
    FLASK_AVAILABLE = False

try:
    from config.config import TRADING_CONFIG, LOGGING_CONFIG
except ImportError:
    # Default config if not available
    TRADING_CONFIG = {'initial_capital': 10000}
    LOGGING_CONFIG = {'log_level': 'INFO'}

# Simple HTTP server fallback if Flask not available
if not FLASK_AVAILABLE:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    class SimpleAIBotHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(SIMPLE_HTML.encode())
            elif self.path == '/api/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    'success': True,
                    'status': {
                        'is_running': False,
                        'current_capital': 10000,
                        'daily_pnl': 0,
                        'trades_today': 0,
                        'message': 'AIBot Web Interface (Simple Mode)'
                    }
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path in ['/api/start', '/api/stop']:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                action = 'started' if 'start' in self.path else 'stopped'
                response = {'success': True, 'message': f'AIBot {action} (simulation)'}
                self.wfile.write(json.dumps(response).encode())

        def log_message(self, format, *args):
            # Suppress default logs
            pass

# Initialize Flask app if available
if FLASK_AVAILABLE:
    app = Flask(__name__)
    global_bot_status = {
        'is_running': False,
        'current_capital': 10000,
        'daily_pnl': 0,
        'trades_today': 0,
        'start_time': None
    }

# HTML Template
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>AIBot - AI Trading System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #007bff; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .status-card { background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #dee2e6; }
        .btn { background: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin: 5px; font-size: 14px; }
        .btn:hover { background: #0056b3; }
        .btn-success { background: #28a745; }
        .btn-danger { background: #dc3545; }
        .btn-info { background: #17a2b8; }
        .running { color: #28a745; font-weight: bold; }
        .stopped { color: #dc3545; font-weight: bold; }
        .log-area { background: #1e1e1e; color: #00ff00; padding: 15px; border-radius: 6px; height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; margin-top: 20px; }
        .metrics { margin: 20px 0; }
        .metric-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AIBot - AI Trading System</h1>
            <p>Web Interface on localhost:8080</p>
            <p><strong>Status:</strong> <span id="status" class="stopped">Offline</span></p>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h4>💰 Capital</h4>
                <div id="capital">₹10,000</div>
            </div>
            <div class="status-card">
                <h4>📈 Daily P&L</h4>
                <div id="pnl">₹0</div>
            </div>
            <div class="status-card">
                <h4>🔄 Trades Today</h4>
                <div id="trades">0</div>
            </div>
            <div class="status-card">
                <h4>⏰ Uptime</h4>
                <div id="uptime">Not started</div>
            </div>
        </div>
        
        <div style="text-align: center; margin: 20px 0;">
            <button class="btn btn-success" onclick="startBot()">▶ Start AIBot</button>
            <button class="btn btn-danger" onclick="stopBot()">⏹ Stop AIBot</button>
            <button class="btn btn-info" onclick="refreshStatus()">🔄 Refresh</button>
        </div>
        
        <div class="metrics">
            <h3>📊 System Metrics</h3>
            <div class="metric-row">
                <span>Server Status:</span>
                <span id="server-status">Online</span>
            </div>
            <div class="metric-row">
                <span>Connection:</span>
                <span id="connection">Connected</span>
            </div>
            <div class="metric-row">
                <span>Last Update:</span>
                <span id="last-update">-</span>
            </div>
        </div>
        
        <div>
            <h3>📝 Activity Log</h3>
            <div id="log" class="log-area">
                <div>[${new Date().toLocaleTimeString()}] AIBot Web Interface Started</div>
                <div>[${new Date().toLocaleTimeString()}] Ready for commands</div>
            </div>
        </div>
    </div>

    <script>
        let logCount = 2;
        
        function addLog(message) {
            const log = document.getElementById('log');
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.innerHTML = `[${time}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
            
            if (logCount++ > 50) {
                log.removeChild(log.firstChild);
            }
        }
        
        function updateStatus(data) {
            if (data.success && data.status) {
                const s = data.status;
                document.getElementById('status').textContent = s.is_running ? 'Running' : 'Stopped';
                document.getElementById('status').className = s.is_running ? 'running' : 'stopped';
                document.getElementById('capital').textContent = `₹${s.current_capital.toLocaleString()}`;
                document.getElementById('pnl').textContent = `₹${s.daily_pnl}`;
                document.getElementById('trades').textContent = s.trades_today;
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                
                if (s.message) {
                    addLog(s.message);
                }
            }
        }
        
        function startBot() {
            addLog('Sending start command...');
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    addLog(data.message || 'Start command completed');
                    refreshStatus();
                })
                .catch(err => addLog('Error: ' + err.message));
        }
        
        function stopBot() {
            addLog('Sending stop command...');
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    addLog(data.message || 'Stop command completed');
                    refreshStatus();
                })
                .catch(err => addLog('Error: ' + err.message));
        }
        
        function refreshStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateStatus(data);
                    addLog('Status refreshed');
                })
                .catch(err => {
                    addLog('Connection error: ' + err.message);
                    document.getElementById('connection').textContent = 'Disconnected';
                });
        }
        
        // Auto-refresh every 15 seconds
        setInterval(refreshStatus, 15000);
        
        // Initial load
        setTimeout(refreshStatus, 1000);
        
        addLog('Web interface initialized');
    </script>
</body>
</html>"""

SIMPLE_HTML = HTML_TEMPLATE

# Flask routes (if Flask is available)
if FLASK_AVAILABLE:
    @app.route('/')
    def dashboard():
        return HTML_TEMPLATE

    @app.route('/api/status')
    def api_status():
        return jsonify({
            'success': True,
            'status': {
                'is_running': global_bot_status['is_running'],
                'current_capital': global_bot_status['current_capital'],
                'daily_pnl': global_bot_status['daily_pnl'],
                'trades_today': global_bot_status['trades_today'],
                'message': 'AIBot Web Interface Active'
            }
        })

    @app.route('/api/start', methods=['POST'])
    def api_start():
        global_bot_status['is_running'] = True
        global_bot_status['start_time'] = datetime.now()
        return jsonify({'success': True, 'message': 'AIBot started (simulation mode)'})

    @app.route('/api/stop', methods=['POST'])
    def api_stop():
        global_bot_status['is_running'] = False
        global_bot_status['start_time'] = None
        return jsonify({'success': True, 'message': 'AIBot stopped'})

def main():
    print("="*60)
    print("🤖 AIBot Web Server")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("URL: http://localhost:8080")
    print("="*60)
    
    if FLASK_AVAILABLE:
        print("Starting Flask server...")
        try:
            app.run(host='127.0.0.1', port=8080, debug=False, use_reloader=False)
        except Exception as e:
            print(f"Flask server error: {e}")
            print("Trying simple HTTP server...")
            start_simple_server()
    else:
        print("Flask not available, starting simple HTTP server...")
        start_simple_server()

def start_simple_server():
    try:
        server = HTTPServer(('127.0.0.1', 8080), SimpleAIBotHandler)
        print("Simple HTTP server started on http://localhost:8080")
        print("Press Ctrl+C to stop")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main()