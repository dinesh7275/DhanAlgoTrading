#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIBot Main Entry Point
=====================

Main script to run the AI-powered trading bot with real-time predictions
and automated trading capabilities on localhost:8080
"""

import sys
import logging
from datetime import datetime
import argparse
from pathlib import Path
import threading
from flask import Flask, jsonify, render_template_string
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from config.config import *
except ImportError:
    print("Config file not found. Please copy config_template.py to config.py and update settings.")
    print("Creating default config.py from template...")
    try:
        import shutil
        config_dir = Path(__file__).parent / "config"
        shutil.copy(config_dir / "config_template.py", config_dir / "config.py")
        print("Config file created. Please edit config/config.py with your settings.")
        from config.config import *
    except Exception as e:
        print(f"Error creating config file: {e}")
        sys.exit(1)

# Try importing AI models with error handling
try:
    from models.iv_prediction.live_iv_predictor import LiveIVPredictor
except ImportError as e:
    print(f"Warning: Could not import IV predictor: {e}")
    LiveIVPredictor = None

try:
    from models.price_movement.live_price_predictor import LivePricePredictor
except ImportError as e:
    print(f"Warning: Could not import price predictor: {e}")
    LivePricePredictor = None

try:
    from models.options_anomaly.live_anomaly_detector import LiveAnomalyDetector
except ImportError as e:
    print(f"Warning: Could not import anomaly detector: {e}")
    LiveAnomalyDetector = None

try:
    from models.risk_analysis.risk_monitor import RiskMonitor
except ImportError as e:
    print(f"Warning: Could not import risk monitor: {e}")
    RiskMonitor = None

class AIBot:
    """
    Main AIBot class that coordinates all AI models and trading decisions
    """
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI models with error handling
        self.iv_predictor = LiveIVPredictor() if LiveIVPredictor else None
        self.price_predictor = LivePricePredictor() if LivePricePredictor else None
        self.anomaly_detector = LiveAnomalyDetector() if LiveAnomalyDetector else None
        self.risk_monitor = RiskMonitor() if RiskMonitor else None
        
        self.is_running = False
        self.current_capital = TRADING_CONFIG['initial_capital']
        
        # Report which models are available
        available_models = []
        if self.iv_predictor: available_models.append("IV Predictor")
        if self.price_predictor: available_models.append("Price Predictor") 
        if self.anomaly_detector: available_models.append("Anomaly Detector")
        if self.risk_monitor: available_models.append("Risk Monitor")
        
        self.logger.info(f"AIBot initialized with models: {', '.join(available_models) if available_models else 'None'}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = LOGGING_CONFIG
        
        # Create logs directory if it doesn't exist
        Path(log_config['log_file']).parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config['log_level']),
            format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            handlers=[
                logging.FileHandler(log_config['log_file']),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def start(self):
        """Start the AI bot"""
        self.logger.info("Starting AIBot...")
        self.is_running = True
        
        try:
            while self.is_running:
                # Get current market analysis
                analysis = self.analyze_market()
                
                # Make trading decision
                decision = self.make_trading_decision(analysis)
                
                # Execute trade if decision is made
                if decision['action'] != 'HOLD':
                    self.execute_trade(decision)
                
                # Update risk monitoring (if available)
                if self.risk_monitor:
                    try:
                        self.risk_monitor.update_portfolio_metrics()
                    except Exception as e:
                        self.logger.warning(f"Risk monitoring update failed: {e}")
                
                # Wait for next iteration
                import time
                time.sleep(DATA_CONFIG['update_frequency'])
                
        except KeyboardInterrupt:
            self.logger.info("Received stop signal")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def analyze_market(self):
        """Perform comprehensive market analysis using available AI models"""
        self.logger.info("Analyzing market conditions...")
        
        try:
            # Get IV prediction (if available)
            iv_analysis = {}
            if self.iv_predictor:
                try:
                    iv_analysis = self.iv_predictor.predict_iv("NIFTY")
                except Exception as e:
                    self.logger.warning(f"IV prediction failed: {e}")
                    iv_analysis = {'predicted_iv': None, 'confidence': 0}
            
            # Get price movement prediction (if available)
            price_analysis = {}
            if self.price_predictor:
                try:
                    price_analysis = self.price_predictor.predict_next_movement()
                except Exception as e:
                    self.logger.warning(f"Price prediction failed: {e}")
                    price_analysis = {'direction': 'HOLD', 'confidence': 0}
            
            # Get options anomalies (if available)
            anomalies = []
            if self.anomaly_detector:
                try:
                    anomalies = self.anomaly_detector.find_mispriced_options()
                except Exception as e:
                    self.logger.warning(f"Anomaly detection failed: {e}")
                    anomalies = []
            
            # Get risk metrics (if available)
            risk_metrics = {}
            if self.risk_monitor:
                try:
                    risk_metrics = self.risk_monitor.get_current_metrics()
                except Exception as e:
                    self.logger.warning(f"Risk monitoring failed: {e}")
                    risk_metrics = {}
            
            # Combine all analyses
            market_analysis = {
                'timestamp': datetime.now(),
                'iv_prediction': iv_analysis,
                'price_prediction': price_analysis,
                'options_anomalies': anomalies,
                'risk_metrics': risk_metrics
            }
            
            iv_val = iv_analysis.get('predicted_iv', 'N/A')
            iv_str = f"{iv_val:.2f}%" if iv_val is not None else "N/A"
            
            self.logger.info(f"Market analysis complete: "
                           f"IV={iv_str}, "
                           f"Price Direction={price_analysis.get('direction', 'N/A')}, "
                           f"Anomalies Found={len(anomalies)}")
            
            return market_analysis
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return None
    
    def make_trading_decision(self, analysis):
        """Make trading decision based on AI analysis"""
        if not analysis:
            return {'action': 'HOLD', 'reason': 'No analysis available'}
        
        # Extract key metrics
        iv_prediction = analysis['iv_prediction']
        price_prediction = analysis['price_prediction']
        anomalies = analysis['options_anomalies']
        risk_metrics = analysis['risk_metrics']
        
        # Risk check first
        if risk_metrics.get('daily_loss', 0) > TRADING_CONFIG['max_daily_loss']:
            return {'action': 'HOLD', 'reason': 'Daily loss limit reached'}
        
        # Decision logic based on AI predictions
        confidence_threshold = 0.7
        
        # High confidence price prediction
        if (price_prediction.get('confidence', 0) > confidence_threshold and
            price_prediction.get('direction') in ['UP', 'DOWN']):
            
            # Choose appropriate option type
            option_type = 'CE' if price_prediction['direction'] == 'UP' else 'PE'
            
            return {
                'action': 'BUY',
                'instrument_type': 'OPTION',
                'option_type': option_type,
                'confidence': price_prediction['confidence'],
                'reason': f"High confidence {price_prediction['direction']} prediction"
            }
        
        # Options anomaly opportunity
        if anomalies and len(anomalies) > 0:
            best_anomaly = max(anomalies, key=lambda x: x.get('expected_profit', 0))
            if best_anomaly.get('expected_profit', 0) > 0.05:  # 5% expected profit
                return {
                    'action': 'BUY',
                    'symbol': best_anomaly['symbol'],
                    'reason': f"Mispriced option opportunity: {best_anomaly['expected_profit']:.2%}"
                }
        
        return {'action': 'HOLD', 'reason': 'No high-confidence opportunities'}
    
    def execute_trade(self, decision):
        """Execute trading decision"""
        self.logger.info(f"Executing trade: {decision}")
        
        try:
            # In a real implementation, this would connect to the trading API
            # For now, we'll simulate the trade
            
            trade_result = {
                'status': 'SIMULATED',
                'action': decision['action'],
                'timestamp': datetime.now(),
                'reason': decision['reason']
            }
            
            self.logger.info(f"Trade executed (simulated): {trade_result}")
            
            # Update portfolio tracking
            # self.update_portfolio(trade_result)
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    def stop(self):
        """Stop the AI bot"""
        self.logger.info("Stopping AIBot...")
        self.is_running = False
        
        # Save final state
        self.save_session_data()
        
        self.logger.info("AIBot stopped successfully")
    
    def save_session_data(self):
        """Save session data for analysis"""
        try:
            session_data = {
                'timestamp': datetime.now(),
                'final_capital': self.current_capital,
                'performance_metrics': self.risk_monitor.get_current_metrics()
            }
            
            # Save to file
            import json
            with open(f'data/session_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
                
            self.logger.info("Session data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving session data: {e}")

# Global bot instance
global_bot = None
app = Flask(__name__)

# Simple HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AIBot - AI Trading System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #007bff; }
        .status { display: flex; justify-content: space-around; margin: 20px 0; }
        .status-card { background: #f8f9fa; padding: 20px; border-radius: 8px; min-width: 150px; text-align: center; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; }
        .log-container { background: #1e1e1e; color: #00ff00; padding: 20px; border-radius: 8px; height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px; }
        .btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #0056b3; }
        .btn-danger { background: #dc3545; }
        .btn-success { background: #28a745; }
        .running { color: #28a745; font-weight: bold; }
        .stopped { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AIBot - AI Trading System</h1>
            <p>Real-time AI-powered Options Trading Bot</p>
        </div>
        
        <div class="status">
            <div class="status-card">
                <h3>Status</h3>
                <div id="bot-status" class="stopped">Stopped</div>
            </div>
            <div class="status-card">
                <h3>Capital</h3>
                <div id="capital">₹10,000</div>
            </div>
            <div class="status-card">
                <h3>Daily P&L</h3>
                <div id="daily-pnl">₹0</div>
            </div>
            <div class="status-card">
                <h3>Trades Today</h3>
                <div id="trades-today">0</div>
            </div>
        </div>
        
        <div style="text-align: center; margin: 20px 0;">
            <button class="btn btn-success" onclick="startBot()">Start Trading</button>
            <button class="btn btn-danger" onclick="stopBot()">Stop Trading</button>
            <button class="btn" onclick="getStatus()">Refresh Status</button>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h4>Market Analysis</h4>
                <div id="market-analysis">Initializing...</div>
            </div>
            <div class="metric-card">
                <h4>AI Signal</h4>
                <div id="ai-signal">HOLD</div>
            </div>
            <div class="metric-card">
                <h4>Risk Metrics</h4>
                <div id="risk-metrics">Monitoring...</div>
            </div>
            <div class="metric-card">
                <h4>Available Models</h4>
                <div id="models-status">Loading...</div>
            </div>
        </div>
        
        <div style="margin-top: 30px;">
            <h3>Live Activity Log</h3>
            <div id="log-container" class="log-container">
                <div>AIBot Web Interface Started on localhost:8080</div>
                <div>Waiting for bot initialization...</div>
            </div>
        </div>
    </div>

    <script>
        let logCount = 2;
        
        function addLog(message) {
            const logContainer = document.getElementById('log-container');
            const logEntry = document.createElement('div');
            const timestamp = new Date().toLocaleTimeString();
            logEntry.innerHTML = `[${timestamp}] ${message}`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
            
            // Keep only last 100 log entries
            if (logCount++ > 100) {
                logContainer.removeChild(logContainer.firstChild);
            }
        }
        
        function startBot() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    addLog(`Start command sent: ${data.message}`);
                    if (data.success) {
                        document.getElementById('bot-status').className = 'running';
                        document.getElementById('bot-status').textContent = 'Running';
                    }
                });
        }
        
        function stopBot() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    addLog(`Stop command sent: ${data.message}`);
                    if (data.success) {
                        document.getElementById('bot-status').className = 'stopped';
                        document.getElementById('bot-status').textContent = 'Stopped';
                    }
                });
        }
        
        function getStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const status = data.status;
                        document.getElementById('bot-status').textContent = status.is_running ? 'Running' : 'Stopped';
                        document.getElementById('bot-status').className = status.is_running ? 'running' : 'stopped';
                        document.getElementById('capital').textContent = `₹${status.current_capital.toLocaleString()}`;
                        document.getElementById('daily-pnl').textContent = `₹${status.daily_pnl}`;
                        document.getElementById('trades-today').textContent = status.trades_today || 0;
                        document.getElementById('market-analysis').textContent = status.market_analysis || 'No data';
                        document.getElementById('ai-signal').textContent = status.ai_signal || 'HOLD';
                        document.getElementById('risk-metrics').textContent = status.risk_status || 'OK';
                        document.getElementById('models-status').textContent = status.available_models || 'Loading...';
                        
                        addLog(`Status updated - ${status.is_running ? 'Bot Running' : 'Bot Stopped'}`);
                    }
                });
        }
        
        // Auto-refresh status every 10 seconds
        setInterval(getStatus, 10000);
        
        // Initial status load
        setTimeout(getStatus, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    """Get bot status"""
    if global_bot:
        analysis = global_bot.analyze_market()
        signal = global_bot.make_trading_decision(analysis) if analysis else {'action': 'HOLD'}
        
        # Get available models
        available_models = []
        if global_bot.iv_predictor: available_models.append("IV Predictor")
        if global_bot.price_predictor: available_models.append("Price Predictor") 
        if global_bot.anomaly_detector: available_models.append("Anomaly Detector")
        if global_bot.risk_monitor: available_models.append("Risk Monitor")
        
        return jsonify({
            'success': True,
            'status': {
                'is_running': global_bot.is_running,
                'current_capital': global_bot.current_capital,
                'daily_pnl': 0,  # Would be calculated from risk monitor
                'trades_today': 0,
                'market_analysis': f"IV: {analysis.get('iv_prediction', {}).get('predicted_iv', 'N/A')}" if analysis else "No data",
                'ai_signal': signal.get('action', 'HOLD'),
                'risk_status': 'OK',
                'available_models': ', '.join(available_models) if available_models else 'None'
            }
        })
    else:
        return jsonify({
            'success': True,
            'status': {
                'is_running': False,
                'current_capital': 10000,
                'daily_pnl': 0,
                'trades_today': 0,
                'market_analysis': 'Bot not initialized',
                'ai_signal': 'HOLD',
                'risk_status': 'Offline',
                'available_models': 'Bot not started'
            }
        })

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start the bot"""
    global global_bot
    if not global_bot:
        global_bot = AIBot()
    
    if not global_bot.is_running:
        # Start bot in background thread
        bot_thread = threading.Thread(target=global_bot.start, daemon=True)
        bot_thread.start()
        return jsonify({'success': True, 'message': 'AIBot started'})
    else:
        return jsonify({'success': False, 'message': 'AIBot is already running'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop the bot"""
    global global_bot
    if global_bot and global_bot.is_running:
        global_bot.stop()
        return jsonify({'success': True, 'message': 'AIBot stopped'})
    else:
        return jsonify({'success': False, 'message': 'AIBot is not running'})

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='AIBot - AI-Powered Trading System')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--port', type=int, default=8080, help='Port to run web interface')
    
    args = parser.parse_args()
    
    # Update logging level if specified
    if args.log_level:
        LOGGING_CONFIG['log_level'] = args.log_level
    
    print("="*60)
    print("🤖 AIBot - AI-Powered Trading System")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Capital: ₹{TRADING_CONFIG['initial_capital']:,}")
    print(f"Web Interface: http://localhost:{args.port}")
    print("="*60)
    
    try:
        # Start Flask web server
        print(f"Starting web server on http://localhost:{args.port}")
        app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping AIBot...")
        if global_bot:
            global_bot.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        if global_bot:
            global_bot.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()