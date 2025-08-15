#!/usr/bin/env python3
"""
Flask Web Application for Dhan Algorithmic Trading
==================================================

Main Flask application with dashboard interface for trading system
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from functools import wraps

# Import trading components
from controllers.tradingController import TradingController
from services.tradingService import TradingService
from models.models import User, TradingSession, Trade
from live_trading.dhan_data_fetcher import DhanLiveDataFetcher
from live_trading.ai_ensemble import TradingSignalEnsemble
from live_trading.risk_manager import LiveRiskManager

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
trading_controller = TradingController()
trading_service = TradingService()

def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Main dashboard page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Get trading statistics
        stats = trading_service.get_trading_statistics()
        
        # Get recent trades
        recent_trades = trading_service.get_recent_trades(limit=10)
        
        # Get current positions
        positions = trading_service.get_current_positions()
        
        # Get market data
        market_data = trading_service.get_market_overview()
        
        return render_template('dashboard.html', 
                             stats=stats,
                             recent_trades=recent_trades,
                             positions=positions,
                             market_data=market_data)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        flash(f'Error loading dashboard: {e}', 'error')
        return render_template('dashboard.html', stats={}, recent_trades=[], positions=[], market_data={})

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Simple authentication (replace with proper auth)
        if username == 'admin' and password == 'admin':
            session['user_id'] = 1
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/api/market-data')
@login_required
def api_market_data():
    """API endpoint for market data"""
    try:
        market_data = trading_service.get_live_market_data()
        return jsonify(market_data)
    except Exception as e:
        logger.error(f"Market data API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading-signal')
@login_required
def api_trading_signal():
    """API endpoint for trading signals"""
    try:
        signal = trading_service.get_current_trading_signal()
        return jsonify(signal)
    except Exception as e:
        logger.error(f"Trading signal API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute-trade', methods=['POST'])
@login_required
def api_execute_trade():
    """API endpoint to execute trades"""
    try:
        trade_data = request.get_json()
        result = trading_controller.execute_trade(trade_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Trade execution API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions')
@login_required
def api_positions():
    """API endpoint for current positions"""
    try:
        positions = trading_service.get_current_positions()
        return jsonify(positions)
    except Exception as e:
        logger.error(f"Positions API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio-performance')
@login_required
def api_portfolio_performance():
    """API endpoint for portfolio performance"""
    try:
        performance = trading_service.get_portfolio_performance()
        return jsonify(performance)
    except Exception as e:
        logger.error(f"Portfolio performance API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk-metrics')
@login_required
def api_risk_metrics():
    """API endpoint for risk metrics"""
    try:
        risk_metrics = trading_service.get_risk_metrics()
        return jsonify(risk_metrics)
    except Exception as e:
        logger.error(f"Risk metrics API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/settings')
@login_required
def settings():
    """Settings page"""
    try:
        current_config = trading_service.get_trading_configuration()
        return render_template('settings.html', config=current_config)
    except Exception as e:
        logger.error(f"Settings page error: {e}")
        flash(f'Error loading settings: {e}', 'error')
        return render_template('settings.html', config={})

@app.route('/api/update-settings', methods=['POST'])
@login_required
def api_update_settings():
    """API endpoint to update settings"""
    try:
        settings_data = request.get_json()
        result = trading_service.update_trading_configuration(settings_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Settings update API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/trades')
@login_required
def trades():
    """Trades history page"""
    try:
        page = request.args.get('page', 1, type=int)
        trades_data = trading_service.get_trades_history(page=page, per_page=50)
        return render_template('trades.html', trades=trades_data)
    except Exception as e:
        logger.error(f"Trades page error: {e}")
        flash(f'Error loading trades: {e}', 'error')
        return render_template('trades.html', trades=[])

@app.route('/analytics')
@login_required
def analytics():
    """Analytics and reports page"""
    try:
        analytics_data = trading_service.get_analytics_data()
        return render_template('analytics.html', data=analytics_data)
    except Exception as e:
        logger.error(f"Analytics page error: {e}")
        flash(f'Error loading analytics: {e}', 'error')
        return render_template('analytics.html', data={})

@app.route('/api/start-trading', methods=['POST'])
@login_required
def api_start_trading():
    """API endpoint to start automated trading"""
    try:
        config = request.get_json()
        result = trading_controller.start_automated_trading(config)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Start trading API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-trading', methods=['POST'])
@login_required
def api_stop_trading():
    """API endpoint to stop automated trading"""
    try:
        result = trading_controller.stop_automated_trading()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Stop trading API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading-status')
@login_required
def api_trading_status():
    """API endpoint for trading system status"""
    try:
        status = trading_service.get_trading_system_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Trading status API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {error}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Dhan Algorithmic Trading Dashboard on port {port}")
    print(f"Debug mode: {debug_mode}")
    print("Access at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)