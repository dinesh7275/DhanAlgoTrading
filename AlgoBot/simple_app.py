#!/usr/bin/env python3
"""
Simple AlgoBot Web Application
=============================

Basic Flask web application for algorithmic trading interface
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import json
import logging
from datetime import datetime
import random

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'algobot-trading-secret-key')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock data for demonstration
PORTFOLIO_DATA = {
    'total_value': 10000,
    'available_cash': 8500,
    'invested_value': 1500,
    'total_pnl': 0,
    'daily_pnl': 0,
    'positions': [],
    'trades': []
}

@app.route('/')
def dashboard():
    """Main dashboard route"""
    return render_template('dashboard.html')

@app.route('/login')
def login():
    """Login page route"""
    return render_template('login.html')

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Handle login API"""
    data = request.get_json()
    email = data.get('email', '')
    password = data.get('password', '')
    
    # Simple authentication (demo purposes)
    if email == 'demo@dhanalgo.com' and password == 'demo123':
        session['logged_in'] = True
        session['user'] = {'email': email, 'username': 'Demo User'}
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'redirect_url': '/'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid credentials'
        }), 401

@app.route('/api/auth/demo-login', methods=['POST'])
def api_demo_login():
    """Handle demo login"""
    session['logged_in'] = True
    session['user'] = {'email': 'demo@dhanalgo.com', 'username': 'Demo User'}
    return jsonify({
        'success': True,
        'message': 'Demo login successful',
        'redirect_url': '/'
    })

@app.route('/api/portfolio')
def api_portfolio():
    """Get portfolio data"""
    return jsonify({
        'success': True,
        'portfolio': PORTFOLIO_DATA
    })

@app.route('/api/statistics')
def api_statistics():
    """Get trading statistics"""
    return jsonify({
        'success': True,
        'statistics': {
            'total_trades': len(PORTFOLIO_DATA['trades']),
            'win_rate': 0,
            'today_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': PORTFOLIO_DATA['total_pnl']
        }
    })

@app.route('/api/market-overview')
def api_market_overview():
    """Get market overview data"""
    # Generate some random market data for demo
    nifty_base = 25000
    nifty_change = random.uniform(-200, 200)
    vix_base = 18.5
    vix_change = random.uniform(-1, 1)
    
    return jsonify({
        'success': True,
        'market': {
            'nifty_50': {
                'current': nifty_base + nifty_change,
                'change': nifty_change,
                'change_percent': (nifty_change / nifty_base) * 100,
                'high': nifty_base + abs(nifty_change),
                'low': nifty_base - abs(nifty_change)
            },
            'india_vix': {
                'current': vix_base + vix_change,
                'change': vix_change,
                'change_percent': (vix_change / vix_base) * 100
            },
            'market_status': 'OPEN' if is_market_hours() else 'CLOSED'
        }
    })

@app.route('/api/trades/recent')
def api_recent_trades():
    """Get recent trades"""
    return jsonify({
        'success': True,
        'trades': PORTFOLIO_DATA['trades'][-10:]  # Last 10 trades
    })

@app.route('/api/system/status')
def api_system_status():
    """Get system status"""
    return jsonify({
        'success': True,
        'status': {
            'status': 'ACTIVE',
            'market_hours': is_market_hours(),
            'total_positions': len(PORTFOLIO_DATA['positions']),
            'components': {
                'data_feed': 'CONNECTED',
                'ai_models': 'ACTIVE',
                'risk_manager': 'ACTIVE',
                'order_management': 'ACTIVE'
            }
        }
    })

@app.route('/api/trading/start', methods=['POST'])
def api_start_trading():
    """Start trading"""
    return jsonify({
        'success': True,
        'message': 'Trading started (simulation mode)'
    })

@app.route('/api/trading/stop', methods=['POST'])
def api_stop_trading():
    """Stop trading"""
    return jsonify({
        'success': True,
        'message': 'Trading stopped'
    })

@app.route('/api/trade/execute', methods=['POST'])
def api_execute_trade():
    """Execute a trade"""
    data = request.get_json()
    
    # Create a mock trade
    trade = {
        'trade_id': f"T{len(PORTFOLIO_DATA['trades']) + 1:04d}",
        'timestamp': datetime.now().isoformat(),
        'symbol': data.get('symbol', 'NIFTY2500025000CE'),
        'action': data.get('action', 'BUY'),
        'quantity': data.get('quantity', 50),
        'price': data.get('price', 150.0),
        'status': 'COMPLETED',
        'pnl': 0
    }
    
    PORTFOLIO_DATA['trades'].append(trade)
    
    return jsonify({
        'success': True,
        'message': 'Trade executed successfully (simulation)',
        'trade': trade
    })

@app.route('/api/settings', methods=['POST'])
def api_save_settings():
    """Save settings"""
    data = request.get_json()
    return jsonify({
        'success': True,
        'message': 'Settings saved successfully'
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

def is_market_hours():
    """Check if currently in market hours"""
    now = datetime.now()
    
    # Weekend check
    if now.weekday() > 4:  # Saturday = 5, Sunday = 6
        return False
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close

if __name__ == '__main__':
    print("="*60)
    print("AlgoBot - Simple Web Trading Platform")
    print("="*60)
    print("Starting on http://localhost:5000")
    print("Demo Login: demo@dhanalgo.com / demo123")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)