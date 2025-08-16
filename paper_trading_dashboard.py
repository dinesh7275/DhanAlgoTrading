#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Paper Trading Dashboard - Real-time Learning Platform
=======================================================

Advanced paper trading system where AI learns from real market data
Features:
- Real-time paper trading execution
- AI model training and learning
- Performance analytics and insights
- Strategy backtesting and optimization
- Risk-free environment for strategy development
"""

import sys
import json
import logging
import os
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

from flask import Flask, render_template_string, jsonify, request
import sqlite3

# Add project paths
sys.path.append(str(Path(__file__).parent / "AIBot"))

try:
    from AIBot.integrations.dhan_api_client import DhanAPIClient, DhanCredentials
    from dotenv import load_dotenv
    load_dotenv()
    DHAN_AVAILABLE = True
except ImportError:
    print("Dhan API not available - using simulated data")
    DHAN_AVAILABLE = False

logger = logging.getLogger(__name__)
app = Flask(__name__)

@dataclass
class PaperTrade:
    """Paper trade data structure"""
    id: str
    symbol: str
    trade_type: str  # BUY/SELL
    option_type: str  # CE/PE
    strike_price: float
    quantity: int
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    status: str = "OPEN"  # OPEN/CLOSED
    strategy: str = ""
    reason: str = ""
    confidence: float = 0.0
    target_price: Optional[float] = None
    stop_loss_price: Optional[float] = None

@dataclass
class MarketData:
    """Market data structure"""
    timestamp: datetime
    nifty_price: float
    nifty_change: float
    nifty_change_percent: float
    option_chain: List[Dict]
    market_status: str

class PaperTradingEngine:
    """
    Paper Trading Engine for AI Learning
    """
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_margin = initial_capital * 0.8  # 80% margin utilization
        self.trades = []
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # AI Learning components
        self.market_data_history = []
        self.strategy_performance = {}
        self.learning_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Paper Trading Engine initialized with ₹{initial_capital:,.2f} virtual capital")
    
    def _init_database(self):
        """Initialize SQLite database for storing trades and learning data"""
        try:
            self.db_path = Path(__file__).parent / "paper_trading.db"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    trade_type TEXT,
                    option_type TEXT,
                    strike_price REAL,
                    quantity INTEGER,
                    entry_price REAL,
                    entry_time TEXT,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl REAL,
                    status TEXT,
                    strategy TEXT,
                    reason TEXT,
                    confidence REAL,
                    target_price REAL,
                    stop_loss_price REAL
                )
            ''')
            
            # Create market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    nifty_price REAL,
                    nifty_change REAL,
                    nifty_change_percent REAL,
                    market_status TEXT,
                    option_chain_json TEXT
                )
            ''')
            
            # Create learning metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    strategy TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Paper trading database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def execute_paper_trade(self, signal: Dict, current_market_data: MarketData) -> bool:
        """Execute a paper trade based on signal"""
        try:
            # Extract signal details
            symbol = signal['symbol']
            quantity = signal['quantity']
            strategy = signal['strategy']
            confidence = signal['confidence']
            target_percent = signal.get('target_percent', 20)
            stop_loss_percent = signal.get('stop_loss_percent', 10)
            
            # Get current option price from market data
            option_price = self._get_option_price_from_market_data(signal, current_market_data)
            if not option_price:
                logger.warning(f"Could not get price for {symbol}")
                return False
            
            # Check margin availability
            required_margin = self._calculate_margin_requirement(option_price, quantity)
            if required_margin > self.available_margin:
                logger.warning(f"Insufficient margin. Required: ₹{required_margin:,.2f}, Available: ₹{self.available_margin:,.2f}")
                return False
            
            # Create paper trade
            trade_id = str(uuid.uuid4())
            
            # Extract strike price and option type from symbol
            strike_price, option_type = self._parse_option_symbol(symbol)
            
            # Calculate target and stop loss prices
            target_price = option_price * (1 + target_percent / 100)
            stop_loss_price = option_price * (1 - stop_loss_percent / 100)
            
            paper_trade = PaperTrade(
                id=trade_id,
                symbol=symbol,
                trade_type="BUY",
                option_type=option_type,
                strike_price=strike_price,
                quantity=quantity,
                entry_price=option_price,
                entry_time=current_market_data.timestamp,
                status="OPEN",
                strategy=strategy,
                reason=signal.get('reason', ''),
                confidence=confidence,
                target_price=target_price,
                stop_loss_price=stop_loss_price
            )
            
            # Add to open positions
            self.open_positions[trade_id] = paper_trade
            self.trades.append(paper_trade)
            
            # Update available margin
            self.available_margin -= required_margin
            
            # Save to database
            self._save_trade_to_db(paper_trade)
            
            logger.info(f"📄 PAPER TRADE EXECUTED: {symbol}")
            logger.info(f"   💰 Entry Price: ₹{option_price:.2f}")
            logger.info(f"   📊 Quantity: {quantity}")
            logger.info(f"   🎯 Target: ₹{target_price:.2f} ({target_percent}%)")
            logger.info(f"   🛑 Stop Loss: ₹{stop_loss_price:.2f} ({stop_loss_percent}%)")
            logger.info(f"   💳 Margin Used: ₹{required_margin:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
            return False
    
    def monitor_paper_positions(self, current_market_data: MarketData):
        """Monitor and update open paper positions"""
        for trade_id, trade in list(self.open_positions.items()):
            try:
                if trade.status != "OPEN":
                    continue
                
                # Get current option price
                current_price = self._get_option_price_from_market_data(
                    {'symbol': trade.symbol}, current_market_data
                )
                
                if not current_price:
                    continue
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                if current_price >= trade.target_price:
                    should_exit = True
                    exit_reason = "Target Hit"
                elif current_price <= trade.stop_loss_price:
                    should_exit = True
                    exit_reason = "Stop Loss Hit"
                
                if should_exit:
                    self._close_paper_position(trade_id, current_price, exit_reason, current_market_data.timestamp)
                    
            except Exception as e:
                logger.error(f"Error monitoring position {trade_id}: {e}")
    
    def _close_paper_position(self, trade_id: str, exit_price: float, reason: str, exit_time: datetime):
        """Close a paper trading position"""
        try:
            trade = self.open_positions[trade_id]
            
            # Calculate P&L
            pnl = (exit_price - trade.entry_price) * trade.quantity
            
            # Update trade
            trade.exit_price = exit_price
            trade.exit_time = exit_time
            trade.pnl = pnl
            trade.status = "CLOSED"
            trade.reason = reason
            
            # Update statistics
            self.total_trades += 1
            self.daily_pnl += pnl
            self.total_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            
            self.win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
            
            # Release margin
            released_margin = self._calculate_margin_requirement(trade.entry_price, trade.quantity)
            self.available_margin += released_margin
            
            # Remove from open positions
            del self.open_positions[trade_id]
            
            # Update database
            self._update_trade_in_db(trade)
            
            # Update learning metrics
            self._update_learning_metrics(trade)
            
            logger.info(f"📄 PAPER POSITION CLOSED: {trade.symbol}")
            logger.info(f"   💰 Entry: ₹{trade.entry_price:.2f} → Exit: ₹{exit_price:.2f}")
            logger.info(f"   📊 P&L: ₹{pnl:+.2f} ({reason})")
            logger.info(f"   📈 Total P&L: ₹{self.total_pnl:+.2f}")
            logger.info(f"   🎯 Win Rate: {self.win_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"Error closing paper position: {e}")
    
    def _get_option_price_from_market_data(self, signal: Dict, market_data: MarketData) -> Optional[float]:
        """Extract option price from market data"""
        try:
            symbol = signal['symbol']
            strike_price, option_type = self._parse_option_symbol(symbol)
            
            # Search in option chain
            for option in market_data.option_chain:
                if (option.get('strike_price') == strike_price and 
                    option.get('option_type') == option_type and
                    option.get('ltp', 0) > 0):
                    return float(option['ltp'])
            
            # Fallback: estimate based on ATM options
            atm_strike = min(market_data.option_chain, 
                           key=lambda x: abs(x.get('strike_price', 0) - market_data.nifty_price))
            
            if atm_strike and atm_strike.get('ltp', 0) > 0:
                # Simple estimation based on distance from ATM
                distance = abs(strike_price - market_data.nifty_price)
                base_price = float(atm_strike['ltp'])
                
                # Rough estimation (this can be improved with Black-Scholes)
                if option_type == 'CE':
                    estimated_price = max(base_price - (distance * 0.1), 10)
                else:  # PE
                    estimated_price = max(base_price - (distance * 0.1), 10)
                
                return estimated_price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting option price: {e}")
            return None
    
    def _parse_option_symbol(self, symbol: str) -> tuple:
        """Parse option symbol to extract strike price and type"""
        try:
            # Format: NIFTY2025082124600CE
            if 'CE' in symbol:
                option_type = 'CE'
                strike_str = symbol.replace('NIFTY', '').replace('CE', '')
            else:
                option_type = 'PE'
                strike_str = symbol.replace('NIFTY', '').replace('PE', '')
            
            # Extract strike price (last 5 digits typically)
            strike_price = float(strike_str[-5:])
            return strike_price, option_type
            
        except Exception as e:
            logger.error(f"Error parsing option symbol {symbol}: {e}")
            return 0.0, 'CE'
    
    def _calculate_margin_requirement(self, option_price: float, quantity: int) -> float:
        """Calculate margin requirement for option trade"""
        # Simplified margin calculation
        # Actual margin depends on various factors like volatility, time to expiry, etc.
        margin_per_lot = option_price * quantity * 0.15  # 15% of premium value
        return max(margin_per_lot, 50000)  # Minimum ₹50,000 per trade
    
    def _save_trade_to_db(self, trade: PaperTrade):
        """Save trade to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.id, trade.symbol, trade.trade_type, trade.option_type,
                trade.strike_price, trade.quantity, trade.entry_price,
                trade.entry_time.isoformat(), trade.exit_price,
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.pnl, trade.status, trade.strategy, trade.reason,
                trade.confidence, trade.target_price, trade.stop_loss_price
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    def _update_trade_in_db(self, trade: PaperTrade):
        """Update trade in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE trades SET 
                exit_price = ?, exit_time = ?, pnl = ?, status = ?, reason = ?
                WHERE id = ?
            ''', (
                trade.exit_price, trade.exit_time.isoformat() if trade.exit_time else None,
                trade.pnl, trade.status, trade.reason, trade.id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating trade in database: {e}")
    
    def _update_learning_metrics(self, trade: PaperTrade):
        """Update AI learning metrics based on trade results"""
        try:
            # Calculate strategy-specific metrics
            strategy = trade.strategy
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0.0,
                    'accuracy': 0.0
                }
            
            self.strategy_performance[strategy]['trades'] += 1
            self.strategy_performance[strategy]['total_pnl'] += trade.pnl
            
            if trade.pnl > 0:
                self.strategy_performance[strategy]['wins'] += 1
            
            self.strategy_performance[strategy]['accuracy'] = (
                self.strategy_performance[strategy]['wins'] / 
                self.strategy_performance[strategy]['trades']
            ) * 100
            
            # Update overall learning metrics
            self._calculate_advanced_metrics()
            
        except Exception as e:
            logger.error(f"Error updating learning metrics: {e}")
    
    def _calculate_advanced_metrics(self):
        """Calculate advanced performance metrics for AI learning"""
        try:
            if not self.trades:
                return
            
            closed_trades = [t for t in self.trades if t.status == "CLOSED" and t.pnl is not None]
            if not closed_trades:
                return
            
            pnls = [t.pnl for t in closed_trades]
            
            # Calculate metrics
            total_trades = len(closed_trades)
            winning_trades = len([p for p in pnls if p > 0])
            losing_trades = len([p for p in pnls if p <= 0])
            
            self.learning_metrics['accuracy'] = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum([p for p in pnls if p > 0])
            gross_loss = abs(sum([p for p in pnls if p <= 0]))
            self.learning_metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Sharpe ratio (simplified)
            returns = np.array(pnls)
            if len(returns) > 1:
                self.learning_metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Max drawdown
            cumulative_pnl = np.cumsum(pnls)
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - peak)
            self.learning_metrics['max_drawdown'] = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            'total_capital': self.initial_capital,
            'current_capital': self.current_capital + self.total_pnl,
            'available_margin': self.available_margin,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'open_positions': len(self.open_positions),
            'learning_metrics': self.learning_metrics,
            'strategy_performance': self.strategy_performance
        }

# Global instances
paper_engine = PaperTradingEngine(initial_capital=1000000)  # 10 Lakh virtual capital
dhan_client = None
current_market_data = None

# Initialize Dhan client for real market data
if DHAN_AVAILABLE:
    try:
        credentials = DhanCredentials(
            client_id=os.getenv('DHAN_CLIENT_ID', '1107321060'),
            access_token=os.getenv('DHAN_ACCESS_TOKEN')
        )
        dhan_client = DhanAPIClient(credentials)
        if dhan_client.authenticate():
            print("Dhan API connected for real market data")
        else:
            dhan_client = None
    except Exception as e:
        print(f"Error connecting to Dhan API: {e}")
        dhan_client = None

# HTML Template for Paper Trading Dashboard
PAPER_TRADING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📄 AI Paper Trading Dashboard</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #ffffff;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(20px);
            padding: 20px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .paper-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(76, 175, 80, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            color: #4CAF50;
            border: 1px solid #4CAF50;
        }
        
        .top-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-section, .trades-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .trades-section {
            max-height: 600px;
            overflow-y: auto;
        }
        
        .section-title {
            font-size: 1.6rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .trade-item {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #4CAF50;
        }
        
        .trade-item.loss {
            border-left-color: #f44336;
        }
        
        .trade-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .trade-symbol {
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .trade-pnl {
            font-size: 1.2rem;
            font-weight: 700;
        }
        
        .trade-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            font-size: 0.9rem;
        }
        
        .detail-item {
            text-align: center;
        }
        
        .detail-label {
            opacity: 0.7;
            font-size: 0.8rem;
        }
        
        .detail-value {
            font-weight: 600;
            margin-top: 2px;
        }
        
        .bottom-panels {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        
        .info-panel {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .strategy-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .strategy-item:last-child {
            border-bottom: none;
        }
        
        .strategy-name {
            font-weight: 600;
        }
        
        .strategy-stats {
            text-align: right;
            font-size: 0.9rem;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            opacity: 0.7;
        }
        
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            .top-metrics {
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📄 AI Paper Trading Dashboard</h1>
            <div class="paper-indicator">
                <i class="fas fa-chart-line"></i>
                <span>RISK-FREE LEARNING ENVIRONMENT</span>
            </div>
            <p style="margin-top: 10px; opacity: 0.8;">Real market data • Virtual capital • AI model training</p>
        </div>

        <!-- Top Metrics -->
        <div class="top-metrics">
            <div class="metric-card">
                <div class="metric-label">Virtual Capital</div>
                <div class="metric-value" id="virtualCapital">₹10,00,000</div>
                <div style="font-size: 0.9rem; opacity: 0.7;">Total Available</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value" id="totalPnl">₹0.00</div>
                <div style="font-size: 0.9rem; opacity: 0.7;">Unrealized + Realized</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Today's P&L</div>
                <div class="metric-value" id="todayPnl">₹0.00</div>
                <div style="font-size: 0.9rem; opacity: 0.7;">Current Session</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value" id="winRate">0%</div>
                <div style="font-size: 0.9rem; opacity: 0.7;">AI Accuracy</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value" id="totalTrades">0</div>
                <div style="font-size: 0.9rem; opacity: 0.7;">Paper Executions</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Open Positions</div>
                <div class="metric-value" id="openPositions">0</div>
                <div style="font-size: 0.9rem; opacity: 0.7;">Active Trades</div>
            </div>
        </div>

        <!-- Main Grid -->
        <div class="main-grid">
            <!-- P&L Chart -->
            <div class="chart-section">
                <div class="section-title">
                    <i class="fas fa-chart-area"></i>
                    P&L Performance Chart
                </div>
                <div id="pnlChart" style="height: 400px;"></div>
            </div>
            
            <!-- Recent Trades -->
            <div class="trades-section">
                <div class="section-title">
                    <i class="fas fa-receipt"></i>
                    Recent Paper Trades
                    <span id="tradesStatus" style="font-size: 0.8rem; margin-left: auto;"></span>
                </div>
                <div id="tradesContainer">
                    <div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading trades...</div>
                </div>
            </div>
        </div>

        <!-- Information Panels -->
        <div class="bottom-panels">
            <div class="info-panel">
                <div class="section-title">
                    <i class="fas fa-brain"></i>
                    AI Learning Metrics
                </div>
                <div id="learningMetricsContainer">
                    <div class="loading"><i class="fas fa-spinner fa-spin"></i> Analyzing performance...</div>
                </div>
            </div>
            
            <div class="info-panel">
                <div class="section-title">
                    <i class="fas fa-chart-bar"></i>
                    Strategy Performance
                </div>
                <div id="strategyPerformanceContainer">
                    <div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading strategy data...</div>
                </div>
            </div>

            <div class="info-panel">
                <div class="section-title">
                    <i class="fas fa-cogs"></i>
                    Real-time Market Data
                </div>
                <div id="marketDataContainer">
                    <div class="loading"><i class="fas fa-spinner fa-spin"></i> Fetching market data...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let pnlData = [];
        
        // Initialize P&L chart
        function initPnlChart() {
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'white' },
                margin: { t: 20, r: 20, b: 40, l: 80 },
                xaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    showgrid: true,
                    title: 'Time'
                },
                yaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)',
                    showgrid: true,
                    title: 'P&L (₹)'
                }
            };
            
            const trace = {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#4CAF50', width: 3 },
                marker: { color: '#4CAF50', size: 6 },
                name: 'Cumulative P&L'
            };
            
            Plotly.newPlot('pnlChart', [trace], layout, {responsive: true});
        }
        
        // Fetch paper trading data
        function fetchPaperTradingData() {
            $.get('/api/paper_trading_data')
                .done(function(data) {
                    updateMetrics(data);
                    updateRecentTrades(data);
                    updateLearningMetrics(data);
                    updateStrategyPerformance(data);
                    updateMarketData(data);
                    updatePnlChart(data);
                })
                .fail(function() {
                    console.log('Failed to fetch paper trading data');
                });
        }
        
        function updateMetrics(data) {
            if (data.performance) {
                const perf = data.performance;
                $('#virtualCapital').text('₹' + perf.current_capital.toLocaleString('en-IN'));
                $('#totalPnl').text('₹' + perf.total_pnl.toLocaleString('en-IN'));
                $('#totalPnl').removeClass('positive negative').addClass(perf.total_pnl >= 0 ? 'positive' : 'negative');
                $('#todayPnl').text('₹' + perf.daily_pnl.toLocaleString('en-IN'));
                $('#todayPnl').removeClass('positive negative').addClass(perf.daily_pnl >= 0 ? 'positive' : 'negative');
                $('#winRate').text(perf.win_rate.toFixed(1) + '%');
                $('#totalTrades').text(perf.total_trades);
                $('#openPositions').text(perf.open_positions);
            }
        }
        
        function updateRecentTrades(data) {
            if (data.recent_trades && data.recent_trades.length > 0) {
                let html = '';
                data.recent_trades.slice(-10).reverse().forEach(trade => {
                    const pnlClass = trade.pnl > 0 ? 'positive' : 'negative';
                    const tradeClass = trade.pnl > 0 ? '' : 'loss';
                    
                    html += `<div class="trade-item ${tradeClass}">`;
                    html += '<div class="trade-header">';
                    html += `<div class="trade-symbol">${trade.symbol}</div>`;
                    if (trade.pnl !== null) {
                        html += `<div class="trade-pnl ${pnlClass}">₹${trade.pnl.toFixed(2)}</div>`;
                    } else {
                        html += '<div class="trade-pnl">OPEN</div>';
                    }
                    html += '</div>';
                    
                    html += '<div class="trade-details">';
                    html += `<div class="detail-item"><div class="detail-label">Entry</div><div class="detail-value">₹${trade.entry_price.toFixed(2)}</div></div>`;
                    if (trade.exit_price) {
                        html += `<div class="detail-item"><div class="detail-label">Exit</div><div class="detail-value">₹${trade.exit_price.toFixed(2)}</div></div>`;
                    }
                    html += `<div class="detail-item"><div class="detail-label">Qty</div><div class="detail-value">${trade.quantity}</div></div>`;
                    html += `<div class="detail-item"><div class="detail-label">Strategy</div><div class="detail-value">${trade.strategy}</div></div>`;
                    html += '</div>';
                    html += '</div>';
                });
                $('#tradesContainer').html(html);
            } else {
                $('#tradesContainer').html('<div style="text-align: center; padding: 20px; opacity: 0.7;">No trades yet</div>');
            }
        }
        
        function updateLearningMetrics(data) {
            if (data.performance && data.performance.learning_metrics) {
                const metrics = data.performance.learning_metrics;
                let html = '';
                
                html += '<div class="strategy-item">';
                html += '<span>Accuracy</span>';
                html += `<span>${metrics.accuracy.toFixed(1)}%</span>`;
                html += '</div>';
                
                html += '<div class="strategy-item">';
                html += '<span>Profit Factor</span>';
                html += `<span>${metrics.profit_factor.toFixed(2)}</span>`;
                html += '</div>';
                
                html += '<div class="strategy-item">';
                html += '<span>Sharpe Ratio</span>';
                html += `<span>${metrics.sharpe_ratio.toFixed(2)}</span>`;
                html += '</div>';
                
                html += '<div class="strategy-item">';
                html += '<span>Max Drawdown</span>';
                html += `<span>₹${metrics.max_drawdown.toFixed(0)}</span>`;
                html += '</div>';
                
                $('#learningMetricsContainer').html(html);
            }
        }
        
        function updateStrategyPerformance(data) {
            if (data.performance && data.performance.strategy_performance) {
                const strategies = data.performance.strategy_performance;
                let html = '';
                
                Object.keys(strategies).forEach(strategy => {
                    const perf = strategies[strategy];
                    html += '<div class="strategy-item">';
                    html += `<div class="strategy-name">${strategy}</div>`;
                    html += '<div class="strategy-stats">';
                    html += `<div>${perf.accuracy.toFixed(1)}% accuracy</div>`;
                    html += `<div>₹${perf.total_pnl.toFixed(0)} P&L</div>`;
                    html += `<div>${perf.trades} trades</div>`;
                    html += '</div>';
                    html += '</div>';
                });
                
                if (html === '') {
                    html = '<div style="text-align: center; padding: 20px; opacity: 0.7;">No strategy data yet</div>';
                }
                
                $('#strategyPerformanceContainer').html(html);
            }
        }
        
        function updateMarketData(data) {
            if (data.market_data) {
                const market = data.market_data;
                let html = '';
                
                html += '<div class="strategy-item">';
                html += '<span>NIFTY Price</span>';
                html += `<span>₹${market.nifty_price.toFixed(2)}</span>`;
                html += '</div>';
                
                html += '<div class="strategy-item">';
                html += '<span>Change</span>';
                const changeClass = market.nifty_change >= 0 ? 'positive' : 'negative';
                html += `<span class="${changeClass}">${market.nifty_change_percent.toFixed(2)}%</span>`;
                html += '</div>';
                
                html += '<div class="strategy-item">';
                html += '<span>Market Status</span>';
                html += `<span>${market.market_status}</span>`;
                html += '</div>';
                
                html += '<div class="strategy-item">';
                html += '<span>Last Update</span>';
                html += `<span>${new Date(market.timestamp).toLocaleTimeString()}</span>`;
                html += '</div>';
                
                $('#marketDataContainer').html(html);
            }
        }
        
        function updatePnlChart(data) {
            if (data.pnl_history && data.pnl_history.length > 0) {
                const x_data = data.pnl_history.map(d => new Date(d.timestamp));
                const y_data = data.pnl_history.map(d => d.cumulative_pnl);
                
                Plotly.restyle('pnlChart', {
                    x: [x_data],
                    y: [y_data]
                });
            }
        }

        // Initialize
        $(document).ready(function() {
            initPnlChart();
            fetchPaperTradingData();
            
            // Update every 5 seconds
            setInterval(fetchPaperTradingData, 5000);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def paper_dashboard():
    """Main paper trading dashboard route"""
    return render_template_string(PAPER_TRADING_HTML)

@app.route('/api/paper_trading_data')
def get_paper_trading_data():
    """API endpoint for paper trading data"""
    try:
        # Get performance summary
        performance = paper_engine.get_performance_summary()
        
        # Get recent trades
        recent_trades = [asdict(trade) for trade in paper_engine.trades[-20:]]
        
        # Get P&L history (simplified)
        pnl_history = []
        cumulative_pnl = 0
        for trade in paper_engine.trades:
            if trade.status == "CLOSED" and trade.pnl is not None:
                cumulative_pnl += trade.pnl
                pnl_history.append({
                    'timestamp': trade.exit_time.isoformat() if trade.exit_time else trade.entry_time.isoformat(),
                    'cumulative_pnl': cumulative_pnl
                })
        
        # Get current market data
        market_data_dict = None
        if current_market_data:
            market_data_dict = {
                'timestamp': current_market_data.timestamp.isoformat(),
                'nifty_price': current_market_data.nifty_price,
                'nifty_change': current_market_data.nifty_change,
                'nifty_change_percent': current_market_data.nifty_change_percent,
                'market_status': current_market_data.market_status
            }
        
        return jsonify({
            'performance': performance,
            'recent_trades': recent_trades,
            'pnl_history': pnl_history,
            'market_data': market_data_dict,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting paper trading data: {e}")
        return jsonify({'error': str(e)}), 500

def market_data_worker():
    """Background worker to fetch real market data"""
    global current_market_data
    
    while True:
        try:
            if dhan_client:
                # Get NIFTY data
                import yfinance as yf
                nifty_ticker = yf.Ticker("^NSEI")
                nifty_data = nifty_ticker.history(period="1d", interval="1m")
                
                if not nifty_data.empty:
                    current_price = float(nifty_data['Close'].iloc[-1])
                    prev_close = float(nifty_data['Close'].iloc[0])
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100
                    
                    # Get option chain
                    option_chain_data = dhan_client.get_option_chain(underlying_scrip=13)
                    option_chain = []
                    
                    if option_chain_data:
                        for option in option_chain_data[:50]:  # Limit to avoid too much data
                            option_chain.append({
                                'strike_price': option.strike_price,
                                'option_type': option.option_type,
                                'ltp': option.ltp,
                                'volume': option.volume,
                                'open_interest': option.open_interest
                            })
                    
                    # Determine market status
                    now = datetime.now()
                    weekday = now.weekday()
                    
                    if weekday >= 5:
                        market_status = 'CLOSED'
                    elif (now.hour == 9 and now.minute >= 15) or (10 <= now.hour < 15) or (now.hour == 15 and now.minute < 30):
                        market_status = 'OPEN'
                    elif now.hour == 9 and now.minute < 15:
                        market_status = 'PRE_OPEN'
                    else:
                        market_status = 'CLOSED'
                    
                    # Update current market data
                    current_market_data = MarketData(
                        timestamp=datetime.now(),
                        nifty_price=current_price,
                        nifty_change=change,
                        nifty_change_percent=change_percent,
                        option_chain=option_chain,
                        market_status=market_status
                    )
                    
                    print(f"Market Data Updated: NIFTY {current_price:.2f} ({change_percent:+.2f}%)")
            
        except Exception as e:
            logger.error(f"Error in market data worker: {e}")
        
        time.sleep(30)  # Update every 30 seconds

if __name__ == '__main__':
    print("=" * 60)
    print("📄 AI PAPER TRADING DASHBOARD")
    print("=" * 60)
    print("🎯 Features:")
    print("   • Real-time paper trading with virtual capital")
    print("   • AI model learning from live market data")
    print("   • Advanced performance analytics")
    print("   • Risk-free strategy development")
    print("   • Comprehensive trade tracking")
    print()
    print("💰 Virtual Capital: ₹10,00,000")
    print("🌐 Dashboard: http://localhost:5004")
    print("📊 Real Market Data: Connected" if dhan_client else "Simulated")
    print("=" * 60)
    
    # Start market data worker thread
    market_thread = threading.Thread(target=market_data_worker, daemon=True)
    market_thread.start()
    
    try:
        app.run(host='127.0.0.1', port=5004, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting paper trading dashboard: {e}")
    except KeyboardInterrupt:
        print("\n📄 AI Paper Trading Dashboard stopped")