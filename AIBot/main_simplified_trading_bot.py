#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified AI Trading Bot - Core Functionality
==============================================

Streamlined version focusing on essential features:
- Basic signal generation
- Paper trading simulation
- Simple dashboard
- Core monitoring

Advanced features are commented out for easier deployment.
"""

import sys
import logging
import threading
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import json
import pandas as pd
import numpy as np
import yfinance as yf

# Add project paths
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)

class SimpleSignalGenerator:
    """
    Simplified signal generator using basic technical analysis
    """
    
    def __init__(self, symbol: str = "^NSEI", capital: float = 10000):
        self.symbol = symbol
        self.capital = capital
        self.current_price = 0.0
        
    def calculate_simple_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic technical indicators"""
        if df.empty or len(df) < 20:
            return {}
        
        indicators = {}
        close = df['Close']
        
        # Simple Moving Averages
        indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Price data
        indicators['current_price'] = close.iloc[-1]
        indicators['volume'] = df['Volume'].iloc[-1]
        
        return indicators
    
    def generate_simple_signal(self) -> Dict[str, Any]:
        """Generate simple trading signal"""
        try:
            # Fetch recent data
            df = yf.download(self.symbol, period="60d", interval="1d", progress=False)
            if df.empty:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No data'}
            
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            indicators = self.calculate_simple_indicators(df)
            
            if not indicators:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No indicators'}
            
            self.current_price = indicators['current_price']
            
            # Simple signal logic
            signal = 'HOLD'
            confidence = 0.5
            reason = []
            
            # RSI signals
            if indicators['rsi'] < 30:
                signal = 'BUY'
                confidence = 0.7
                reason.append(f"RSI oversold: {indicators['rsi']:.1f}")
            elif indicators['rsi'] > 70:
                signal = 'SELL'
                confidence = 0.7
                reason.append(f"RSI overbought: {indicators['rsi']:.1f}")
            
            # Moving average signals
            if indicators['current_price'] > indicators['sma_20'] > indicators['sma_50']:
                if signal == 'HOLD':
                    signal = 'BUY'
                    confidence = 0.6
                reason.append("Price above moving averages")
            elif indicators['current_price'] < indicators['sma_20'] < indicators['sma_50']:
                if signal == 'HOLD':
                    signal = 'SELL'
                    confidence = 0.6
                reason.append("Price below moving averages")
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': ' | '.join(reason) if reason else 'No clear signal',
                'price': indicators['current_price'],
                'indicators': indicators,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}

class SimplePaperTrading:
    """
    Basic paper trading engine
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.trades = []
        self.total_pnl = 0.0
        
    def place_trade(self, signal_data: Dict[str, Any]) -> bool:
        """Place a simple paper trade"""
        try:
            if signal_data['signal'] == 'HOLD':
                return False
            
            # Simple position sizing (10% of capital)
            position_size = self.current_capital * 0.1
            quantity = int(position_size / signal_data['price'])
            
            if quantity <= 0:
                return False
            
            trade = {
                'id': len(self.trades) + 1,
                'timestamp': datetime.now(),
                'signal': signal_data['signal'],
                'price': signal_data['price'],
                'quantity': quantity,
                'value': signal_data['price'] * quantity,
                'status': 'OPEN'
            }
            
            self.trades.append(trade)
            self.positions.append(trade)
            
            logger.info(f"Paper trade: {trade['signal']} {quantity} @ Rs.{signal_data['price']:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return False
    
    def update_positions(self, current_price: float):
        """Update open positions with current price"""
        for position in self.positions:
            if position['status'] == 'OPEN':
                if position['signal'] == 'BUY':
                    pnl = (current_price - position['price']) * position['quantity']
                else:  # SELL
                    pnl = (position['price'] - current_price) * position['quantity']
                
                position['current_price'] = current_price
                position['pnl'] = pnl
                
                # Simple exit conditions (±5% or 24 hours)
                pnl_percent = (pnl / position['value']) * 100
                hours_open = (datetime.now() - position['timestamp']).total_seconds() / 3600
                
                if abs(pnl_percent) > 5 or hours_open > 24:
                    position['status'] = 'CLOSED'
                    position['exit_price'] = current_price
                    position['exit_time'] = datetime.now()
                    self.total_pnl += pnl
                    
                    logger.info(f"Closed trade {position['id']}: P&L Rs.{pnl:.2f} ({pnl_percent:+.1f}%)")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get trading summary"""
        open_positions = [p for p in self.positions if p['status'] == 'OPEN']
        closed_positions = [p for p in self.positions if p['status'] == 'CLOSED']
        
        total_trades = len(self.trades)
        winning_trades = len([p for p in closed_positions if p.get('pnl', 0) > 0])
        win_rate = (winning_trades / len(closed_positions)) * 100 if closed_positions else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'total_trades': total_trades,
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'win_rate': win_rate,
            'positions': open_positions
        }

class SimplifiedTradingBot:
    """
    Simplified AI Trading Bot with core functionality
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        
        # Core components
        self.signal_generator = SimpleSignalGenerator(
            symbol=config.get('symbol', '^NSEI'),
            capital=config.get('initial_capital', 10000)
        )
        self.paper_trading = SimplePaperTrading(
            initial_capital=config.get('initial_capital', 10000)
        )
        
        # Stats
        self.session_stats = {
            'start_time': datetime.now(),
            'signals_generated': 0,
            'trades_executed': 0
        }
        
        logger.info("Simplified Trading Bot initialized")
    
    def start_trading(self):
        """Start the simplified trading bot"""
        if self.is_running:
            logger.warning("Bot is already running")
            return
        
        self.is_running = True
        logger.info("Starting Simplified AI Trading Bot...")
        
        # Print startup info
        self._print_startup_info()
        
        # Start main loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop_trading()
    
    def _print_startup_info(self):
        """Print startup information"""
        print("\n" + "="*60)
        print("SIMPLIFIED AI TRADING BOT")
        print("="*60)
        print(f"Symbol: {self.config.get('symbol', '^NSEI')}")
        print(f"Capital: Rs.{self.config.get('initial_capital', 10000):,}")
        print(f"Update Interval: {self.config.get('update_interval', 60)} seconds")
        print("="*60)
        print("Press Ctrl+C to stop")
        print("="*60)
    
    def _main_loop(self):
        """Main trading loop"""
        update_interval = self.config.get('update_interval', 60)  # 1 minute default
        
        while self.is_running:
            try:
                # Generate signal
                signal_data = self.signal_generator.generate_simple_signal()
                self.session_stats['signals_generated'] += 1
                
                # Print signal info
                print(f"\n[{signal_data['timestamp'].strftime('%H:%M:%S')}] Signal: {signal_data['signal']} "
                      f"(Confidence: {signal_data['confidence']:.1%})")
                print(f"Price: Rs.{signal_data['price']:.2f} | Reason: {signal_data['reason']}")
                
                # Execute trade if signal is strong enough
                if signal_data['confidence'] > 0.6:
                    if self.paper_trading.place_trade(signal_data):
                        self.session_stats['trades_executed'] += 1
                
                # Update positions
                self.paper_trading.update_positions(signal_data['price'])
                
                # Print summary every 5 signals
                if self.session_stats['signals_generated'] % 5 == 0:
                    self._print_summary()
                
                # Wait for next update
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _print_summary(self):
        """Print trading summary"""
        summary = self.paper_trading.get_summary()
        runtime = datetime.now() - self.session_stats['start_time']
        
        print("\n" + "-"*60)
        print("TRADING SUMMARY")
        print("-"*60)
        print(f"Runtime: {runtime}")
        print(f"Capital: Rs.{summary['current_capital']:,.2f}")
        print(f"Total P&L: Rs.{summary['total_pnl']:+,.2f}")
        print(f"Signals: {self.session_stats['signals_generated']}")
        print(f"Trades: {summary['total_trades']}")
        print(f"Open Positions: {summary['open_positions']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        
        if summary['positions']:
            print("\nOpen Positions:")
            for pos in summary['positions']:
                pnl = pos.get('pnl', 0)
                print(f"  {pos['signal']} {pos['quantity']} @ Rs.{pos['price']:.2f} | P&L: Rs.{pnl:+.2f}")
        
        print("-"*60)
    
    def stop_trading(self):
        """Stop the trading bot"""
        self.is_running = False
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        summary = self.paper_trading.get_summary()
        runtime = datetime.now() - self.session_stats['start_time']
        
        print(f"Total Runtime: {runtime}")
        print(f"Final Capital: Rs.{summary['current_capital']:,.2f}")
        print(f"Total P&L: Rs.{summary['total_pnl']:+,.2f}")
        print(f"Return: {(summary['total_pnl']/summary['initial_capital'])*100:+.2f}%")
        print(f"Total Signals: {self.session_stats['signals_generated']}")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print("="*60)
        
        logger.info("Simplified Trading Bot stopped")

def load_simple_config() -> Dict[str, Any]:
    """Load simple configuration"""
    return {
        'symbol': '^NSEI',  # NIFTY 50
        'initial_capital': 10000,
        'update_interval': 60,  # 1 minute
        'min_confidence': 0.6
    }

def main():
    """Main entry point for simplified bot"""
    parser = argparse.ArgumentParser(description='Simplified AI Trading Bot')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simple_trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load configuration
    config = load_simple_config()
    config['update_interval'] = args.interval
    
    # Create and start bot
    try:
        bot = SimplifiedTradingBot(config)
        
        # Setup signal handler
        def signal_handler(signum, frame):
            print("\nShutdown signal received...")
            bot.stop_trading()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start trading
        bot.start_trading()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# COMMENTED OUT ADVANCED FEATURES
# ================================
# The following features are available but commented out for simplicity:

"""
# Advanced ML Models (uncomment to enable)
# from models.enhanced_learning.advanced_market_learner import AdvancedMarketLearner
# from models.enhanced_learning.multi_timeframe_analyzer import MultiTimeframeAnalyzer
# from models.enhanced_learning.candlestick_pattern_ml import CandlestickPatternML

# Real-time Monitoring (uncomment to enable)
# from monitoring.real_time_indicator_monitor import RealTimeIndicatorMonitor

# Adaptive Learning (uncomment to enable)
# from learning.adaptive_learning_system import AdaptiveLearningSystem

# Live Charts (uncomment to enable)
# from visualization.live_candlestick_chart import LiveCandlestickChart

# Dhan API Integration (uncomment to enable)
# from integrations.dhan_api_client import DhanAPIClient, DhanCredentials

# Advanced Paper Trading (uncomment to enable)
# from trading.paper_trading_engine import PaperTradingEngine

# Enhanced Dashboard (uncomment to enable)
# from enhanced_dashboard import app as dashboard_app

To enable advanced features:
1. Uncomment the imports above
2. Install additional dependencies (tensorflow, plotly, etc.)
3. Replace SimplifiedTradingBot with EnhancedAITradingBot
4. Update the configuration accordingly
"""