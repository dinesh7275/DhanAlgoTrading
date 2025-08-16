#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Trading Bot Engine
=======================

Complete AI trading bot that executes real trades when started
"""

import sys
import os
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add project paths
sys.path.append(str(Path(__file__).parent / "AIBot"))

try:
    from AIBot.integrations.dhan_api_client import DhanAPIClient, DhanCredentials
    from dotenv import load_dotenv
    load_dotenv()
    DHAN_AVAILABLE = True
except ImportError:
    print("Dhan API not available")
    DHAN_AVAILABLE = False

logger = logging.getLogger(__name__)

class TradingBotEngine:
    """
    Complete AI Trading Bot that executes real trades
    """
    
    def __init__(self, config: Dict[str, Any], dhan_client: DhanAPIClient = None):
        self.config = config
        self.dhan_client = dhan_client
        self.is_running = False
        self.thread = None
        
        # Trading state
        self.positions = {}
        self.trades_today = 0
        self.pnl_today = 0.0
        self.success_count = 0
        
        # Risk management
        self.max_daily_loss = config.get('capital', 10000) * (config.get('stop_loss', 2.0) / 100)
        self.max_positions = config.get('maxPositions', 3)
        
        # 15-minute breakout strategy state
        self.first_15min_completed = False
        self.first_candle_high = 0
        self.first_candle_low = 0
        self.market_open_time = None
        self.breakout_signals_active = False
        
        logger.info(f"Trading Bot initialized with config: {config}")
        logger.info("15-minute breakout strategy enabled")
    
    def start(self):
        """Start the trading bot"""
        if self.is_running:
            return False, "Bot is already running"
        
        if not self.dhan_client:
            return False, "Dhan API client not available"
        
        self.is_running = True
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        
        logger.info("🚀 Trading Bot STARTED - Live trading active!")
        return True, "Trading bot started successfully"
    
    def stop(self):
        """Stop the trading bot"""
        if not self.is_running:
            return False, "Bot is not running"
        
        self.is_running = False
        
        # Close all open positions
        self._close_all_positions()
        
        logger.info("🛑 Trading Bot STOPPED - All positions closed")
        return True, "Trading bot stopped successfully"
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update bot configuration"""
        self.config.update(new_config)
        self.max_daily_loss = self.config.get('capital', 10000) * (self.config.get('stop_loss', 2.0) / 100)
        self.max_positions = self.config.get('maxPositions', 3)
        
        logger.info(f"Bot configuration updated: {self.config}")
        return True, "Configuration updated successfully"
    
    def get_status(self):
        """Get current bot status"""
        return {
            'running': self.is_running,
            'strategy': self.config.get('strategy', 'Conservative'),
            'capital_allocated': self.config.get('capital', 10000),
            'max_positions': self.max_positions,
            'risk_level': self.config.get('riskLevel', 'Medium'),
            'trades_today': self.trades_today,
            'pnl_today': self.pnl_today,
            'success_rate': (self.success_count / max(self.trades_today, 1)) * 100,
            'active_positions': len(self.positions)
        }
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("🔄 Trading loop started")
        logger.info("🎯 15-minute breakout strategy is ACTIVE")
        
        while self.is_running:
            try:
                # Check market hours
                if not self._is_market_open():
                    # Reset daily state when market is closed
                    self._reset_daily_state()
                    logger.info("Market closed - waiting...")
                    time.sleep(60)
                    continue
                
                # Generate trading signals
                signals = self._generate_trading_signals()
                
                # Execute trades based on signals
                for signal in signals:
                    if not self.is_running:
                        break
                    
                    if self._should_execute_trade(signal):
                        self._execute_trade(signal)
                
                # Monitor existing positions
                self._monitor_positions()
                
                # Risk management check
                if abs(self.pnl_today) >= self.max_daily_loss:
                    logger.warning(f"Daily loss limit reached: Rs.{self.pnl_today:.2f}")
                    self._close_all_positions()
                    break
                
                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
        
        logger.info("🔄 Trading loop stopped")
    
    def _reset_daily_state(self):
        """Reset strategy state for new trading day"""
        current_date = datetime.now().date()
        
        if hasattr(self, '_last_reset_date') and self._last_reset_date == current_date:
            return  # Already reset today
        
        self.first_15min_completed = False
        self.first_candle_high = 0
        self.first_candle_low = 0
        self.market_open_time = None
        self.breakout_signals_active = False
        self._last_reset_date = current_date
        
        logger.info("🔄 Daily strategy state reset for new trading session")
    
    def _is_market_open(self):
        """Check if market is open"""
        now = datetime.now()
        weekday = now.weekday()
        
        # Weekend check
        if weekday >= 5:
            return False
        
        # Market hours: 9:15 AM to 3:30 PM
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def _generate_trading_signals(self):
        """Generate AI trading signals with 15-minute breakout strategy"""
        signals = []
        
        try:
            # Get current NIFTY data
            import yfinance as yf
            nifty_ticker = yf.Ticker("^NSEI")
            nifty_data = nifty_ticker.history(period="1d", interval="1m")
            
            if nifty_data.empty:
                return signals
            
            current_price = float(nifty_data['Close'].iloc[-1])
            current_time = datetime.now()
            
            # Check if market just opened (9:15 AM)
            if current_time.hour == 9 and current_time.minute >= 15 and self.market_open_time is None:
                self.market_open_time = current_time
                self.first_15min_completed = False
                self.breakout_signals_active = False
                logger.info("🚀 Market opened - Starting 15-minute breakout strategy")
            
            # Get option chain for signal generation
            option_chain = self.dhan_client.get_option_chain(underlying_scrip=13)
            if not option_chain:
                return signals
            
            # 15-MINUTE BREAKOUT STRATEGY (Primary)
            if self.market_open_time:
                breakout_signals = self._fifteen_minute_breakout_strategy(nifty_data, current_price, option_chain)
                if breakout_signals:
                    signals.extend(breakout_signals)
                    logger.info(f"15-min breakout signals: {len(breakout_signals)}")
            
            # Fallback to other strategies if no breakout signals
            if not signals:
                prev_price = float(nifty_data['Close'].iloc[-2])
                change_percent = ((current_price - prev_price) / prev_price) * 100
                
                strategy = self.config.get('strategy', 'Conservative')
                
                if strategy == 'Aggressive':
                    signals.extend(self._aggressive_strategy(current_price, change_percent, option_chain))
                elif strategy == 'Moderate':
                    signals.extend(self._moderate_strategy(current_price, change_percent, option_chain))
                elif strategy == 'Scalping':
                    signals.extend(self._scalping_strategy(current_price, change_percent, option_chain))
                else:  # Conservative
                    signals.extend(self._conservative_strategy(current_price, change_percent, option_chain))
            
            logger.info(f"Generated {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def _conservative_strategy(self, nifty_price, change_percent, option_chain):
        """Conservative trading strategy"""
        signals = []
        
        # Only trade on significant moves
        if abs(change_percent) > 0.5:
            atm_strike = self._find_atm_strike(nifty_price, option_chain)
            
            if change_percent > 0.5:  # Bullish
                signals.append({
                    'type': 'BUY',
                    'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}CE",
                    'quantity': 75,  # 1 lot (NIFTY lot size = 75)
                    'strategy': 'Conservative Bullish',
                    'confidence': 0.7,
                    'target_percent': 15,
                    'stop_loss_percent': 10
                })
            elif change_percent < -0.5:  # Bearish
                signals.append({
                    'type': 'BUY',
                    'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}PE",
                    'quantity': 75,  # 1 lot (NIFTY lot size = 75)
                    'strategy': 'Conservative Bearish',
                    'confidence': 0.7,
                    'target_percent': 15,
                    'stop_loss_percent': 10
                })
        
        return signals
    
    def _moderate_strategy(self, nifty_price, change_percent, option_chain):
        """Moderate trading strategy"""
        signals = []
        
        if abs(change_percent) > 0.3:
            atm_strike = self._find_atm_strike(nifty_price, option_chain)
            
            if change_percent > 0.3:
                signals.append({
                    'type': 'BUY',
                    'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}CE",
                    'quantity': 150,  # 2 lots (75 x 2)
                    'strategy': 'Moderate Bullish',
                    'confidence': 0.75,
                    'target_percent': 20,
                    'stop_loss_percent': 12
                })
            elif change_percent < -0.3:
                signals.append({
                    'type': 'BUY',
                    'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}PE",
                    'quantity': 150,  # 2 lots (75 x 2)
                    'strategy': 'Moderate Bearish',
                    'confidence': 0.75,
                    'target_percent': 20,
                    'stop_loss_percent': 12
                })
        
        return signals
    
    def _aggressive_strategy(self, nifty_price, change_percent, option_chain):
        """Aggressive trading strategy"""
        signals = []
        
        if abs(change_percent) > 0.2:
            atm_strike = self._find_atm_strike(nifty_price, option_chain)
            
            if change_percent > 0.2:
                signals.append({
                    'type': 'BUY',
                    'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}CE",
                    'quantity': 225,  # 3 lots (75 x 3)
                    'strategy': 'Aggressive Bullish',
                    'confidence': 0.8,
                    'target_percent': 25,
                    'stop_loss_percent': 15
                })
            elif change_percent < -0.2:
                signals.append({
                    'type': 'BUY',
                    'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}PE",
                    'quantity': 225,  # 3 lots (75 x 3)
                    'strategy': 'Aggressive Bearish',
                    'confidence': 0.8,
                    'target_percent': 25,
                    'stop_loss_percent': 15
                })
        
        return signals
    
    def _scalping_strategy(self, nifty_price, change_percent, option_chain):
        """Scalping trading strategy"""
        signals = []
        
        if abs(change_percent) > 0.1:
            atm_strike = self._find_atm_strike(nifty_price, option_chain)
            
            if change_percent > 0.1:
                signals.append({
                    'type': 'BUY',
                    'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}CE",
                    'quantity': 75,  # 1 lot (NIFTY lot size = 75)
                    'strategy': 'Scalping Bullish',
                    'confidence': 0.85,
                    'target_percent': 10,
                    'stop_loss_percent': 5
                })
            elif change_percent < -0.1:
                signals.append({
                    'type': 'BUY',
                    'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}PE",
                    'quantity': 75,  # 1 lot (NIFTY lot size = 75)
                    'strategy': 'Scalping Bearish',
                    'confidence': 0.85,
                    'target_percent': 10,
                    'stop_loss_percent': 5
                })
        
        return signals
    
    def _fifteen_minute_breakout_strategy(self, nifty_data, current_price, option_chain):
        """
        POWERFUL 15-MINUTE BREAKOUT STRATEGY
        
        Logic:
        1. Wait for first 15 minutes after market open (9:15-9:30)
        2. Mark high and low of that first 15-minute candle
        3. If next candle breaks above high = BUY CALL
        4. If next candle breaks below low = BUY PUT
        5. Scalping approach with quick exits
        """
        signals = []
        
        try:
            current_time = datetime.now()
            minutes_since_open = (current_time - self.market_open_time).total_seconds() / 60
            
            # Step 1: Wait for first 15 minutes to complete
            if not self.first_15min_completed and minutes_since_open >= 15:
                # First 15 minutes completed - capture high/low
                market_open_15min = self.market_open_time + timedelta(minutes=15)
                
                # Get 15-minute data from 9:15 to 9:30
                first_15min_data = nifty_data[
                    (nifty_data.index >= self.market_open_time) & 
                    (nifty_data.index <= market_open_15min)
                ]
                
                if not first_15min_data.empty:
                    self.first_candle_high = float(first_15min_data['High'].max())
                    self.first_candle_low = float(first_15min_data['Low'].min())
                    self.first_15min_completed = True
                    self.breakout_signals_active = True
                    
                    logger.info(f"🎯 First 15-min candle completed!")
                    logger.info(f"   📈 High: {self.first_candle_high:.2f}")
                    logger.info(f"   📉 Low: {self.first_candle_low:.2f}")
                    logger.info(f"   🔍 Watching for breakouts...")
            
            # Step 2: Check for breakouts after first 15 minutes
            if self.first_15min_completed and self.breakout_signals_active:
                atm_strike = self._find_atm_strike(current_price, option_chain)
                
                # CALL BUY Signal: Current price breaks above first candle high
                if current_price > self.first_candle_high:
                    signals.append({
                        'type': 'BUY',
                        'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}CE",
                        'quantity': 150,  # 2 lots for breakout (75 x 2)
                        'strategy': '15min Breakout CALL',
                        'confidence': 0.9,
                        'target_percent': 20,  # Quick scalping target
                        'stop_loss_percent': 8,  # Tight stop loss
                        'breakout_type': 'HIGH_BREAK',
                        'breakout_level': self.first_candle_high,
                        'current_price': current_price
                    })
                    
                    logger.info(f"🚀 HIGH BREAKOUT! CALL BUY Signal")
                    logger.info(f"   📊 Breakout Level: {self.first_candle_high:.2f}")
                    logger.info(f"   💰 Current Price: {current_price:.2f}")
                    logger.info(f"   📞 Buying: NIFTY{int(atm_strike)}CE")
                
                # PUT BUY Signal: Current price breaks below first candle low
                elif current_price < self.first_candle_low:
                    signals.append({
                        'type': 'BUY',
                        'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}PE",
                        'quantity': 150,  # 2 lots for breakout (75 x 2)
                        'strategy': '15min Breakout PUT',
                        'confidence': 0.9,
                        'target_percent': 20,  # Quick scalping target
                        'stop_loss_percent': 8,  # Tight stop loss
                        'breakout_type': 'LOW_BREAK',
                        'breakout_level': self.first_candle_low,
                        'current_price': current_price
                    })
                    
                    logger.info(f"🎯 LOW BREAKOUT! PUT BUY Signal")
                    logger.info(f"   📊 Breakout Level: {self.first_candle_low:.2f}")
                    logger.info(f"   💰 Current Price: {current_price:.2f}")
                    logger.info(f"   📞 Buying: NIFTY{int(atm_strike)}PE")
                
                # Disable further breakout signals after first trade to avoid overtrading
                if signals:
                    self.breakout_signals_active = False
                    logger.info("🔒 Breakout signals disabled after first trade (anti-whipsaw)")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in 15-minute breakout strategy: {e}")
            return []
    
    def _find_atm_strike(self, nifty_price, option_chain):
        """Find At-The-Money strike price"""
        strikes = [opt.strike_price for opt in option_chain]
        return min(strikes, key=lambda x: abs(x - nifty_price))
    
    def _get_expiry_string(self):
        """Get next weekly expiry string"""
        # Get next Thursday
        today = datetime.now()
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        
        expiry_date = today + timedelta(days=days_ahead)
        return expiry_date.strftime("%Y%m%d")
    
    def _should_execute_trade(self, signal):
        """Check if trade should be executed"""
        # Check position limits
        if len(self.positions) >= self.max_positions:
            logger.warning("Maximum positions reached")
            return False
        
        # Check daily loss limit
        if abs(self.pnl_today) >= self.max_daily_loss * 0.8:
            logger.warning("Approaching daily loss limit")
            return False
        
        # Check confidence threshold
        if signal['confidence'] < 0.65:
            return False
        
        return True
    
    def _execute_trade(self, signal):
        """Execute a trade"""
        try:
            logger.info(f"🎯 Executing trade: {signal['symbol']} - {signal['strategy']}")
            
            # Get current option price
            option_data = self._get_option_price(signal['symbol'])
            if not option_data:
                logger.error("Could not get option price")
                return
            
            current_price = option_data['ltp']
            if current_price <= 0:
                logger.error("Invalid option price")
                return
            
            # Calculate target and stop loss
            target_price = current_price * (1 + signal['target_percent'] / 100)
            stop_loss_price = current_price * (1 - signal['stop_loss_percent'] / 100)
            
            # Place order with Dhan API
            order_id = self.dhan_client.place_super_order(
                symbol=signal['symbol'],
                transaction_type="BUY",
                quantity=signal['quantity'],
                price=current_price,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                order_type="LIMIT",
                product_type="INTRADAY"
            )
            
            if order_id:
                # Track position
                self.positions[order_id] = {
                    'order_id': order_id,
                    'symbol': signal['symbol'],
                    'entry_price': current_price,
                    'quantity': signal['quantity'],
                    'target_price': target_price,
                    'stop_loss_price': stop_loss_price,
                    'strategy': signal['strategy'],
                    'entry_time': datetime.now(),
                    'status': 'ACTIVE'
                }
                
                self.trades_today += 1
                
                logger.info(f"✅ Trade executed: {order_id} - {signal['symbol']} @ Rs.{current_price}")
                logger.info(f"   Target: Rs.{target_price:.2f}, SL: Rs.{stop_loss_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _get_option_price(self, symbol):
        """Get current option price"""
        try:
            # Extract strike and type from symbol
            # Format: NIFTY2025082124600CE
            parts = symbol.replace('NIFTY', '').replace('CE', '').replace('PE', '')
            strike = float(parts[-5:])  # Last 5 digits are strike
            option_type = 'CE' if 'CE' in symbol else 'PE'
            
            # Get option chain
            option_chain = self.dhan_client.get_option_chain(underlying_scrip=13)
            
            for option in option_chain:
                if option.strike_price == strike and option.option_type == option_type:
                    return {
                        'ltp': option.ltp,
                        'bid': option.bid_price,
                        'ask': option.ask_price,
                        'volume': option.volume
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting option price: {e}")
            return None
    
    def _monitor_positions(self):
        """Monitor existing positions"""
        for order_id, position in list(self.positions.items()):
            try:
                if position['status'] != 'ACTIVE':
                    continue
                
                # Get current price
                option_data = self._get_option_price(position['symbol'])
                if not option_data:
                    continue
                
                current_price = option_data['ltp']
                if current_price <= 0:
                    continue
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                if current_price >= position['target_price']:
                    should_exit = True
                    exit_reason = "Target Hit"
                elif current_price <= position['stop_loss_price']:
                    should_exit = True
                    exit_reason = "Stop Loss Hit"
                
                if should_exit:
                    self._close_position(order_id, current_price, exit_reason)
                
            except Exception as e:
                logger.error(f"Error monitoring position {order_id}: {e}")
    
    def _close_position(self, order_id, exit_price, reason):
        """Close a specific position"""
        try:
            position = self.positions[order_id]
            
            # Calculate P&L
            entry_price = position['entry_price']
            quantity = position['quantity']
            pnl = (exit_price - entry_price) * quantity
            
            # Update position
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['pnl'] = pnl
            position['exit_reason'] = reason
            position['status'] = 'CLOSED'
            
            # Update statistics
            self.pnl_today += pnl
            if pnl > 0:
                self.success_count += 1
            
            logger.info(f"🔄 Position closed: {position['symbol']}")
            logger.info(f"   Entry: Rs.{entry_price:.2f}, Exit: Rs.{exit_price:.2f}")
            logger.info(f"   P&L: Rs.{pnl:+.2f} ({reason})")
            logger.info(f"   Today's P&L: Rs.{self.pnl_today:+.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position {order_id}: {e}")
    
    def _close_all_positions(self):
        """Close all open positions"""
        logger.info("🔄 Closing all positions...")
        
        for order_id, position in list(self.positions.items()):
            if position['status'] == 'ACTIVE':
                try:
                    option_data = self._get_option_price(position['symbol'])
                    if option_data:
                        self._close_position(order_id, option_data['ltp'], "Bot Stopped")
                except Exception as e:
                    logger.error(f"Error closing position {order_id}: {e}")

# Global bot instance
trading_bot = None

def initialize_bot(config, dhan_client):
    """Initialize the trading bot"""
    global trading_bot
    trading_bot = TradingBotEngine(config, dhan_client)
    return trading_bot

def start_bot():
    """Start the trading bot"""
    global trading_bot
    if trading_bot:
        return trading_bot.start()
    return False, "Bot not initialized"

def stop_bot():
    """Stop the trading bot"""
    global trading_bot
    if trading_bot:
        return trading_bot.stop()
    return False, "Bot not initialized"

def configure_bot(new_config):
    """Configure the trading bot"""
    global trading_bot
    if trading_bot:
        return trading_bot.update_config(new_config)
    return False, "Bot not initialized"

def get_bot_status():
    """Get trading bot status"""
    global trading_bot
    if trading_bot:
        return trading_bot.get_status()
    return {
        'running': False,
        'strategy': 'Conservative',
        'capital_allocated': 10000,
        'max_positions': 3,
        'risk_level': 'Medium',
        'trades_today': 0,
        'pnl_today': 0,
        'success_rate': 0,
        'active_positions': 0
    }