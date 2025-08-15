#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Trading Engine
===================

Comprehensive paper trading system with performance tracking, risk management,
and realistic trade execution simulation for testing AI trading strategies.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import threading
import time
import uuid

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    symbol: str
    trade_type: str  # BUY, SELL
    option_type: str  # CE, PE, STOCK
    quantity: int
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED
    pnl: float = 0.0
    charges: float = 0.0
    net_pnl: float = 0.0
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    strategy: str = "Manual"
    confidence: float = 0.0
    market_price: float = 0.0
    
    def __post_init__(self):
        if self.market_price == 0.0:
            self.market_price = self.entry_price

@dataclass
class Position:
    """Current position"""
    symbol: str
    option_type: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    trades: List[str] = field(default_factory=list)
    
@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    return_on_capital: float = 0.0
    
class PaperTradingEngine:
    """
    Advanced paper trading engine with realistic execution simulation
    """
    
    def __init__(self, initial_capital: float = 10000, commission_rate: float = 0.0003):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.commission_rate = commission_rate
        
        # Trading state
        self.trades: Dict[str, Trade] = {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.capital_history: List[Dict] = []
        self.performance_metrics = PerformanceMetrics()
        
        # Indian market charges
        self.charges = {
            'stt_rate': 0.017 / 100,      # STT 0.017% for options
            'exchange_rate': 0.0019 / 100, # Exchange charges 0.0019%
            'gst_rate': 0.18,             # GST 18%
            'sebi_rate': 10 / 10000000,   # SEBI charges ₹10 per crore
            'stamp_duty': 0.003 / 100     # Stamp duty 0.003%
        }
        
        # Risk management
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_position_size = 0.2  # 20% max position size
        self.max_open_trades = 10
        
        # Tracking variables
        self.daily_pnl = 0.0
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Data directory
        self.data_dir = Path("data/paper_trading")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize capital tracking
        self.record_capital_snapshot()
        
        logger.info(f"PaperTradingEngine initialized with ₹{initial_capital:,} capital")
    
    def calculate_charges(self, price: float, quantity: int, trade_type: str = "BUY") -> float:
        """
        Calculate realistic Indian market charges
        """
        try:
            turnover = price * quantity
            
            # STT (Securities Transaction Tax)
            stt = turnover * self.charges['stt_rate']
            
            # Exchange charges
            exchange_charges = turnover * self.charges['exchange_rate']
            
            # SEBI charges
            sebi_charges = turnover * self.charges['sebi_rate']
            
            # Stamp duty (only on buy side)
            stamp_duty = 0
            if trade_type == "BUY":
                stamp_duty = turnover * self.charges['stamp_duty']
            
            # Total charges before GST
            charges_before_gst = exchange_charges + sebi_charges + stamp_duty
            
            # GST on charges (not on STT)
            gst = charges_before_gst * self.charges['gst_rate']
            
            # Brokerage (flat rate)
            brokerage = turnover * self.commission_rate
            
            # Total charges
            total_charges = stt + charges_before_gst + gst + brokerage
            
            return round(total_charges, 2)
        
        except Exception as e:
            logger.error(f"Error calculating charges: {e}")
            return turnover * 0.001  # Fallback 0.1%
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price (simulated)
        """
        try:
            # In real implementation, this would fetch from Dhan API or Yahoo Finance
            # For simulation, we'll use some price movement logic
            base_prices = {
                "NIFTY": 25000,
                "BANKNIFTY": 52000,
            }
            
            # Extract base symbol
            base_symbol = "NIFTY"  # Default
            for key in base_prices.keys():
                if key in symbol.upper():
                    base_symbol = key
                    break
            
            base_price = base_prices.get(base_symbol, 25000)
            
            # Add some realistic price movement (±2% random)
            price_change = np.random.normal(0, 0.02)
            current_price = base_price * (1 + price_change)
            
            # For options, simulate option pricing
            if "CE" in symbol or "PE" in symbol:
                # Extract strike price from symbol (simplified)
                try:
                    parts = symbol.replace("CE", "").replace("PE", "")
                    strike = float(''.join(filter(str.isdigit, parts[-5:])))
                    
                    # Simple option pricing simulation
                    if "CE" in symbol:
                        intrinsic = max(0, current_price - strike)
                        time_value = np.random.uniform(20, 200)  # Random time value
                        option_price = intrinsic + time_value
                    else:  # PE
                        intrinsic = max(0, strike - current_price)
                        time_value = np.random.uniform(20, 200)
                        option_price = intrinsic + time_value
                    
                    return max(option_price, 0.05)  # Minimum 5 paisa
                
                except:
                    return np.random.uniform(50, 300)  # Random option price
            
            return current_price
        
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 100.0  # Fallback price
    
    def place_order(self, symbol: str, trade_type: str, quantity: int, 
                   order_type: str = "MARKET", price: float = None,
                   stop_loss: float = None, target: float = None,
                   strategy: str = "Manual", confidence: float = 0.0) -> str:
        """
        Place a paper trade order
        """
        try:
            # Validate inputs
            if trade_type not in ["BUY", "SELL"]:
                raise ValueError("trade_type must be BUY or SELL")
            
            if quantity <= 0:
                raise ValueError("quantity must be positive")
            
            # Check risk limits
            if not self.check_risk_limits(symbol, trade_type, quantity, price):
                raise ValueError("Order rejected due to risk limits")
            
            # Get execution price
            if order_type == "MARKET" or price is None:
                execution_price = self.get_current_price(symbol)
            else:
                execution_price = price
                # Add some slippage for limit orders
                slippage = np.random.normal(0, 0.001)  # 0.1% slippage
                execution_price *= (1 + slippage)
            
            # Calculate charges
            charges = self.calculate_charges(execution_price, quantity, trade_type)
            
            # Check if sufficient capital
            required_capital = (execution_price * quantity) + charges
            if trade_type == "BUY" and required_capital > self.available_capital:
                raise ValueError(f"Insufficient capital. Required: ₹{required_capital:,.2f}, Available: ₹{self.available_capital:,.2f}")
            
            # Create trade
            trade_id = str(uuid.uuid4())[:8]
            
            # Determine option type
            option_type = "STOCK"
            if "CE" in symbol.upper():
                option_type = "CE"
            elif "PE" in symbol.upper():
                option_type = "PE"
            
            trade = Trade(
                trade_id=trade_id,
                symbol=symbol,
                trade_type=trade_type,
                option_type=option_type,
                quantity=quantity,
                entry_price=execution_price,
                entry_time=datetime.now(),
                charges=charges,
                stop_loss=stop_loss,
                target=target,
                strategy=strategy,
                confidence=confidence,
                market_price=execution_price
            )
            
            # Execute trade
            self.execute_trade(trade)
            
            logger.info(f"Order placed: {trade_id} - {trade_type} {quantity} {symbol} @ ₹{execution_price:.2f}")
            
            return trade_id
        
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    def execute_trade(self, trade: Trade):
        """
        Execute a trade and update positions
        """
        try:
            # Store trade
            self.trades[trade.trade_id] = trade
            
            # Update capital
            if trade.trade_type == "BUY":
                self.available_capital -= (trade.entry_price * trade.quantity + trade.charges)
            else:  # SELL
                self.available_capital += (trade.entry_price * trade.quantity - trade.charges)
            
            # Update positions
            self.update_position(trade)
            
            # Add to history
            self.trade_history.append({
                'timestamp': trade.entry_time,
                'trade_id': trade.trade_id,
                'action': 'OPEN',
                'symbol': trade.symbol,
                'type': trade.trade_type,
                'quantity': trade.quantity,
                'price': trade.entry_price,
                'charges': trade.charges,
                'strategy': trade.strategy
            })
            
            # Record capital snapshot
            self.record_capital_snapshot()
            
        except Exception as e:
            logger.error(f"Error executing trade {trade.trade_id}: {e}")
            raise
    
    def update_position(self, trade: Trade):
        """
        Update position based on trade
        """
        position_key = f"{trade.symbol}_{trade.option_type}"
        
        if position_key not in self.positions:
            # New position
            self.positions[position_key] = Position(
                symbol=trade.symbol,
                option_type=trade.option_type,
                quantity=0,
                avg_price=0.0,
                current_price=trade.market_price,
                unrealized_pnl=0.0,
                trades=[trade.trade_id]
            )
        else:
            self.positions[position_key].trades.append(trade.trade_id)
        
        position = self.positions[position_key]
        
        if trade.trade_type == "BUY":
            # Add to position
            total_cost = (position.avg_price * position.quantity) + (trade.entry_price * trade.quantity)
            total_quantity = position.quantity + trade.quantity
            
            if total_quantity > 0:
                position.avg_price = total_cost / total_quantity
            position.quantity = total_quantity
            
        else:  # SELL
            # Reduce position
            if position.quantity >= trade.quantity:
                position.quantity -= trade.quantity
                
                # If position is closed, remove it
                if position.quantity == 0:
                    del self.positions[position_key]
                    return
            else:
                # Partial close or reverse position
                remaining_sell = trade.quantity - position.quantity
                position.quantity = -remaining_sell  # Short position
        
        # Update current price and unrealized P&L
        position.current_price = trade.market_price
        position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
    
    def close_trade(self, trade_id: str, exit_price: float = None) -> Dict[str, Any]:
        """
        Close an open trade
        """
        try:
            if trade_id not in self.trades:
                raise ValueError(f"Trade {trade_id} not found")
            
            trade = self.trades[trade_id]
            
            if trade.status != "OPEN":
                raise ValueError(f"Trade {trade_id} is already {trade.status}")
            
            # Get exit price
            if exit_price is None:
                exit_price = self.get_current_price(trade.symbol)
            
            # Calculate exit charges
            exit_charges = self.calculate_charges(exit_price, trade.quantity, 
                                                "SELL" if trade.trade_type == "BUY" else "BUY")
            
            # Update trade
            trade.exit_price = exit_price
            trade.exit_time = datetime.now()
            trade.status = "CLOSED"
            trade.charges += exit_charges
            
            # Calculate P&L
            if trade.trade_type == "BUY":
                trade.pnl = (exit_price - trade.entry_price) * trade.quantity
            else:  # SELL
                trade.pnl = (trade.entry_price - exit_price) * trade.quantity
            
            trade.net_pnl = trade.pnl - trade.charges
            
            # Update capital
            if trade.trade_type == "BUY":
                self.available_capital += (exit_price * trade.quantity - exit_charges)
            else:  # SELL
                self.available_capital += (trade.entry_price * trade.quantity + trade.pnl - exit_charges)
            
            self.current_capital = self.available_capital + self.get_total_unrealized_pnl()
            
            # Update daily P&L
            self.daily_pnl += trade.net_pnl
            
            # Update position
            self.update_position_on_close(trade)
            
            # Add to history
            self.trade_history.append({
                'timestamp': trade.exit_time,
                'trade_id': trade.trade_id,
                'action': 'CLOSE',
                'symbol': trade.symbol,
                'type': 'SELL' if trade.trade_type == 'BUY' else 'BUY',
                'quantity': trade.quantity,
                'price': exit_price,
                'pnl': trade.net_pnl,
                'charges': exit_charges
            })
            
            # Update performance metrics
            self.update_performance_metrics(trade)
            
            # Record capital snapshot
            self.record_capital_snapshot()
            
            logger.info(f"Trade closed: {trade_id} - P&L: ₹{trade.net_pnl:.2f}")
            
            return {
                'trade_id': trade_id,
                'pnl': trade.net_pnl,
                'exit_price': exit_price,
                'charges': trade.charges
            }
        
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            raise
    
    def update_position_on_close(self, trade: Trade):
        """
        Update position when trade is closed
        """
        position_key = f"{trade.symbol}_{trade.option_type}"
        
        if position_key in self.positions:
            position = self.positions[position_key]
            
            if trade.trade_type == "BUY":
                # Closing a long position (selling)
                position.quantity -= trade.quantity
            else:
                # Closing a short position (buying back)
                position.quantity += trade.quantity
            
            # Remove position if quantity is zero
            if position.quantity == 0:
                del self.positions[position_key]
    
    def close_all_positions(self) -> List[Dict[str, Any]]:
        """
        Close all open positions
        """
        results = []
        
        # Get all open trades
        open_trades = [trade for trade in self.trades.values() if trade.status == "OPEN"]
        
        for trade in open_trades:
            try:
                result = self.close_trade(trade.trade_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error closing trade {trade.trade_id}: {e}")
        
        logger.info(f"Closed {len(results)} positions")
        return results
    
    def check_risk_limits(self, symbol: str, trade_type: str, quantity: int, price: float) -> bool:
        """
        Check if trade violates risk limits
        """
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.initial_capital * self.max_daily_loss:
                logger.warning("Daily loss limit exceeded")
                return False
            
            # Check maximum open trades
            open_trades_count = len([t for t in self.trades.values() if t.status == "OPEN"])
            if open_trades_count >= self.max_open_trades:
                logger.warning("Maximum open trades limit exceeded")
                return False
            
            # Check position size limit
            if price:
                position_value = price * quantity
                if position_value > self.current_capital * self.max_position_size:
                    logger.warning("Position size limit exceeded")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def update_performance_metrics(self, closed_trade: Trade):
        """
        Update performance metrics after trade closure
        """
        try:
            metrics = self.performance_metrics
            
            # Basic counts
            metrics.total_trades += 1
            
            if closed_trade.net_pnl > 0:
                metrics.winning_trades += 1
                metrics.gross_profit += closed_trade.net_pnl
                metrics.max_win = max(metrics.max_win, closed_trade.net_pnl)
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                metrics.max_consecutive_wins = max(metrics.max_consecutive_wins, self.consecutive_wins)
            else:
                metrics.losing_trades += 1
                metrics.gross_loss += abs(closed_trade.net_pnl)
                metrics.max_loss = max(metrics.max_loss, abs(closed_trade.net_pnl))
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                metrics.max_consecutive_losses = max(metrics.max_consecutive_losses, self.consecutive_losses)
            
            # Calculate derived metrics
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades
            
            metrics.total_pnl = metrics.gross_profit - metrics.gross_loss
            
            if metrics.gross_loss > 0:
                metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
            
            if metrics.winning_trades > 0:
                metrics.avg_win = metrics.gross_profit / metrics.winning_trades
            
            if metrics.losing_trades > 0:
                metrics.avg_loss = metrics.gross_loss / metrics.losing_trades
            
            # Update drawdown
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
            
            self.current_drawdown = self.peak_capital - self.current_capital
            metrics.max_drawdown = max(metrics.max_drawdown, self.current_drawdown)
            
            if self.peak_capital > 0:
                drawdown_percent = (self.current_drawdown / self.peak_capital) * 100
                metrics.max_drawdown_percent = max(metrics.max_drawdown_percent, drawdown_percent)
            
            # Return on capital
            metrics.return_on_capital = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            
            # Calculate Sharpe ratio (simplified)
            if len(self.capital_history) > 1:
                returns = [entry['pnl_change'] for entry in self.capital_history if 'pnl_change' in entry]
                if len(returns) > 1:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    if std_return > 0:
                        metrics.sharpe_ratio = avg_return / std_return
            
            # Calmar ratio
            if metrics.max_drawdown_percent > 0:
                annual_return = metrics.return_on_capital  # Simplified
                metrics.calmar_ratio = annual_return / metrics.max_drawdown_percent
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_total_unrealized_pnl(self) -> float:
        """
        Calculate total unrealized P&L from open positions
        """
        total_unrealized = 0.0
        
        for position in self.positions.values():
            # Update current price
            current_price = self.get_current_price(position.symbol)
            position.current_price = current_price
            
            # Calculate unrealized P&L
            position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
            total_unrealized += position.unrealized_pnl
        
        return total_unrealized
    
    def record_capital_snapshot(self):
        """
        Record capital snapshot for performance tracking
        """
        try:
            unrealized_pnl = self.get_total_unrealized_pnl()
            total_capital = self.available_capital + unrealized_pnl
            
            snapshot = {
                'timestamp': datetime.now(),
                'available_capital': self.available_capital,
                'unrealized_pnl': unrealized_pnl,
                'total_capital': total_capital,
                'daily_pnl': self.daily_pnl,
                'drawdown': self.current_drawdown,
                'open_positions': len(self.positions)
            }
            
            # Add P&L change if not first record
            if len(self.capital_history) > 0:
                prev_capital = self.capital_history[-1]['total_capital']
                snapshot['pnl_change'] = total_capital - prev_capital
            else:
                snapshot['pnl_change'] = total_capital - self.initial_capital
            
            self.capital_history.append(snapshot)
            self.current_capital = total_capital
            
            # Keep only last 1000 records
            if len(self.capital_history) > 1000:
                self.capital_history = self.capital_history[-1000:]
        
        except Exception as e:
            logger.error(f"Error recording capital snapshot: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary
        """
        try:
            # Update current capital
            unrealized_pnl = self.get_total_unrealized_pnl()
            total_capital = self.available_capital + unrealized_pnl
            
            # Open trades summary
            open_trades = [trade for trade in self.trades.values() if trade.status == "OPEN"]
            closed_trades = [trade for trade in self.trades.values() if trade.status == "CLOSED"]
            
            # Position summary
            positions_summary = []
            for position in self.positions.values():
                positions_summary.append({
                    'symbol': position.symbol,
                    'type': position.option_type,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_percent': (position.unrealized_pnl / (position.avg_price * abs(position.quantity))) * 100 if position.avg_price > 0 else 0
                })
            
            return {
                'timestamp': datetime.now(),
                'capital': {
                    'initial': self.initial_capital,
                    'current': total_capital,
                    'available': self.available_capital,
                    'unrealized_pnl': unrealized_pnl,
                    'total_return': total_capital - self.initial_capital,
                    'total_return_percent': ((total_capital - self.initial_capital) / self.initial_capital) * 100,
                    'daily_pnl': self.daily_pnl
                },
                'trades': {
                    'total': len(self.trades),
                    'open': len(open_trades),
                    'closed': len(closed_trades),
                    'winning': self.performance_metrics.winning_trades,
                    'losing': self.performance_metrics.losing_trades
                },
                'positions': positions_summary,
                'performance': asdict(self.performance_metrics),
                'risk': {
                    'max_drawdown': self.performance_metrics.max_drawdown,
                    'max_drawdown_percent': self.performance_metrics.max_drawdown_percent,
                    'current_drawdown': self.current_drawdown,
                    'daily_loss_limit': self.initial_capital * self.max_daily_loss,
                    'daily_loss_used': abs(self.daily_pnl) if self.daily_pnl < 0 else 0
                }
            }
        
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {}
    
    def export_trading_data(self, filename: str = None) -> str:
        """
        Export all trading data to JSON file
        """
        try:
            if not filename:
                filename = f"paper_trading_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Prepare export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'engine_config': {
                    'initial_capital': self.initial_capital,
                    'commission_rate': self.commission_rate,
                    'charges': self.charges
                },
                'portfolio_summary': self.get_portfolio_summary(),
                'trades': {trade_id: asdict(trade) for trade_id, trade in self.trades.items()},
                'capital_history': self.capital_history,
                'trade_history': self.trade_history,
                'performance_metrics': asdict(self.performance_metrics)
            }
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            export_data = convert_datetime(export_data)
            
            # Save to file
            export_path = self.data_dir / filename
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Trading data exported to {export_path}")
            return str(export_path)
        
        except Exception as e:
            logger.error(f"Error exporting trading data: {e}")
            return ""
    
    def reset_daily_metrics(self):
        """
        Reset daily metrics (call at market open)
        """
        self.daily_pnl = 0.0
        logger.info("Daily metrics reset")
    
    def simulate_market_close(self):
        """
        Simulate end of trading day activities
        """
        try:
            # Update all position prices
            for position in self.positions.values():
                position.current_price = self.get_current_price(position.symbol)
                position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
            
            # Record final snapshot of the day
            self.record_capital_snapshot()
            
            # Update performance metrics
            total_capital = self.available_capital + self.get_total_unrealized_pnl()
            self.current_capital = total_capital
            
            logger.info(f"Market close simulation - Total capital: ₹{total_capital:,.2f}, Daily P&L: ₹{self.daily_pnl:,.2f}")
        
        except Exception as e:
            logger.error(f"Error in market close simulation: {e}")
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """
        Get trade by ID
        """
        return self.trades.get(trade_id)
    
    def get_open_trades(self) -> List[Trade]:
        """
        Get all open trades
        """
        return [trade for trade in self.trades.values() if trade.status == "OPEN"]
    
    def get_closed_trades(self) -> List[Trade]:
        """
        Get all closed trades
        """
        return [trade for trade in self.trades.values() if trade.status == "CLOSED"]
    
    def cancel_trade(self, trade_id: str) -> bool:
        """
        Cancel an open trade (mark as cancelled)
        """
        try:
            if trade_id not in self.trades:
                return False
            
            trade = self.trades[trade_id]
            if trade.status != "OPEN":
                return False
            
            # Mark as cancelled
            trade.status = "CANCELLED"
            
            # Restore capital if it was a buy order
            if trade.trade_type == "BUY":
                self.available_capital += (trade.entry_price * trade.quantity + trade.charges)
            
            # Remove from positions
            self.update_position_on_close(trade)
            
            logger.info(f"Trade cancelled: {trade_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error cancelling trade {trade_id}: {e}")
            return False

def create_sample_trading_session():
    """
    Create a sample trading session for demonstration
    """
    # Initialize paper trading engine
    engine = PaperTradingEngine(initial_capital=50000)
    
    # Place some sample trades
    try:
        # Buy NIFTY Call option
        trade1_id = engine.place_order(
            symbol="NIFTY2412525000CE",
            trade_type="BUY",
            quantity=50,
            strategy="AI Signal - Bullish Breakout",
            confidence=0.75
        )
        
        time.sleep(1)  # Simulate time passage
        
        # Buy NIFTY Put option (hedge)
        trade2_id = engine.place_order(
            symbol="NIFTY2412524900PE",
            trade_type="BUY",
            quantity=25,
            strategy="AI Signal - Hedge",
            confidence=0.60
        )
        
        time.sleep(2)
        
        # Close first trade with profit
        engine.close_trade(trade1_id, exit_price=engine.get_current_price("NIFTY2412525000CE") * 1.1)
        
        time.sleep(1)
        
        # Place another trade
        trade3_id = engine.place_order(
            symbol="BANKNIFTY2412552000CE",
            trade_type="BUY",
            quantity=25,
            strategy="AI Signal - Momentum",
            confidence=0.65
        )
        
        # Get portfolio summary
        summary = engine.get_portfolio_summary()
        
        print("\n" + "="*60)
        print("📊 PAPER TRADING SESSION SUMMARY")
        print("="*60)
        print(f"💰 Initial Capital: ₹{summary['capital']['initial']:,}")
        print(f"💰 Current Capital: ₹{summary['capital']['current']:,.2f}")
        print(f"📈 Total Return: ₹{summary['capital']['total_return']:,.2f} ({summary['capital']['total_return_percent']:+.2f}%)")
        print(f"📊 Total Trades: {summary['trades']['total']} (Open: {summary['trades']['open']}, Closed: {summary['trades']['closed']})")
        print(f"🎯 Win Rate: {summary['performance']['win_rate']:.1%}")
        print(f"📉 Max Drawdown: ₹{summary['risk']['max_drawdown']:,.2f} ({summary['risk']['max_drawdown_percent']:.2f}%)")
        
        if summary['positions']:
            print("\n🔍 OPEN POSITIONS:")
            for pos in summary['positions']:
                print(f"   {pos['symbol']} | Qty: {pos['quantity']} | P&L: ₹{pos['unrealized_pnl']:,.2f}")
        
        # Export data
        export_file = engine.export_trading_data()
        print(f"\n💾 Data exported to: {export_file}")
        print("="*60)
        
        return engine
    
    except Exception as e:
        print(f"Error in sample session: {e}")
        return engine

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run sample trading session
    engine = create_sample_trading_session()