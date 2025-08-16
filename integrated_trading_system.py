#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Trading System - Real + Paper Trading
===============================================

Combined system that runs both real trading and paper trading simultaneously:
- Real trading bot for actual trades
- Paper trading engine for AI learning
- Synchronized strategies and signals
- Comprehensive performance comparison
"""

import sys
import json
import logging
import os
import time
import threading
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from flask import Flask, render_template_string, jsonify, request

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

# Import trading engines
from trading_bot_engine import TradingBotEngine
from paper_trading_dashboard import PaperTradingEngine, MarketData

logger = logging.getLogger(__name__)
app = Flask(__name__)

@dataclass
class TradingCosts:
    """Trading cost constants based on Indian broker charges"""
    LOT_SIZE: int = 75
    BROKERAGE_BUY: float = 20.0
    BROKERAGE_SELL: float = 20.0
    STT_RATE: float = 0.0025  # 0.25% on sell side
    EXCHANGE_CHARGES: float = 0.00053  # ~0.053%
    GST_RATE: float = 0.18  # 18% on brokerage + exchange charges
    SEBI_FEES: float = 0.0001  # 0.01%

@dataclass
class ProfitTarget:
    """Profit target calculation result"""
    capital: float
    daily_target_percent: float
    daily_target_amount: float
    max_lots: int
    total_quantity: int
    actual_investment: float
    remaining_balance: float
    min_points_needed: Optional[float]
    recommended_points: Optional[float]
    is_achievable: bool

class ProfitCalculator:
    """Intelligent profit calculator with real trading costs"""
    
    def __init__(self):
        self.costs = TradingCosts()
    
    def calculate_trading_costs(self, sell_value: float) -> Dict[str, float]:
        """Calculate all trading costs for a given sell value"""
        stt = sell_value * self.costs.STT_RATE
        exchange_charges = sell_value * self.costs.EXCHANGE_CHARGES
        total_charges_before_gst = self.costs.BROKERAGE_BUY + self.costs.BROKERAGE_SELL + exchange_charges
        gst = total_charges_before_gst * self.costs.GST_RATE
        sebi_charges = sell_value * self.costs.SEBI_FEES
        
        total_charges = self.costs.BROKERAGE_BUY + self.costs.BROKERAGE_SELL + stt + exchange_charges + gst + sebi_charges
        
        return {
            'brokerage_buy': self.costs.BROKERAGE_BUY,
            'brokerage_sell': self.costs.BROKERAGE_SELL,
            'stt': stt,
            'exchange_charges': exchange_charges,
            'gst': gst,
            'sebi_charges': sebi_charges,
            'total_charges': total_charges
        }
    
    def calculate_optimal_target(self, capital: float, premium: float, daily_target_percent: float) -> ProfitTarget:
        """Calculate optimal trading target with cost analysis"""
        premium_per_lot = premium * self.costs.LOT_SIZE
        max_lots = math.floor(capital / premium_per_lot)
        actual_investment = max_lots * premium_per_lot
        remaining_balance = capital - actual_investment
        total_quantity = max_lots * self.costs.LOT_SIZE
        daily_target_amount = (capital * daily_target_percent) / 100
        
        # Find minimum points needed
        min_points_needed = None
        for points in [x * 0.5 for x in range(20, 101)]:  # 10 to 50 points in 0.5 increments
            selling_price = premium + points
            sell_value = selling_price * total_quantity
            
            costs = self.calculate_trading_costs(sell_value)
            gross_profit = sell_value - (premium * total_quantity)
            net_profit = gross_profit - costs['total_charges']
            
            if net_profit >= daily_target_amount and min_points_needed is None:
                min_points_needed = points
                break
        
        recommended_points = min_points_needed + 3 if min_points_needed else None
        is_achievable = min_points_needed is not None and min_points_needed <= 25  # Realistic limit
        
        return ProfitTarget(
            capital=capital,
            daily_target_percent=daily_target_percent,
            daily_target_amount=daily_target_amount,
            max_lots=max_lots,
            total_quantity=total_quantity,
            actual_investment=actual_investment,
            remaining_balance=remaining_balance,
            min_points_needed=min_points_needed,
            recommended_points=recommended_points,
            is_achievable=is_achievable
        )
    
    def calculate_points_analysis(self, capital: float, premium: float, daily_target_amount: float) -> List[Dict]:
        """Generate points analysis table"""
        premium_per_lot = premium * self.costs.LOT_SIZE
        max_lots = math.floor(capital / premium_per_lot)
        total_quantity = max_lots * self.costs.LOT_SIZE
        
        analysis = []
        for points in range(10, 51, 5):  # 10 to 50 points in 5-point increments
            selling_price = premium + points
            sell_value = selling_price * total_quantity
            
            costs = self.calculate_trading_costs(sell_value)
            gross_profit = sell_value - (premium * total_quantity)
            net_profit = gross_profit - costs['total_charges']
            capital_gain_percent = (net_profit / capital) * 100
            achieves_target = net_profit >= daily_target_amount
            
            analysis.append({
                'points': points,
                'selling_price': selling_price,
                'gross_profit': gross_profit,
                'total_charges': costs['total_charges'],
                'net_profit': net_profit,
                'capital_gain_percent': capital_gain_percent,
                'achieves_target': achieves_target
            })
        
        return analysis
    
    def calculate_compounding(self, starting_capital: float, daily_percent: float, working_days: int, months: int) -> Dict:
        """Calculate compounding projections"""
        total_days = working_days * months
        balance = starting_capital
        daily_data = []
        
        # Add starting point
        daily_data.append({
            'day': 0,
            'profit': 0,
            'balance': starting_capital,
            'period': 1
        })
        
        # Calculate daily compounding
        for day in range(1, total_days + 1):
            daily_profit = balance * (daily_percent / 100)
            balance = balance + daily_profit
            
            daily_data.append({
                'day': day,
                'profit': daily_profit,
                'balance': balance,
                'period': math.ceil(day / working_days)
            })
        
        final_balance = balance
        total_profit = final_balance - starting_capital
        growth_rate = (total_profit / starting_capital) * 100
        
        return {
            'starting_balance': starting_capital,
            'final_balance': final_balance,
            'total_profit': total_profit,
            'growth_rate': growth_rate,
            'daily_data': daily_data,
            'total_days': total_days
        }

class IntegratedTradingSystem:
    """
    Integrated system that manages both real and paper trading
    """
    
    def __init__(self, dhan_client=None):
        self.dhan_client = dhan_client
        self.profit_calculator = ProfitCalculator()
        
        # Initialize bot configs with intelligent targeting
        self.real_bot_config = {
            'strategy': '15min_Breakout',
            'capital': 100000,  # 1 Lakh real capital
            'max_positions': 2,
            'risk_level': 'Medium',
            'stop_loss': 2.0,
            'target_profit': 5.0
        }
        
        self.paper_bot_config = {
            'strategy': '15min_Breakout',
            'capital': 1000000,  # 10 Lakh virtual capital
            'max_positions': 5,  # More aggressive in paper trading
            'risk_level': 'High',
            'stop_loss': 2.0,
            'target_profit': 5.0
        }
        
        # Initialize engines
        self.real_trading_bot = None
        self.paper_trading_engine = PaperTradingEngine(initial_capital=1000000)
        
        if dhan_client:
            self.real_trading_bot = TradingBotEngine(self.real_bot_config, dhan_client)
        
        self.is_running = False
        self.current_market_data = None
        
        # Performance comparison
        self.comparison_metrics = {
            'real_trades': 0,
            'paper_trades': 0,
            'real_pnl': 0.0,
            'paper_pnl': 0.0,
            'real_win_rate': 0.0,
            'paper_win_rate': 0.0
        }
        
        logger.info("Integrated Trading System initialized with intelligent profit calculator")
    
    def get_intelligent_targets(self, capital: float, premium: float, daily_target: float = 10.0) -> Dict:
        """Get intelligent profit targets based on current market conditions"""
        target = self.profit_calculator.calculate_optimal_target(capital, premium, daily_target)
        analysis = self.profit_calculator.calculate_points_analysis(capital, premium, target.daily_target_amount)
        
        return {
            'target': target.__dict__,
            'analysis': analysis,
            'recommendations': self._generate_recommendations(target)
        }
    
    def _generate_recommendations(self, target: ProfitTarget) -> List[str]:
        """Generate intelligent trading recommendations"""
        recommendations = []
        
        if not target.is_achievable:
            recommendations.append("Target is too aggressive for available capital - consider reducing to 5-8%")
        
        if target.min_points_needed and target.min_points_needed > 20:
            recommendations.append("High points requirement - focus on volatile market conditions")
        
        if target.remaining_balance < target.actual_investment * 0.1:
            recommendations.append("Low remaining balance - consider increasing capital for safety")
        
        if target.max_lots < 2:
            recommendations.append("Limited lots available - consider increasing capital or lower premium options")
        
        if target.total_quantity > 0:
            premium_per_unit = target.actual_investment / target.total_quantity
            recommendations.append(f"Set stop loss at 30-50% of premium (₹{premium_per_unit * 0.3:.0f}-{premium_per_unit * 0.5:.0f})")
        else:
            recommendations.append("Increase capital to enable trading - insufficient funds for even 1 lot")
        
        return recommendations
    
    def _get_atm_premium(self) -> Optional[float]:
        """Get ATM option premium for calculations"""
        try:
            nifty_price = self.current_market_data.nifty_price
            for option in self.current_market_data.option_chain:
                if abs(option['strike_price'] - nifty_price) < 50:
                    return max(option.get('ce_ltp', 0), option.get('pe_ltp', 0))
            return 100  # Default fallback
        except:
            return 100

    def start_integrated_system(self):
        """Start both real and paper trading systems"""
        try:
            if self.is_running:
                return False, "System already running"
            
            self.is_running = True
            
            # Start real trading bot if available
            if self.real_trading_bot:
                success, message = self.real_trading_bot.start()
                if success:
                    logger.info("✅ Real trading bot started")
                else:
                    logger.warning(f"❌ Real trading bot failed to start: {message}")
            
            # Start integrated trading loop
            self.trading_thread = threading.Thread(target=self._integrated_trading_loop, daemon=True)
            self.trading_thread.start()
            
            logger.info("🚀 INTEGRATED TRADING SYSTEM STARTED")
            logger.info("   📈 Real Trading: Active" if self.real_trading_bot else "   📈 Real Trading: Disabled")
            logger.info("   📄 Paper Trading: Active")
            logger.info("   🤖 AI Learning: Active")
            
            return True, "Integrated system started successfully"
            
        except Exception as e:
            logger.error(f"Error starting integrated system: {e}")
            return False, str(e)
    
    def stop_integrated_system(self):
        """Stop both trading systems"""
        try:
            if not self.is_running:
                return False, "System not running"
            
            self.is_running = False
            
            # Stop real trading bot
            if self.real_trading_bot:
                success, message = self.real_trading_bot.stop()
                logger.info(f"Real trading bot stopped: {message}")
            
            logger.info("🛑 INTEGRATED TRADING SYSTEM STOPPED")
            return True, "Integrated system stopped successfully"
            
        except Exception as e:
            logger.error(f"Error stopping integrated system: {e}")
            return False, str(e)
    
    def _integrated_trading_loop(self):
        """Main integrated trading loop"""
        logger.info("🔄 Integrated trading loop started")
        
        while self.is_running:
            try:
                # Update market data
                self._update_market_data()
                
                if not self.current_market_data:
                    time.sleep(30)
                    continue
                
                # Get intelligent targets for current market
                if self.current_market_data.option_chain:
                    atm_premium = self._get_atm_premium()
                    if atm_premium:
                        targets = self.get_intelligent_targets(
                            capital=self.real_bot_config['capital'],
                            premium=atm_premium,
                            daily_target=10.0
                        )
                        
                        # Adjust strategy based on intelligent analysis
                        if targets['target']['is_achievable']:
                            signals = self._generate_intelligent_signals(targets)
                            
                            # Execute signals
                            for signal in signals:
                                if not self.is_running:
                                    break
                                
                                # Execute in paper trading (always)
                                paper_success = self.paper_trading_engine.execute_paper_trade(
                                    signal, self.current_market_data
                                )
                                
                                if paper_success:
                                    logger.info(f"📄 Paper trade executed: {signal['symbol']}")
                                
                                # Execute in real trading (if enabled and conditions met)
                                if self.real_trading_bot and self._should_execute_real_trade(signal, targets):
                                    try:
                                        if self.real_trading_bot._should_execute_trade(signal):
                                            self.real_trading_bot._execute_trade(signal)
                                            logger.info(f"💰 Real trade executed: {signal['symbol']}")
                                    except Exception as e:
                                        logger.error(f"Error executing real trade: {e}")
                        else:
                            logger.warning("Daily target not achievable with current market conditions")
                
                # Monitor positions
                self.paper_trading_engine.monitor_paper_positions(self.current_market_data)
                
                if self.real_trading_bot:
                    self.real_trading_bot._monitor_positions()
                
                # Update comparison metrics
                self._update_comparison_metrics()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in integrated trading loop: {e}")
                time.sleep(60)
        
        logger.info("🔄 Integrated trading loop stopped")
    
    def _update_market_data(self):
        """Update current market data"""
        try:
            if not self.dhan_client:
                return
            
            # Get NIFTY data
            import yfinance as yf
            nifty_ticker = yf.Ticker("^NSEI")
            nifty_data = nifty_ticker.history(period="1d", interval="1m")
            
            if nifty_data.empty:
                return
            
            current_price = float(nifty_data['Close'].iloc[-1])
            prev_close = float(nifty_data['Close'].iloc[0])
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            # Get option chain
            option_chain_data = self.dhan_client.get_option_chain(underlying_scrip=13)
            option_chain = []
            
            if option_chain_data:
                for option in option_chain_data:
                    option_chain.append({
                        'strike_price': option.strike_price,
                        'option_type': option.option_type,
                        'ltp': option.ltp,
                        'volume': option.volume,
                        'open_interest': option.open_interest
                    })
            
            # Market status
            now = datetime.now()
            weekday = now.weekday()
            
            if weekday >= 5:
                market_status = 'CLOSED'
            elif (now.hour == 9 and now.minute >= 15) or (10 <= now.hour < 15) or (now.hour == 15 and now.minute < 30):
                market_status = 'OPEN'
            else:
                market_status = 'CLOSED'
            
            self.current_market_data = MarketData(
                timestamp=datetime.now(),
                nifty_price=current_price,
                nifty_change=change,
                nifty_change_percent=change_percent,
                option_chain=option_chain,
                market_status=market_status
            )
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def _generate_intelligent_signals(self, targets: Dict) -> List[Dict]:
        """Generate signals based on intelligent analysis"""
        signals = []
        
        if not self.real_trading_bot:
            return signals
        
        # Use regular signal generation but adjust quantities based on intelligent analysis
        base_signals = self.real_trading_bot._generate_trading_signals()
        
        for signal in base_signals:
            # Adjust quantity based on intelligent lot calculation
            max_lots = targets['target']['max_lots']
            if max_lots > 0:
                signal['quantity'] = min(signal['quantity'], max_lots * 75)
                signal['intelligent_target'] = targets['target']['min_points_needed']
                signal['recommended_points'] = targets['target']['recommended_points']
                signals.append(signal)
        
        # Add paper-specific signals for more aggressive learning
        if self.current_market_data.market_status == 'OPEN':
            paper_signals = self._generate_paper_specific_signals()
            signals.extend(paper_signals)
        
        return signals
    
    def _generate_integrated_signals(self):
        """Generate trading signals for both systems (fallback method)"""
        try:
            if not self.real_trading_bot:
                return []
            
            # Use the real trading bot's signal generation
            signals = self.real_trading_bot._generate_trading_signals()
            
            # Add paper-specific signals for more aggressive learning
            if self.current_market_data.market_status == 'OPEN':
                paper_signals = self._generate_paper_specific_signals()
                signals.extend(paper_signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating integrated signals: {e}")
            return []
    
    def _generate_paper_specific_signals(self):
        """Generate additional signals for paper trading (more aggressive)"""
        signals = []
        
        try:
            # More aggressive paper trading signals for learning
            change_percent = self.current_market_data.nifty_change_percent
            
            # Micro movements for learning
            if abs(change_percent) > 0.05:  # Very small movements
                atm_strike = self._find_atm_strike(self.current_market_data.nifty_price)
                
                if change_percent > 0.05:
                    signals.append({
                        'type': 'BUY',
                        'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}CE",
                        'quantity': 75,  # 1 lot
                        'strategy': 'Paper Micro Trend',
                        'confidence': 0.6,
                        'target_percent': 5,
                        'stop_loss_percent': 3
                    })
                elif change_percent < -0.05:
                    signals.append({
                        'type': 'BUY',
                        'symbol': f"NIFTY{self._get_expiry_string()}{int(atm_strike)}PE",
                        'quantity': 75,  # 1 lot
                        'strategy': 'Paper Micro Trend',
                        'confidence': 0.6,
                        'target_percent': 5,
                        'stop_loss_percent': 3
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating paper-specific signals: {e}")
            return []
    
    def _find_atm_strike(self, nifty_price):
        """Find ATM strike price"""
        if not self.current_market_data.option_chain:
            return round(nifty_price / 50) * 50  # Round to nearest 50
        
        strikes = [opt['strike_price'] for opt in self.current_market_data.option_chain]
        return min(strikes, key=lambda x: abs(x - nifty_price))
    
    def _get_expiry_string(self):
        """Get next weekly expiry string"""
        today = datetime.now()
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        
        expiry_date = today + timedelta(days=days_ahead)
        return expiry_date.strftime("%Y%m%d")
    
    def _should_execute_real_trade(self, signal, targets=None):
        """Enhanced trade execution logic"""
        # If targets provided, use intelligent logic
        if targets:
            # Only execute if target is achievable
            if not targets['target']['is_achievable']:
                return False
            
            # Only execute high-confidence signals
            if signal.get('confidence', 0) < 0.8:
                return False
            
            # Check if we have sufficient capital buffer
            if targets['target']['remaining_balance'] < targets['target']['actual_investment'] * 0.1:
                return False
            
            return True
        
        # Fallback to original logic
        return signal.get('confidence', 0) >= 0.8 and signal.get('strategy') != 'Paper Micro Trend'
    
    def _update_comparison_metrics(self):
        """Update performance comparison metrics"""
        try:
            # Get paper trading performance
            paper_perf = self.paper_trading_engine.get_performance_summary()
            
            self.comparison_metrics['paper_trades'] = paper_perf['total_trades']
            self.comparison_metrics['paper_pnl'] = paper_perf['total_pnl']
            self.comparison_metrics['paper_win_rate'] = paper_perf['win_rate']
            
            # Get real trading performance
            if self.real_trading_bot:
                real_status = self.real_trading_bot.get_status()
                self.comparison_metrics['real_trades'] = real_status['trades_today']
                self.comparison_metrics['real_pnl'] = real_status['pnl_today']
                self.comparison_metrics['real_win_rate'] = real_status['success_rate']
            
        except Exception as e:
            logger.error(f"Error updating comparison metrics: {e}")
    
    def get_integrated_status(self):
        """Get comprehensive system status with intelligent targets"""
        try:
            paper_perf = self.paper_trading_engine.get_performance_summary()
            real_status = self.real_trading_bot.get_status() if self.real_trading_bot else {}
            
            # Get current intelligent targets
            atm_premium = self._get_atm_premium() if self.current_market_data else 100
            targets = self.get_intelligent_targets(
                capital=self.real_bot_config['capital'],
                premium=atm_premium,
                daily_target=10.0
            )
            
            return {
                'system_running': self.is_running,
                'real_trading': {
                    'enabled': self.real_trading_bot is not None,
                    'running': real_status.get('running', False),
                    'trades_today': real_status.get('trades_today', 0),
                    'pnl_today': real_status.get('pnl_today', 0),
                    'success_rate': real_status.get('success_rate', 0),
                    'active_positions': real_status.get('active_positions', 0)
                },
                'paper_trading': {
                    'enabled': True,
                    'total_capital': paper_perf['total_capital'],
                    'current_capital': paper_perf['current_capital'],
                    'total_pnl': paper_perf['total_pnl'],
                    'daily_pnl': paper_perf['daily_pnl'],
                    'total_trades': paper_perf['total_trades'],
                    'win_rate': paper_perf['win_rate'],
                    'open_positions': paper_perf['open_positions']
                },
                'market_data': {
                    'nifty_price': self.current_market_data.nifty_price if self.current_market_data else 0,
                    'nifty_change_percent': self.current_market_data.nifty_change_percent if self.current_market_data else 0,
                    'market_status': self.current_market_data.market_status if self.current_market_data else 'UNKNOWN'
                },
                'intelligent_targets': targets,
                'comparison_metrics': self.comparison_metrics,
                'ai_learning': paper_perf['learning_metrics'],
                'strategy_performance': paper_perf['strategy_performance']
            }
            
        except Exception as e:
            logger.error(f"Error getting integrated status: {e}")
            return {}

# Global integrated system
integrated_system = None

# Initialize Dhan client
dhan_client = None
if DHAN_AVAILABLE:
    try:
        credentials = DhanCredentials(
            client_id=os.getenv('DHAN_CLIENT_ID', '1107321060'),
            access_token=os.getenv('DHAN_ACCESS_TOKEN')
        )
        dhan_client = DhanAPIClient(credentials)
        if dhan_client.authenticate():
            print("Dhan API connected for integrated system")
        else:
            dhan_client = None
    except Exception as e:
        print(f"Error connecting to Dhan API: {e}")
        dhan_client = None

# Initialize integrated system
integrated_system = IntegratedTradingSystem(dhan_client)

# HTML Template for Integrated Dashboard
INTEGRATED_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔗 Integrated Trading System</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        
        .system-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }
        
        .btn-start { background: #4CAF50; color: white; }
        .btn-stop { background: #f44336; color: white; }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .trading-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .trading-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .section-title {
            font-size: 1.6rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .real-trading { border-left: 4px solid #4CAF50; }
        .paper-trading { border-left: 4px solid #2196F3; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-item {
            text-align: center;
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 5px 0;
        }
        
        .metric-label {
            font-size: 0.8rem;
            opacity: 0.8;
            text-transform: uppercase;
        }
        
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        
        .comparison-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .comparison-table th {
            background: rgba(255,255,255,0.1);
            font-weight: 600;
        }
        
        .profit-section {
            background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            margin-bottom: 30px;
        }
        
        .calculator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .input-group {
            display: flex;
            flex-direction: column;
        }
        
        .input-group label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #ffffff;
        }
        
        .input-group input {
            padding: 12px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 8px;
            font-size: 16px;
            background: rgba(255,255,255,0.1);
            color: white;
        }
        
        .input-group input:focus {
            outline: none;
            border-color: #ffffff;
        }
        
        .recommendations {
            background: rgba(255,193,7,0.2);
            border-left: 4px solid #FFC107;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        
        .recommendations h3 {
            color: #FFC107;
            margin-bottom: 10px;
        }
        
        .recommendations ul {
            list-style: none;
            padding-left: 0;
        }
        
        .recommendations li {
            margin-bottom: 8px;
            padding-left: 20px;
            position: relative;
        }
        
        .recommendations li:before {
            content: "⚡";
            position: absolute;
            left: 0;
        }
        
        .analysis-table {
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            overflow: hidden;
            margin-top: 20px;
        }
        
        .analysis-table th,
        .analysis-table td {
            padding: 12px 8px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            font-size: 0.9rem;
        }
        
        .analysis-table th {
            background: rgba(255,255,255,0.2);
            font-weight: 600;
        }
        
        .target-met { color: #4CAF50; font-weight: bold; }
        .target-not-met { color: #ff9800; font-weight: bold; }

        @media (max-width: 768px) {
            .trading-grid { grid-template-columns: 1fr; }
            .calculator-grid { grid-template-columns: 1fr; }
            .system-controls { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔗 Integrated Trading System</h1>
            <p style="opacity: 0.8; margin-top: 10px;">Real Trading + Paper Trading + AI Learning</p>
            
            <div class="system-controls">
                <button id="startSystem" class="btn btn-start">
                    <i class="fas fa-play"></i> Start System
                </button>
                <button id="stopSystem" class="btn btn-stop" disabled>
                    <i class="fas fa-stop"></i> Stop System
                </button>
            </div>
            
            <div id="systemStatus" style="margin-top: 15px;">
                <span id="statusIndicator">🔴</span>
                <span id="statusText">System Stopped</span>
            </div>
        </div>

        <!-- Intelligent Profit Calculator Section -->
        <div class="trading-section profit-section">
            <div class="section-title">
                <i class="fas fa-calculator"></i>
                Intelligent Profit Calculator
            </div>
            
            <div class="calculator-grid">
                <div class="input-group">
                    <label for="capital">Available Capital (₹)</label>
                    <input type="number" id="capital" value="100000" min="10000" step="1000">
                </div>
                <div class="input-group">
                    <label for="dailyTarget">Daily Target (%)</label>
                    <input type="number" id="dailyTarget" value="10" min="1" max="50" step="1">
                </div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Daily Target Amount</div>
                    <div class="metric-value" id="dailyTargetAmount">₹0</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Max Lots</div>
                    <div class="metric-value" id="maxLots">0</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Investment Required</div>
                    <div class="metric-value" id="actualInvestment">₹0</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Min Points Needed</div>
                    <div class="metric-value" id="minPointsNeeded">-</div>
                </div>
            </div>
            
            <div id="recommendations" class="recommendations" style="display: none;">
                <h3><i class="fas fa-lightbulb"></i> AI Recommendations</h3>
                <ul id="recommendationsList"></ul>
            </div>
        </div>

        <div class="trading-grid">
            <!-- Real Trading Section -->
            <div class="trading-section real-trading">
                <div class="section-title">
                    <i class="fas fa-coins"></i>
                    Real Trading
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Status</div>
                        <div class="metric-value" id="realStatus">Stopped</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Trades Today</div>
                        <div class="metric-value" id="realTrades">0</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Today's P&L</div>
                        <div class="metric-value" id="realPnl">₹0</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value" id="realWinRate">0%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Positions</div>
                        <div class="metric-value" id="realPositions">0</div>
                    </div>
                </div>
            </div>

            <!-- Paper Trading Section -->
            <div class="trading-section paper-trading">
                <div class="section-title">
                    <i class="fas fa-file-alt"></i>
                    Paper Trading
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Virtual Capital</div>
                        <div class="metric-value" id="paperCapital">₹10,00,000</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value" id="paperTrades">0</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Total P&L</div>
                        <div class="metric-value" id="paperPnl">₹0</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value" id="paperWinRate">0%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Positions</div>
                        <div class="metric-value" id="paperPositions">0</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="comparison-section">
            <div class="section-title">
                <i class="fas fa-chart-bar"></i>
                Performance Comparison
            </div>
            
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Real Trading</th>
                        <th>Paper Trading</th>
                        <th>Difference</th>
                    </tr>
                </thead>
                <tbody id="comparisonTableBody">
                    <tr>
                        <td>Total Trades</td>
                        <td id="compRealTrades">0</td>
                        <td id="compPaperTrades">0</td>
                        <td id="compTradesDiff">0</td>
                    </tr>
                    <tr>
                        <td>P&L</td>
                        <td id="compRealPnl">₹0</td>
                        <td id="compPaperPnl">₹0</td>
                        <td id="compPnlDiff">₹0</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td id="compRealWin">0%</td>
                        <td id="compPaperWin">0%</td>
                        <td id="compWinDiff">0%</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Points Analysis Table -->
        <div class="trading-section">
            <div class="section-title">
                <i class="fas fa-chart-bar"></i>
                Points Analysis (Live Market Data)
            </div>
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Points</th>
                        <th>Selling Price</th>
                        <th>Gross Profit</th>
                        <th>Net Profit</th>
                        <th>Gain %</th>
                        <th>Target</th>
                    </tr>
                </thead>
                <tbody id="analysisTableBody">
                    <tr><td colspan="6">Loading market data...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let currentTargets = null;
        
        function formatCurrency(amount) {
            return new Intl.NumberFormat('en-IN', {
                style: 'currency',
                currency: 'INR',
                maximumFractionDigits: 0
            }).format(amount);
        }
        
        function updateProfitCalculator() {
            const capital = parseFloat($('#capital').val());
            const dailyTarget = parseFloat($('#dailyTarget').val());
            
            $.get('/api/profit_targets', {
                capital: capital,
                daily_target: dailyTarget
            }).done(function(data) {
                currentTargets = data;
                updateCalculatorDisplay(data);
                updatePointsAnalysis(data);
                updateRecommendations(data);
            });
        }
        
        function updateCalculatorDisplay(data) {
            const target = data.target;
            
            $('#dailyTargetAmount').text(formatCurrency(target.daily_target_amount));
            $('#maxLots').text(target.max_lots + ' lots');
            $('#actualInvestment').text(formatCurrency(target.actual_investment));
            $('#minPointsNeeded').text(target.min_points_needed ? target.min_points_needed + ' pts' : 'N/A');
        }
        
        function updatePointsAnalysis(data) {
            const tbody = $('#analysisTableBody');
            tbody.empty();
            
            data.analysis.forEach(row => {
                const tr = $('<tr>');
                if (row.achieves_target) {
                    tr.addClass('target-achieved');
                }
                
                tr.html(`
                    <td><strong>+${row.points}</strong></td>
                    <td>₹${row.selling_price}</td>
                    <td class="positive">${formatCurrency(row.gross_profit)}</td>
                    <td class="${row.net_profit > 0 ? 'positive' : 'negative'}">${formatCurrency(row.net_profit)}</td>
                    <td class="${row.capital_gain_percent >= parseFloat($('#dailyTarget').val()) ? 'target-met' : 'target-not-met'}">${row.capital_gain_percent.toFixed(1)}%</td>
                    <td>${row.achieves_target ? '<span class="target-met">✅</span>' : '<span class="target-not-met">❌</span>'}</td>
                `);
                tbody.append(tr);
            });
        }
        
        function updateRecommendations(data) {
            const recList = $('#recommendationsList');
            recList.empty();
            
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(rec => {
                    recList.append(`<li>${rec}</li>`);
                });
                $('#recommendations').show();
            } else {
                $('#recommendations').hide();
            }
        }

        function fetchSystemStatus() {
            $.get('/api/integrated_status')
                .done(function(data) {
                    updateSystemStatus(data);
                })
                .fail(function() {
                    console.log('Failed to fetch system status');
                });
        }
        
        function updateSystemStatus(data) {
            // Update system status
            if (data.system_running) {
                $('#statusIndicator').text('🟢');
                $('#statusText').text('System Running');
                $('#startSystem').prop('disabled', true);
                $('#stopSystem').prop('disabled', false);
            } else {
                $('#statusIndicator').text('🔴');
                $('#statusText').text('System Stopped');
                $('#startSystem').prop('disabled', false);
                $('#stopSystem').prop('disabled', true);
            }
            
            // Update real trading metrics
            if (data.real_trading) {
                const real = data.real_trading;
                $('#realStatus').text(real.running ? 'Running' : 'Stopped');
                $('#realTrades').text(real.trades_today);
                $('#realPnl').text('₹' + real.pnl_today.toLocaleString('en-IN'));
                $('#realPnl').removeClass('positive negative').addClass(real.pnl_today >= 0 ? 'positive' : 'negative');
                $('#realWinRate').text(real.success_rate.toFixed(1) + '%');
                $('#realPositions').text(real.active_positions);
            }
            
            // Update paper trading metrics
            if (data.paper_trading) {
                const paper = data.paper_trading;
                $('#paperCapital').text('₹' + paper.current_capital.toLocaleString('en-IN'));
                $('#paperTrades').text(paper.total_trades);
                $('#paperPnl').text('₹' + paper.total_pnl.toLocaleString('en-IN'));
                $('#paperPnl').removeClass('positive negative').addClass(paper.total_pnl >= 0 ? 'positive' : 'negative');
                $('#paperWinRate').text(paper.win_rate.toFixed(1) + '%');
                $('#paperPositions').text(paper.open_positions);
            }
            
            // Update comparison table
            if (data.comparison_metrics) {
                const comp = data.comparison_metrics;
                $('#compRealTrades').text(comp.real_trades);
                $('#compPaperTrades').text(comp.paper_trades);
                $('#compTradesDiff').text(comp.paper_trades - comp.real_trades);
                
                $('#compRealPnl').text('₹' + comp.real_pnl.toLocaleString('en-IN'));
                $('#compPaperPnl').text('₹' + comp.paper_pnl.toLocaleString('en-IN'));
                const pnlDiff = comp.paper_pnl - comp.real_pnl;
                $('#compPnlDiff').text('₹' + pnlDiff.toLocaleString('en-IN'));
                $('#compPnlDiff').removeClass('positive negative').addClass(pnlDiff >= 0 ? 'positive' : 'negative');
                
                $('#compRealWin').text(comp.real_win_rate.toFixed(1) + '%');
                $('#compPaperWin').text(comp.paper_win_rate.toFixed(1) + '%');
                const winDiff = comp.paper_win_rate - comp.real_win_rate;
                $('#compWinDiff').text(winDiff.toFixed(1) + '%');
                $('#compWinDiff').removeClass('positive negative').addClass(winDiff >= 0 ? 'positive' : 'negative');
            }
        }
        
        $('#startSystem').click(function() {
            $.post('/api/start_integrated_system')
                .done(function(response) {
                    if (response.success) {
                        alert('Integrated system started successfully!');
                    } else {
                        alert('Failed to start system: ' + response.message);
                    }
                });
        });
        
        $('#stopSystem').click(function() {
            $.post('/api/stop_integrated_system')
                .done(function(response) {
                    if (response.success) {
                        alert('Integrated system stopped successfully!');
                    } else {
                        alert('Failed to stop system: ' + response.message);
                    }
                });
        });
        
        // Event listeners for profit calculator
        $('#capital, #dailyTarget').on('input', updateProfitCalculator);
        
        // Initialize
        $(document).ready(function() {
            updateProfitCalculator();
            fetchSystemStatus();
            setInterval(fetchSystemStatus, 3000);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def integrated_dashboard():
    """Main integrated dashboard route"""
    return render_template_string(INTEGRATED_DASHBOARD_HTML)

@app.route('/api/integrated_status')
def get_integrated_status():
    """Get integrated system status"""
    try:
        status = integrated_system.get_integrated_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting integrated status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_integrated_system', methods=['POST'])
def start_integrated_system():
    """Start the integrated trading system"""
    try:
        success, message = integrated_system.start_integrated_system()
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        logger.error(f"Error starting integrated system: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop_integrated_system', methods=['POST'])
def stop_integrated_system():
    """Stop the integrated trading system"""
    try:
        success, message = integrated_system.stop_integrated_system()
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        logger.error(f"Error stopping integrated system: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/profit_targets')
def get_profit_targets():
    """Get intelligent profit targets"""
    try:
        capital = float(request.args.get('capital', 100000))
        daily_target = float(request.args.get('daily_target', 10))
        
        # Get current ATM premium from market data
        atm_premium = integrated_system._get_atm_premium() if integrated_system.current_market_data else 100
        
        targets = integrated_system.get_intelligent_targets(capital, atm_premium, daily_target)
        return jsonify(targets)
    except Exception as e:
        logger.error(f"Error getting profit targets: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compounding')
def get_compounding():
    """Get compounding projections"""
    try:
        capital = float(request.args.get('capital', 100000))
        daily_percent = float(request.args.get('daily_percent', 10))
        working_days = int(request.args.get('working_days', 20))
        months = int(request.args.get('months', 1))
        
        compounding = integrated_system.profit_calculator.calculate_compounding(
            capital, daily_percent, working_days, months
        )
        return jsonify(compounding)
    except Exception as e:
        logger.error(f"Error getting compounding: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("ENHANCED AI TRADING SYSTEM")
    print("=" * 60)
    print("Features:")
    print("   * Real trading with intelligent targeting")
    print("   * Paper trading for AI learning")
    print("   * Live profit calculator with real costs")
    print("   * Compounding projections")
    print("   * Optimal position sizing")
    print("   * Smart risk management")
    print("   * Performance comparison analysis")
    print()
    print("Real Capital: Configured amount")
    print("Virtual Capital: Rs.10,00,000")
    print("Dashboard: http://localhost:5005")
    print("Market Data: Connected" if dhan_client else "Simulated")
    print("=" * 60)
    
    try:
        app.run(host='127.0.0.1', port=5005, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting integrated system: {e}")
    except KeyboardInterrupt:
        print("\n🔗 Integrated Trading System stopped")